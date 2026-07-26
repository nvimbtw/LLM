// Builds the preference catalogue: calibrate the judge against the replies a
// person rated, generate candidates, score every one of them, and write the
// verdicts out as preference training data.
//
// **Why calibration is the first stage and not an optional extra.** Hand-clicking
// thumbs produces a few dozen examples. A KTO run on a few dozen examples does not
// teach the model what good English looks like; it memorizes thirty replies and
// narrows onto them, which is exactly what happens when this stage is skipped. The
// catalogue is what makes the tune a tune — and the human ratings are what make
// the catalogue trustworthy, because they are the only thing that can answer
// "does this rule set agree with me?" before a thousand labels are written on its
// say-so.
//
// So the cycle is: rate a few dozen replies by hand → this job checks the judge
// against them and refuses if it disagrees → it labels thousands → the tune trains
// on those. The human ratings are the seed and the audit, never the training set.
//
// The judging rules live in `llm::autolabel` (they belong with the metrics they
// reuse); this module is the orchestration — where the prompts come from, how many
// candidates per prompt, and which file the result lands in. Same split as
// `training.rs`: the llm crate stays layout-agnostic, `storage.rs` is the only
// thing that knows the data layout.
//
// **The output goes only to the training directory.** `raw/preference/` is read by
// `training::build_kto_dataset`; the benchmark's held-out slice lives in
// `eval/preference/` and is written solely by the human-feedback export. Keeping
// auto-labels out of it is what makes the benchmark's preference margin an
// independent check: if a tune learns to game these rules rather than to write
// better replies, the human-rated margin will not move, and that divergence is
// the thing to look at.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::storage;

// The file auto-labels are written to. A distinct name from the human export's
// `feedback.jsonl` in the same directory, so the two accumulate side by side and
// neither run clobbers the other's work.
const OUTPUT_FILE: &str = "autolabel.jsonl";

// Default candidates per prompt. Enough spread for the best and worst to differ
// meaningfully; every extra candidate is another full generation, so this is the
// knob that sets the run's duration.
const DEFAULT_CANDIDATES: usize = 6;
const DEFAULT_MAX_PROMPTS: i64 = 200;
// Sampling temperature for the candidates. Deliberately above the chat default:
// the candidates have to actually differ from each other for a best/worst split to
// mean anything, and near-greedy decoding would produce six near-identical replies
// and a spread below the labelling margin.
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_TOP_K: usize = 40;
const DEFAULT_MAX_TOKENS: usize = 96;

#[derive(Deserialize, Serialize)]
pub struct AutolabelRequest {
    // Which model generates the candidates. Normally the one about to be tuned:
    // preference data is only meaningful for the policy that produced it.
    pub id_model: Uuid,
    #[serde(default = "default_candidates")]
    pub candidates: usize,
    #[serde(default = "default_max_prompts")]
    pub max_prompts: i64,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_candidates() -> usize {
    DEFAULT_CANDIDATES
}
fn default_max_prompts() -> i64 {
    DEFAULT_MAX_PROMPTS
}
fn default_temperature() -> f32 {
    DEFAULT_TEMPERATURE
}

#[derive(Serialize)]
pub struct AutolabelStats {
    pub prompts: usize,
    // What the human ratings said about the judge before anything was labelled.
    pub calibration: llm::autolabel::Calibration,
    pub examples_written: usize,
    // Split of what was written. Reported prominently because a one-sided
    // catalogue is the failure mode that gets all the way to the KTO run before
    // anything complains.
    pub approved: usize,
    pub rejected: usize,
    // Candidates that landed inside the uncertainty band and were dropped. Not a
    // failure — see `llm::autolabel::UNCERTAIN_BAND`.
    pub skipped_uncertain: usize,
    pub mean_score: f64,
}

pub type ProgressCallback = Box<dyn Fn(&str, f64) + Send>;
// Called once, before any labelling, with what the human ratings said about the
// judge. Reported live rather than only in the final stats because it is the
// number that decides whether the rest of the run is worth watching.
pub type CalibrationCallback = Box<dyn Fn(&llm::autolabel::Calibration) + Send>;
// Called with each verdict as it is decided, and the prompt it answered. The run
// is minutes long and writes nothing until it finishes, so without this the page
// would show a bar and nothing else while the catalogue takes shape.
pub type VerdictCallback = Box<dyn Fn(&str, &llm::autolabel::Verdict) + Send>;

// One line of the output, matching what `build_kto_dataset` and the benchmark's
// preference stage both read.
#[derive(Serialize)]
struct LabelledLine<'a> {
    prompt: &'a str,
    response: &'a str,
    label: &'a str,
}

// Runs the whole pass. Blocking and GPU-bound, so the caller gives it its own OS
// thread — the same constraint training and benchmarking have.
//
// `Ok(None)` means cancelled partway. Unlike a benchmark, a cancelled run still
// writes what it produced: the output is append-only training data rather than a
// score, so half a pass is half a dataset, which is useful, not meaningless.
pub fn run(
    model_dir: &Path,
    prompts: &[String],
    request: &AutolabelRequest,
    on_progress: ProgressCallback,
    on_calibration: CalibrationCallback,
    on_verdict: VerdictCallback,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<Option<AutolabelStats>, String> {
    let merges = tokenize::bpe::load_merges(&model_dir.join("merges.bin").to_string_lossy())
        .map_err(|e| format!("could not load tokenizer: {e}"))?;
    let model = llm::training::transformer::Model::load(&model_dir.to_string_lossy())
        .map_err(|e| format!("could not load model: {e}"))?;

    on_progress("Fitting the judge", 0.0);
    // The judge is fit on the same reference prose the benchmark scores against.
    // Reusing it rather than the training corpus is deliberate: the lexicon and the
    // n-gram statistics should describe English, not this particular model's diet.
    let reference = load_reference()?;
    let judge = llm::autolabel::Judge::build(&reference);

    on_progress("Calibrating against your ratings", 0.0);
    let calibration = calibrate(&judge)?;
    on_calibration(&calibration);

    let ctx_window = model.positional_table.len();
    let reserve = (ctx_window / 4).max(8);
    let candidates_per_prompt = request.candidates.clamp(2, 32);

    let mut out = String::new();
    let mut approved = 0usize;
    let mut rejected = 0usize;
    let mut skipped_uncertain = 0usize;
    let mut score_total = 0.0f64;

    for (i, prompt) in prompts.iter().enumerate() {
        if cancelled(&cancel) {
            // Write what we have before returning — see the note above.
            write_output(&out)?;
            return Ok(None);
        }
        on_progress(
            &format!("Generating ({}/{})", i + 1, prompts.len()),
            i as f64 / prompts.len() as f64,
        );

        let turns = [llm::chat_template::Turn {
            role: llm::chat_template::Role::User,
            text: prompt.clone(),
        }];
        let seq = llm::chat_template::build_prompt(None, &turns, &merges, ctx_window, reserve);

        let mut scored = Vec::with_capacity(candidates_per_prompt);
        for k in 0..candidates_per_prompt {
            let config = llm::eval::generate::SamplingConfig {
                max_tokens: DEFAULT_MAX_TOKENS,
                temperature: request.temperature,
                top_k: DEFAULT_TOP_K,
                // Seeded from the prompt index and the candidate index, so a rerun
                // over the same prompts reproduces the same candidates exactly —
                // otherwise a labelling pass could never be debugged.
                seed: (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k as u64,
            };
            let continuation = llm::eval::generate::continue_tokens_blocking(
                &model,
                &merges,
                seq.clone(),
                config,
            );
            let text = llm::autolabel::strip_terminal(&continuation.text);
            let score = judge.score(&text, continuation.stopped_on_marker);
            scored.push(llm::autolabel::Candidate { text, score });
        }

        // Every candidate gets a verdict against the calibrated threshold, not just
        // the best and the worst. Six candidates per prompt yield up to six
        // examples rather than two — which is the difference between a catalogue
        // and a handful of pairs — and a prompt where everything was bad can now
        // say so, instead of being forced to nominate one of them "best".
        let offered = scored.len();
        let verdicts = llm::autolabel::label_absolute(scored, &calibration);
        skipped_uncertain += offered - verdicts.len();

        for verdict in &verdicts {
            let line = LabelledLine {
                prompt,
                response: &verdict.text,
                label: if verdict.good { "up" } else { "down" },
            };
            let encoded = serde_json::to_string(&line)
                .map_err(|e| format!("could not encode a labelled example: {e}"))?;
            out.push_str(&encoded);
            out.push('\n');

            if verdict.good {
                approved += 1;
            } else {
                rejected += 1;
            }
            score_total += verdict.score;
            on_verdict(prompt, verdict);
        }
    }

    let examples_written = out.lines().count();
    write_output(&out)?;
    on_progress("Done", 1.0);

    Ok(Some(AutolabelStats {
        prompts: prompts.len(),
        calibration,
        examples_written,
        approved,
        rejected,
        skipped_uncertain,
        mean_score: mean(score_total, approved + rejected),
    }))
}

// The human ratings file. Only this one — never the auto-label output in the same
// directory, which would be the judge calibrating itself against its own verdicts
// and would report near-perfect agreement no matter how wrong it was.
const HUMAN_FILE: &str = "feedback.jsonl";

// Below this there is not enough human signal to place a threshold: the boundary
// would be fit to a handful of replies and would move wildly with one more rating.
const MIN_HUMAN_RATINGS: usize = 24;

// Scores the replies a person rated and fits the judge's decision threshold to
// them.
//
// **The format component is scored as though every human-rated reply terminated
// cleanly**, because `stopped_on_marker` is a property of generation and is not
// recoverable from stored text. That biases the threshold slightly upward on that
// one component, and the bias points the right way: a generated candidate that ran
// to the token cap is missing those points and falls below a boundary fit on
// replies that had them, so runaway replies get labelled bad. Which is correct —
// they are.
fn calibrate(judge: &llm::autolabel::Judge) -> Result<llm::autolabel::Calibration, String> {
    let path = storage::preference_dir().join(HUMAN_FILE);
    let content = std::fs::read_to_string(&path).map_err(|_| {
        format!(
            "no human ratings at {} — rate some replies first, so the judge has \
             something to be checked against",
            path.display()
        )
    })?;

    let mut approved = Vec::new();
    let mut rejected = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(parsed) = serde_json::from_str::<RatedLine>(line) else {
            continue;
        };
        if parsed.response.trim().is_empty() {
            continue;
        }
        let score = judge.score(&parsed.response, true).total;
        match parsed.label.as_str() {
            "up" => approved.push(score),
            "down" => rejected.push(score),
            _ => {}
        }
    }

    let total = approved.len() + rejected.len();
    if total < MIN_HUMAN_RATINGS {
        return Err(format!(
            "only {total} human ratings — rate at least {MIN_HUMAN_RATINGS} before \
             auto-labelling, or the threshold is fit to noise"
        ));
    }

    let calibration = llm::autolabel::calibrate(&approved, &rejected).ok_or_else(|| {
        format!(
            "your ratings are one-sided ({} approved, {} rejected) — the judge needs \
             examples of both to find the boundary between them",
            approved.len(),
            rejected.len()
        )
    })?;

    if !calibration.usable() {
        // Refusing is the whole point of the stage. Auto-labels are about to become
        // the overwhelming majority of the training data, so a judge that cannot
        // reproduce the ratings it *has* seen would spend a GPU hour writing a
        // catalogue of its own mistakes for the model to learn.
        return Err(format!(
            "the judge only agrees with your ratings {:.0}% of the time (needs {:.0}%). \
             Auto-labelling now would teach the model the judge's mistakes. Rate more \
             replies, or check that the reference corpus in eval/corpus matches the \
             kind of English you want.",
            calibration.auc * 100.0,
            llm::autolabel::MIN_AUC * 100.0,
        ));
    }

    Ok(calibration)
}

// One line of the human ratings file, as `queue::export_preference_data` writes it.
#[derive(Deserialize)]
struct RatedLine {
    response: String,
    label: String,
}

// How many auto-labelled examples are on disk, split by verdict.
//
// Counted from the file rather than tracked in a table because the file is the
// thing `build_kto_dataset` actually reads — a count kept anywhere else could say
// a thousand while the training run sees none, which is the class of mismatch this
// whole gate exists to prevent. Cheap: a few thousand short lines.
//
// A missing file is `(0, 0)`, not an error: no auto-labelling has run yet is the
// normal state before the first pass.
pub fn catalogue_counts() -> (i64, i64) {
    let path = storage::preference_dir().join(OUTPUT_FILE);
    let Ok(content) = std::fs::read_to_string(&path) else {
        return (0, 0);
    };

    let mut approved = 0i64;
    let mut rejected = 0i64;
    for line in content.lines() {
        let Ok(parsed) = serde_json::from_str::<RatedLine>(line.trim()) else {
            continue;
        };
        match parsed.label.as_str() {
            "up" => approved += 1,
            "down" => rejected += 1,
            _ => {}
        }
    }
    (approved, rejected)
}

// The reference prose the judge is fit on: the benchmark's held-out corpus dir.
fn load_reference() -> Result<String, String> {
    let dir = storage::eval_dir().join("corpus");
    let entries = std::fs::read_dir(&dir)
        .map_err(|e| format!("no reference corpus at {}: {e}", dir.display()))?;

    let mut paths: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("txt"))
        .collect();
    paths.sort();

    let mut text = String::new();
    for path in paths {
        if let Ok(content) = std::fs::read_to_string(&path) {
            text.push_str(&content);
            text.push('\n');
        }
    }
    if text.trim().is_empty() {
        // Without real English to compare against, the fluency and real-word
        // signals are meaningless and the labels would be close to random. Refuse
        // rather than produce a dataset that quietly teaches nothing.
        return Err(format!(
            "the reference corpus at {} is empty — the judge needs real English to score against",
            dir.display()
        ));
    }
    Ok(text)
}

// Appends to the auto-label file rather than replacing it, so successive runs
// accumulate. Unlike the human-feedback export (a projection of a table that can
// change), generated examples are immutable once judged — there is nothing to
// withdraw, and rewriting would throw away every earlier pass.
fn write_output(body: &str) -> Result<(), String> {
    use std::io::Write;

    if body.is_empty() {
        return Ok(());
    }
    let dir = storage::preference_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("could not create {dir:?}: {e}"))?;
    let path = dir.join(OUTPUT_FILE);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("could not open {path:?}: {e}"))?;
    file.write_all(body.as_bytes())
        .map_err(|e| format!("could not write {path:?}: {e}"))
}

fn cancelled(cancel: &Option<Arc<AtomicBool>>) -> bool {
    cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst))
}

#[cfg(test)]
mod tests {
    // Reports what the judge scores against the ratings currently on disk, without
    // queueing a job — calibration is pure CPU (an n-gram fit and some arithmetic),
    // so the number that decides whether auto-labelling is worth doing can be had
    // in a second rather than after a GPU pass.
    //
    // Ignored by default because it reads the live `data/` directory rather than a
    // fixture, so it is a diagnostic rather than a test of anything:
    //
    //     cargo test --bin backend -- --ignored --nocapture calibration_report
    #[test]
    #[ignore]
    fn calibration_report() {
        let reference = super::load_reference().expect("a reference corpus in data/eval/corpus");
        let judge = llm::autolabel::Judge::build(&reference);
        match super::calibrate(&judge) {
            Ok(c) => println!(
                "agreement (AUC) {:.3} over {} approved / {} rejected\n\
                 threshold {:.4}, separation {:.4} — {}",
                c.auc,
                c.approved,
                c.rejected,
                c.threshold,
                c.separation,
                if c.usable() { "usable" } else { "NOT usable" },
            ),
            Err(e) => println!("no calibration: {e}"),
        }
    }
}

fn mean(total: f64, n: usize) -> f64 {
    if n == 0 { f64::NAN } else { total / n as f64 }
}
