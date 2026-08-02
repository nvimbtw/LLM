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


// The file auto-labels are written to. A distinct name from the human export's
// `feedback.jsonl` in the same directory, so the two accumulate side by side and
// neither run clobbers the other's work.
const OUTPUT_FILE: &str = storage::AUTOLABEL_FILE;

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
// How long a candidate reply may run.
//
// **This was 96, and 96 was the reason the post-train cycle regressed models.** The
// assistant benchmark scores held-out replies averaging ~1400 characters (median
// 1149, i.e. ~295 tokens). A 96-token cap is ~265 characters at this vocab, and the
// longest reply in a 4,861-line catalogue was exactly 265 — the cap was binding on
// the whole distribution, not trimming a tail. So RAFT trained on replies 16x
// shorter than the ones it was scored on, and every round taught the model to
// answer shorter than the round before: approved replies fell from a mean of 158
// characters on the freshly fine-tuned seed to 88 after ten rounds.
//
// It compounded through the judge. `stopped_on_marker` is false for a reply cut off
// at the cap, which forfeits `W_FORMAT` (0.30, the second-heaviest term), so 67% of
// candidates ended mid-sentence and long ones were structurally disadvantaged:
// approval ran 97.3% at 50-100 characters and 44.1% at 200-250. The pipeline was a
// short-reply filter in two stages, and RAFT faithfully learned it.
//
// 512 covers the median held-out reply with room to spare and fits alongside a
// prompt at ctx 1024. It costs real time — generation is the cycle's expensive
// stage, ~6 candidates x 80 prompts — but the alternative is spending less time to
// produce training data that moves the model away from the target.
const DEFAULT_MAX_TOKENS: usize = 512;
// Below this a reply cannot be a reply; above `ctx * 3/4` there is no room left for
// the prompt the reply is answering.
const MIN_MAX_TOKENS: usize = 32;

#[derive(Deserialize, Serialize)]
pub struct AutolabelRequest {
    // Which model generates the candidates. Normally the one about to be tuned:
    // preference data is only meaningful for the policy that produced it.
    pub id_model: Uuid,
    // Which round these generations belong to, stamped onto every line written. Set
    // by `jobs::cycle`; absent for a hand-started pass from the API, which belongs to
    // no round and whose output is therefore unattributable backlog to a scoped run.
    // See `LabelledLine` for why the round rather than the model.
    #[serde(default)]
    pub id_cycle: Option<Uuid>,
    #[serde(default)]
    pub round: Option<i32>,
    #[serde(default = "default_candidates")]
    pub candidates: usize,
    #[serde(default = "default_max_prompts")]
    pub max_prompts: i64,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    // How long a candidate reply may run. Clamped against the model's context window
    // at run time, since a request cannot know it. See `DEFAULT_MAX_TOKENS` for why
    // this is a knob at all rather than the constant it used to be.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    // Refit the threshold from the current human ratings instead of reusing the
    // frozen one. The move at a plateau: once the model clears the old bar, that
    // bar has nothing left to say, and a new one is fitted from replies rated at
    // the new quality level. Explicit because it resets the yardstick every earlier
    // round was measured against.
    #[serde(default)]
    pub recalibrate: bool,
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
fn default_max_tokens() -> usize {
    DEFAULT_MAX_TOKENS
}

/// How long a reply may actually run, given what was asked for and what the model can
/// hold.
///
/// Clamped against the context window here rather than at the request, because the
/// ceiling belongs to the model and a stored request does not know which model it will
/// be run against. The upper bound leaves a quarter of the window for the prompt the
/// reply is answering — without it, asking for a reply as long as the context produces
/// a prompt truncated to nothing.
fn reply_budget(requested: usize, ctx_window: usize) -> usize {
    let ceiling = (ctx_window * 3 / 4).max(MIN_MAX_TOKENS);
    requested.clamp(MIN_MAX_TOKENS, ceiling)
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
//
// The provenance stamp. The file is append-only and outlives every model and cycle
// in it, so without this a training run cannot tell a line its own round wrote from
// one written six rounds ago — and a cycle's RAFT stage keeps re-training on the
// union of every round it has ever run.
//
// **`(id_cycle, round)` is the key that scopes a training run; `id_model` is kept
// for provenance only.** Keying on the model does not separate runs: every re-run of
// a cycle starts from the same seed and generates with it, so every run writes the
// same `id_model`. That is not a hypothetical — four runs of one cycle stamped 1160
// lines with one id, the catalogue grew 320 → 604 → 900, and the round score fell
// 58.39 → 56.85 → 54.41 in step. The round is the unit that is actually unique.
//
// Both are `Option` on the read side: a hand-started run from the API belongs to no
// cycle, and lines written before this field existed have no stamp at all. Being in
// this file they are known to be generated, so a scoped run drops them as backlog it
// cannot attribute — see `training::read_catalogue`.
#[derive(Serialize)]
struct LabelledLine<'a> {
    prompt: &'a str,
    response: &'a str,
    label: &'a str,
    id_model: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    id_cycle: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    round: Option<i32>,
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
    // One session for the whole pass: the weights go to the device once rather than
    // once per generated token.
    let mut session =
        llm::training::infer_session::InferSession::new(&model, model.ctx_window());

    on_progress("Fitting the judge", 0.0);
    // The judge is fit on the same reference prose the benchmark scores against.
    // Reusing it rather than the training corpus is deliberate: the lexicon and the
    // n-gram statistics should describe English, not this particular model's diet.
    let reference = load_reference()?;
    let judge = llm::autolabel::Judge::build(&reference);

    on_progress("Calibrating against your ratings", 0.0);
    let calibration = calibrate(&judge, request.recalibrate)?;
    on_calibration(&calibration);

    let ctx_window = model.ctx_window();
    let max_tokens = reply_budget(request.max_tokens, ctx_window);
    // The reply budget *is* the reserve. It used to be a flat `ctx/4`, which was
    // generous against a 96-token cap and would be short against a 512-token one — the
    // prompt would crowd out the reply and the cap would silently bind again, which is
    // the whole failure being fixed here.
    let reserve = max_tokens;
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
        // No budget. What this generates becomes RAFT/preference training data,
        // which `build_training_example` renders unbudgeted — asking for a budget
        // here would generate under a marker the data is never labelled with.
        let seq =
            llm::chat_template::build_prompt(None, &turns, None, &merges, ctx_window, reserve);

        let mut scored = Vec::with_capacity(candidates_per_prompt);
        for k in 0..candidates_per_prompt {
            let config = llm::eval::generate::SamplingConfig {
                max_tokens,
                temperature: request.temperature,
                top_k: DEFAULT_TOP_K,
                // Seeded from the prompt index and the candidate index, so a rerun
                // over the same prompts reproduces the same candidates exactly —
                // otherwise a labelling pass could never be debugged.
                seed: (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k as u64,
            };
            let continuation = llm::eval::generate::continue_tokens(
                &mut session,
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
                id_model: request.id_model,
                id_cycle: request.id_cycle,
                round: request.round,
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
const HUMAN_FILE: &str = storage::HUMAN_FEEDBACK_FILE;

// Where the fitted threshold is frozen.
const CALIBRATION_FILE: &str = "calibration.json";

// The frozen bar, plus what it was fitted from.
//
// **Why it is frozen rather than refitted each run.** The threshold is what a
// round's labels are decided against, so it is the loop's target. Refitting it
// every round against the current model's output would move the target with the
// model: whatever it produces becomes the new definition of good, the same
// fraction is labelled bad forever, and the loop can neither converge nor say how
// far it has come. Frozen, "fraction of replies clearing the bar" is a real
// learning curve against a fixed reference — and when almost everything clears it,
// the catalogue goes one-sided and the run refuses, which is the loop reporting
// that it has extracted what this bar can teach.
//
// The provenance fields are not decoration: a bar is only meaningful next to the
// ratings it came from, and at a plateau the question is always "what quality level
// was this fitted at". `--recalibrate` is how a new one is fitted, deliberately an
// explicit act.
#[derive(Deserialize, Serialize)]
pub struct FrozenCalibration {
    pub calibration: llm::autolabel::Calibration,
    pub fitted_at: String,
    // How many human ratings it was fitted from, so a later run can tell a bar fitted
    // from 30 judgements from one fitted from 300.
    pub ratings: usize,
}

pub fn calibration_path() -> std::path::PathBuf {
    storage::preference_dir().join(CALIBRATION_FILE)
}

/// The frozen bar, if one has been fitted. `None` before the first catalogue run.
pub fn frozen_calibration() -> Option<FrozenCalibration> {
    let content = std::fs::read_to_string(calibration_path()).ok()?;
    serde_json::from_str(&content).ok()
}

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
fn calibrate(
    judge: &llm::autolabel::Judge,
    recalibrate: bool,
) -> Result<llm::autolabel::Calibration, String> {
    // The frozen bar wins unless this run was explicitly asked to refit one. This
    // is the whole point of freezing: round 7 labels against the same threshold as
    // round 1, so "48% clear the bar" means something next to round 1's "12%".
    if !recalibrate {
        if let Some(frozen) = frozen_calibration() {
            return Ok(frozen.calibration);
        }
    }

    let (calibration, ratings) = fit_calibration(judge)?;
    freeze(calibration, ratings);
    Ok(calibration)
}

// Fits a threshold from the human ratings on disk. Pure: reads, computes, returns.
// Returns the ratings count alongside so the caller can record what it was fitted
// from.
fn fit_calibration(
    judge: &llm::autolabel::Judge,
) -> Result<(llm::autolabel::Calibration, usize), String> {
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

    Ok((calibration, total))
}

// Freezes a freshly fitted bar to disk.
//
// Separate from `fit_calibration` so that fitting has no side effects: the
// diagnostic below answers "what would the bar be right now" and must not
// silently replace the number every earlier round was measured against.
fn freeze(calibration: llm::autolabel::Calibration, ratings: usize) {
    let frozen = FrozenCalibration {
        calibration,
        fitted_at: chrono::Utc::now().to_rfc3339(),
        ratings,
    };
    match serde_json::to_string_pretty(&frozen) {
        Ok(encoded) => {
            if let Err(e) = std::fs::write(calibration_path(), encoded) {
                // Not fatal: this run has its threshold and can label with it. The
                // next run refits, which costs a second of CPU — worth reporting,
                // not worth refusing over.
                eprintln!("[autolabel] could not freeze the calibration: {e}");
            }
        }
        Err(e) => eprintln!("[autolabel] could not encode the calibration: {e}"),
    }
}

// One line of the human ratings file, as `queue::export_preference_data` writes it.
#[derive(Deserialize)]
struct RatedLine {
    response: String,
    label: String,
    #[serde(default)]
    id_cycle: Option<Uuid>,
    #[serde(default)]
    round: Option<i32>,
}

// How many auto-labelled examples are on disk, split by verdict.
//
// Counted from the file rather than tracked in a table because the file is the
// thing `build_kto_dataset` actually reads — a count kept anywhere else could say
// a thousand while the training run sees none, which is the class of mismatch this
// whole gate exists to prevent. Cheap: a few thousand short lines.
//
// **`only` must match what the training run will be given.** These counts are what
// the size gates in `gather` are checked against, so counting the whole catalogue
// and then training on one round's slice would let a gate promise 400 examples and
// the build deliver 60. `jobs::cycle` passes its own `(id_cycle, round)` here and
// the same value into the `TrainConfig`; a caller outside a cycle passes `None` and
// gets the whole file, matching what it will train on. The unstamped-line rule is
// `training::read_catalogue`'s, and is deliberately identical.
//
// A missing file is `(0, 0)`, not an error: no auto-labelling has run yet is the
// normal state before the first pass.
pub fn catalogue_counts(only: Option<(Uuid, i32)>) -> (i64, i64) {
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
        // This file is the generated half, so an unstamped line here is backlog
        // from before the stamp existed — dropped under a filter, exactly as
        // `training::read_catalogue` drops it.
        if let Some((want_cycle, want_round)) = only {
            if parsed.id_cycle != Some(want_cycle) || parsed.round != Some(want_round) {
                continue;
            }
        }
        match parsed.label.as_str() {
            "up" => approved += 1,
            "down" => rejected += 1,
            _ => {}
        }
    }
    (approved, rejected)
}

// The reference text the judge is fit on: the benchmark's prose corpus, plus the
// assistant turns of the curated SFT conversations.
//
// **Why both.** Prose alone is what this used to be, and it made the judge score
// the wrong thing. The corpus is children's-story English; the replies being judged
// are chat answers full of paths, flags, identifiers and markup. Measured against a
// prose-only reference, a correct technical reply scored *below* a degenerate
// repeating loop, so the catalogue it produced selected against every property SFT
// had just taught. Adding the assistant turns puts that register in the reference.
//
// **Why the prose stays.** It is ~50× the volume, and the lexicon needs volume: a
// frequency-thresholded word list built from the SFT turns alone is a few hundred
// words, which marks most correct output "unreal". The prose carries the statistics;
// the SFT turns carry the domain.
//
// Markers are stripped from the SFT side — `<|par|>` is structure, not vocabulary,
// and `Judge::fluency` strips them from the reply for the same reason. Leaving them
// in would make marker text part of what "fluent English" means.
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

    // Absent or unreadable SFT conversations are not an error: the judge still works
    // on prose alone, exactly as it did before, just with the old genre blind spot.
    for reply in sft_assistant_turns() {
        text.push_str(&reply);
        text.push('\n');
    }
    Ok(text)
}

// Every assistant turn in `raw/sft/*.jsonl`, markers stripped. Only the assistant
// side: user turns in these conversations are pasted logs, terminal dumps and file
// listings, which are not a model of anything the model should be rewarded for
// writing.
fn sft_assistant_turns() -> Vec<String> {
    #[derive(serde::Deserialize)]
    struct Turn {
        role: String,
        text: String,
    }
    #[derive(serde::Deserialize)]
    struct Conversation {
        turns: Vec<Turn>,
    }

    let dir = storage::sft_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut paths: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("jsonl"))
        .collect();
    paths.sort();

    let mut out = Vec::new();
    for path in paths {
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // A malformed line is skipped rather than failing the run — the same
            // tolerance the generator's own validator applies on the way in.
            if let Ok(conv) = serde_json::from_str::<Conversation>(line) {
                for turn in conv.turns {
                    if turn.role == "assistant" {
                        out.push(llm::eval::surface::strip_markers(&turn.text));
                    }
                }
            }
        }
    }
    out
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
mod reply_budget_tests {
    use super::*;

    // **The regression this whole knob exists to prevent.** The default has to survive
    // the clamp at the context window the models actually run at, or the cap binds
    // again and the cycle goes back to training on replies a quarter the length of the
    // ones it is scored against.
    #[test]
    fn the_default_survives_the_clamp_at_the_context_windows_in_use() {
        for ctx in [1024, 2048, 4096] {
            assert_eq!(
                reply_budget(DEFAULT_MAX_TOKENS, ctx),
                DEFAULT_MAX_TOKENS,
                "ctx {ctx} clipped the default"
            );
        }
    }

    // A reply as long as the whole window leaves nothing for the prompt it answers.
    #[test]
    fn a_reply_cannot_crowd_out_its_own_prompt() {
        assert_eq!(reply_budget(usize::MAX, 1024), 768);
        assert_eq!(reply_budget(4096, 1024), 768);
    }

    // The old hard-coded 96 is still expressible — this is a knob, not a new constant
    // in disguise — but nothing can ask for a budget too short to hold a reply.
    #[test]
    fn a_smaller_budget_is_honoured_down_to_the_floor() {
        assert_eq!(reply_budget(96, 1024), 96);
        assert_eq!(reply_budget(0, 1024), MIN_MAX_TOKENS);
    }

    // A window too small for the floor still has to yield a usable budget rather than
    // an inverted clamp, which would panic.
    #[test]
    fn a_tiny_context_window_does_not_invert_the_clamp() {
        assert_eq!(reply_budget(512, 16), MIN_MAX_TOKENS);
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    // The counting side of the same rule `training::read_catalogue` applies to the
    // building side. These two must agree: the gates in `labeling::gather` are
    // checked against this count and the run is then handed what that other
    // function selects, so a disagreement lets a gate pass on four hundred
    // examples and the build deliver sixty.
    #[test]
    fn counting_is_scoped_the_same_way_reading_is() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let cycle = Uuid::new_v4();
        let mine = (cycle, 3);
        let seed = Uuid::new_v4();
        let tmp = std::env::temp_dir().join(format!("counts_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dir = tmp.join("raw").join("preference");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(super::OUTPUT_FILE),
            format!(
                "{}\n{}\n{}\n{}\n",
                serde_json::json!({"prompt":"p","response":"a","label":"up",
                                   "id_model":seed,"id_cycle":cycle,"round":3}),
                serde_json::json!({"prompt":"p","response":"b","label":"down",
                                   "id_model":seed,"id_cycle":cycle,"round":3}),
                // Same cycle, same *generating model*, an earlier round. Counting by
                // model put this in the round's own bucket and that is the whole bug:
                // every re-run generates with the same seed, so the catalogue
                // accumulated instead of rotating.
                serde_json::json!({"prompt":"p","response":"c","label":"up",
                                   "id_model":seed,"id_cycle":cycle,"round":2}),
                // Unstamped, and in the generated file: backlog. Dropped under a
                // filter and counted without one, exactly as `read_catalogue`
                // treats it.
                serde_json::json!({"prompt":"p","response":"d","label":"up"}),
            ),
        )
        .unwrap();
        // SAFETY: single-threaded under ENV_LOCK; DATA_DIR drives storage::root().
        unsafe { std::env::set_var("DATA_DIR", &tmp) };

        assert_eq!(super::catalogue_counts(Some(mine)), (1, 1));
        assert_eq!(super::catalogue_counts(None), (3, 1));

        std::fs::remove_dir_all(&tmp).ok();
    }

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
        // `true` = fit fresh from the ratings rather than reporting back the frozen
        // number, which is what makes this a diagnostic: it answers "what would the
        // bar be if fitted right now", which at a plateau is exactly the question.
        match super::fit_calibration(&judge) {
            Ok((c, _)) => println!(
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
