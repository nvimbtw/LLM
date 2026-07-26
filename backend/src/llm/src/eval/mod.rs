// Benchmarking for the models this project trains.
//
// The point is to replace "it looks kind of like English" with numbers that can be
// compared between two checkpoints. Everything here is chosen for a model in the
// 10⁵–10⁷ parameter range: the standard leaderboard benchmarks (MMLU, HellaSwag,
// ARC, GSM8K) test knowledge and reasoning these models don't have, so they'd
// return chance forever and measure nothing. What *does* discriminate at this
// scale is how well the model predicts real text, whether it can tell a
// grammatical sentence from an ungrammatical one, and whether its own output is
// made of real words.
//
// Four stages, in the order they became worth measuring:
//
//   1. **Held-out bits per byte** (`corpus`) — the primary number. Also the one
//      thing the training loop never produced: there was no validation split, so
//      "loss went down" could not be distinguished from memorization.
//   2. **The n-gram floor** (`ngram`) — an order-5 byte model fit on the same
//      text. Turns an uninterpretable float into a verdict.
//   3. **Minimal-pair grammar** (`minimal_pairs`) — BLiMP-style forced choice,
//      chance exactly 50%. The headline capability number.
//   4. **Surface metrics** (`surface`) — real-word rate, repetition, marker
//      validity over text the model actually generated.
//   5. **Preference** (`preference`) — the only stage that looks at chat-shaped
//      text: does the model assign higher likelihood to the replies people gave a
//      thumbs up than to the ones they gave a thumbs down? Reported but kept out
//      of the composite (see `preference`'s header for why).
//
// Stages 1–3 and 5 are all sums of `log p(token | prefix)` over supplied text,
// which is why `score` is the only forward path in the module.

pub mod corpus;
pub mod generate;
pub mod minimal_pairs;
pub mod ngram;
pub mod preference;
pub mod score;
pub mod surface;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use serde::Serialize;

use crate::training::transformer::Model;

// Default caps on how much reference text a run reads. The held-out cap is the one
// that sets the run's duration — every held-out token costs a forward pass — so it
// is deliberately small enough that a benchmark finishes while you watch it.
pub const DEFAULT_MAX_HELDOUT_BYTES: usize = 128 * 1024;
pub const DEFAULT_MAX_BASELINE_BYTES: usize = 4 * 1024 * 1024;

// Prompts for the generation stage. Plain prose openings rather than chat turns,
// so the same set works for a pretrained base model and an SFT'd chat model — a
// chat template in front of a base model would measure the template, not the
// model. Fixed and ordered: changing this list changes the numbers, so it is part
// of the benchmark's definition, not a runtime knob.
const DEFAULT_PROMPTS: &[&str] = &[
    "The old man walked down to the river and",
    "She opened the letter and read it slowly. It said",
    "In the morning the town was quiet, because",
    "There was once a small house at the edge of the",
    "He could not remember where he had left the",
    "The children ran across the field toward",
    "It had been raining for three days when",
    "The door opened and a woman came in. She was",
    "Nobody in the village had ever seen anything like",
    "After dinner they sat by the fire and talked about",
    "The book was old, and its pages were",
    "When the train finally arrived, the platform was",
];

pub struct EvalConfig {
    // Directory holding `weights` + `merges.bin` for the model under test.
    pub model_dir: PathBuf,
    // Reference prose (`*.txt`). Split 90/10 into a baseline-fitting slice and a
    // held-out slice. Point this at text the model was **not** trained on.
    pub corpus_dir: PathBuf,
    // Optional real BLiMP JSONL, evaluated alongside the built-in suite.
    pub blimp_dir: PathBuf,
    // Rated replies exported from the chat feedback, as JSONL. Empty/absent until
    // someone has clicked a thumbs, in which case the stage is simply skipped.
    pub preference_dir: PathBuf,
    pub max_heldout_bytes: usize,
    pub max_baseline_bytes: usize,
    pub sampling: generate::SamplingConfig,
}

impl EvalConfig {
    pub fn new(model_dir: PathBuf, eval_dir: &Path) -> EvalConfig {
        EvalConfig {
            model_dir,
            corpus_dir: eval_dir.join("corpus"),
            blimp_dir: eval_dir.join("blimp"),
            preference_dir: eval_dir.join("preference"),
            max_heldout_bytes: DEFAULT_MAX_HELDOUT_BYTES,
            max_baseline_bytes: DEFAULT_MAX_BASELINE_BYTES,
            sampling: generate::SamplingConfig::default(),
        }
    }
}

// Called as the run advances. `fraction` is overall progress in [0, 1] so the
// caller doesn't have to know the stage weights.
pub type ProgressCallback = Box<dyn Fn(&str, f64) + Send>;

// A run is interruptible between forward passes, not during one — the same
// granularity training cancels at. Mirrors `train::train`'s convention: an
// `Option<Arc<AtomicBool>>` the caller flips, checked at each stage boundary and
// inside the two long loops.
type Cancel = Option<Arc<AtomicBool>>;

fn cancelled(cancel: &Cancel) -> bool {
    cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst))
}

// Stage weights for the overall progress fraction. Grammar dominates because it is
// by far the most forward passes (two per pair, hundreds of pairs). Preference gets
// the smallest slice: it is one forward pass per rated reply, and there are rarely
// many. These affect only the progress bar — `composite_score` carries its own,
// separate weights.
const W_CORPUS: f64 = 0.38;
const W_GRAMMAR: f64 = 0.42;
const W_GENERATION: f64 = 0.14;
const W_PREFERENCE: f64 = 0.06;

// ── the report ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ModelInfo {
    pub dimensions: usize,
    pub layers: usize,
    pub heads: usize,
    pub context: usize,
    pub vocab: usize,
    pub parameters: usize,
}

#[derive(Serialize)]
pub struct CorpusMetrics {
    pub bits_per_byte: f64,
    pub bits_per_token: f64,
    pub perplexity: f64,
    pub tokens_scored: usize,
    pub bytes_scored: usize,
    pub baseline_bits_per_byte: f64,
    pub advantage: f64,
    pub files: usize,
}

#[derive(Serialize)]
pub struct ParadigmResult {
    pub name: String,
    pub description: String,
    pub accuracy: f64,
    pub pairs: usize,
}

#[derive(Serialize)]
pub struct GrammarMetrics {
    // Pooled over every pair in every paradigm. Chance is 0.5.
    pub accuracy: f64,
    pub pairs: usize,
    pub paradigms: Vec<ParadigmResult>,
}

#[derive(Serialize)]
pub struct GenerationMetrics {
    pub real_word_rate: f64,
    pub distinct_1: f64,
    pub distinct_2: f64,
    pub distinct_3: f64,
    pub repeat_rate: f64,
    pub mean_sentence_words: f64,
    pub marker_validity: f64,
    pub words_scored: usize,
    pub lexicon_size: usize,
    // A few prompt/continuation pairs, so the numbers can be sanity-checked
    // against what the model actually wrote.
    pub samples: Vec<Sample>,
}

#[derive(Serialize)]
pub struct Sample {
    pub prompt: String,
    pub completion: String,
}

#[derive(Serialize)]
pub struct Report {
    // Composite 0–100. See `composite_score` for the formula; the components are
    // all present individually, and they are what to trust for anything subtle.
    pub score: f64,
    pub model: ModelInfo,
    // Absent when no reference corpus is installed — the run still produces
    // grammar and generation numbers.
    pub corpus: Option<CorpusMetrics>,
    pub grammar: GrammarMetrics,
    pub generation: GenerationMetrics,
    // Absent when no thumbs have been collected yet — which is every run until the
    // chat UI has been used. Never folded into `score`; see `preference`'s header.
    pub preference: Option<preference::PreferenceMetrics>,
    pub duration_secs: f64,
}

// ── the run ─────────────────────────────────────────────────────────────────

// Runs the whole suite. Blocking and GPU-bound (every stage drives CUDA through
// `forward_seq`), so callers must give it its own OS thread rather than an async
// task — the same constraint training has.
// `Ok(None)` means the caller asked for a cancel and the run stopped early — a
// distinct outcome from `Err`, which is a genuine failure. A partial report is
// never returned: half a benchmark is not a score, and storing one would put a
// meaningless point on the trend chart.
pub fn run(
    config: &EvalConfig,
    progress: ProgressCallback,
    cancel: Cancel,
) -> Result<Option<Report>, String> {
    let started = Instant::now();

    let dir = config.model_dir.to_string_lossy().to_string();
    let merges = tokenize::bpe::load_merges(&config.model_dir.join("merges.bin").to_string_lossy())
        .map_err(|e| format!("could not load tokenizer: {e}"))?;
    let model = Model::load(&dir).map_err(|e| format!("could not load model: {e}"))?;

    pollster::block_on(async {
        let split = corpus::load(
            &config.corpus_dir,
            config.max_baseline_bytes,
            config.max_heldout_bytes,
        );

        progress("Scoring held-out text", 0.0);
        let corpus_report = corpus::evaluate(&model, &merges, &split).await;
        progress("Scoring held-out text", W_CORPUS);
        if cancelled(&cancel) {
            return Ok(None);
        }

        let Some(grammar) = run_grammar(&model, &merges, config, &progress, &cancel).await else {
            return Ok(None);
        };

        let Some(generation) =
            run_generation(&model, &merges, config, &split, &progress, &cancel).await
        else {
            return Ok(None);
        };

        let Some(preference) = run_preference(&model, &merges, config, &progress, &cancel).await
        else {
            return Ok(None);
        };
        progress("Done", 1.0);

        let corpus_metrics = corpus_report.map(|c| CorpusMetrics {
            bits_per_byte: c.bits_per_byte,
            bits_per_token: c.bits_per_token,
            perplexity: c.perplexity,
            tokens_scored: c.tokens_scored,
            bytes_scored: c.bytes_scored,
            baseline_bits_per_byte: c.baseline_bits_per_byte,
            advantage: c.advantage,
            files: c.files,
        });

        Ok(Some(Report {
            score: composite_score(&corpus_metrics, &grammar, &generation),
            model: describe(&model),
            corpus: corpus_metrics,
            grammar,
            generation,
            preference,
            duration_secs: started.elapsed().as_secs_f64(),
        }))
    })
}

// Forced-choice grammar judgements. One pair costs two full-sequence scores; the
// model is correct when it assigns the grammatical sentence the higher total
// log-probability.
// `None` => cancelled partway. Checked on the same 25-pair cadence as the progress
// report rather than every pair, so the atomic load doesn't sit in the inner loop
// beside two forward passes it would be noise against.
async fn run_grammar(
    model: &Model,
    merges: &[((u32, u32), u32)],
    config: &EvalConfig,
    progress: &ProgressCallback,
    cancel: &Cancel,
) -> Option<GrammarMetrics> {
    let mut paradigms = minimal_pairs::suite();
    paradigms.extend(minimal_pairs::load_external(&config.blimp_dir));

    let layers = model.layers as usize;
    let total_pairs: usize = paradigms.iter().map(|p| p.pairs.len()).sum();
    let mut done = 0usize;

    let mut results = Vec::new();
    let mut correct_total = 0usize;

    for paradigm in &paradigms {
        let mut correct = 0usize;
        for pair in &paradigm.pairs {
            let good = tokenize::bpe::encode(&pair.good, merges);
            let bad = tokenize::bpe::encode(&pair.bad, merges);
            let good_lp = score::sequence_logprob(model, &good, layers).await;
            let bad_lp = score::sequence_logprob(model, &bad, layers).await;
            // A strict `>` makes an exact tie count as wrong. Ties happen when the
            // model is untrained enough to be uniform, and scoring them as
            // successes would push a random model above 50%.
            if good_lp > bad_lp {
                correct += 1;
            }
            done += 1;
            if done % 25 == 0 && total_pairs > 0 {
                if cancelled(cancel) {
                    return None;
                }
                let fraction = done as f64 / total_pairs as f64;
                progress(
                    &format!("Grammar: {} ({done}/{total_pairs})", paradigm.name),
                    W_CORPUS + W_GRAMMAR * fraction,
                );
            }
        }
        correct_total += correct;
        results.push(ParadigmResult {
            name: paradigm.name.clone(),
            description: paradigm.description.clone(),
            accuracy: rate(correct, paradigm.pairs.len()),
            pairs: paradigm.pairs.len(),
        });
    }

    Some(GrammarMetrics {
        accuracy: rate(correct_total, total_pairs),
        pairs: total_pairs,
        paradigms: results,
    })
}

// Generate from the fixed prompt set and measure the surface properties of what
// came back.
// `None` => cancelled partway, checked once per prompt.
async fn run_generation(
    model: &Model,
    merges: &[((u32, u32), u32)],
    config: &EvalConfig,
    split: &corpus::CorpusSplit,
    progress: &ProgressCallback,
    cancel: &Cancel,
) -> Option<GenerationMetrics> {
    // The lexicon comes from the *training* slice, never the held-out one: it is a
    // reference vocabulary, and building it from the same bytes the perplexity is
    // measured on would couple two metrics that should be independent.
    let lexicon = surface::Lexicon::build(&split.train);

    let mut samples = Vec::new();
    for (i, prompt) in DEFAULT_PROMPTS.iter().enumerate() {
        if cancelled(cancel) {
            return None;
        }
        let fraction = i as f64 / DEFAULT_PROMPTS.len() as f64;
        progress(
            &format!("Generating samples ({}/{})", i + 1, DEFAULT_PROMPTS.len()),
            W_CORPUS + W_GRAMMAR + W_GENERATION * fraction,
        );
        let completion = generate::continue_prompt(model, merges, prompt, config.sampling).await;
        samples.push(Sample {
            prompt: (*prompt).to_string(),
            completion,
        });
    }

    let texts: Vec<String> = samples.iter().map(|s| s.completion.clone()).collect();
    let metrics = surface::analyze(&texts, &lexicon);

    Some(GenerationMetrics {
        real_word_rate: metrics.real_word_rate,
        distinct_1: metrics.distinct_1,
        distinct_2: metrics.distinct_2,
        distinct_3: metrics.distinct_3,
        repeat_rate: metrics.repeat_rate,
        mean_sentence_words: metrics.mean_sentence_words,
        marker_validity: metrics.marker_validity,
        words_scored: metrics.words_scored,
        lexicon_size: lexicon.len(),
        samples,
    })
}

// Score every rated reply and reduce them to the preference metrics.
//
// The return is nested and the two layers mean different things: the outer `None`
// is "cancelled partway", matching the other stages; the inner `None` is "there is
// no feedback to score", which is the normal state of a fresh install and must not
// look like a failure or a zero.
async fn run_preference(
    model: &Model,
    merges: &[((u32, u32), u32)],
    config: &EvalConfig,
    progress: &ProgressCallback,
    cancel: &Cancel,
) -> Option<Option<preference::PreferenceMetrics>> {
    let base = W_CORPUS + W_GRAMMAR + W_GENERATION;
    let examples = preference::load(&config.preference_dir);
    if examples.is_empty() {
        progress("No feedback to score", base + W_PREFERENCE);
        return Some(None);
    }

    let layers = model.layers as usize;
    let mut scored = Vec::with_capacity(examples.len());

    for (i, example) in examples.iter().enumerate() {
        // One forward pass per example here, against two per pair in the grammar
        // stage, so cancellation is checked every example rather than every 25th.
        if cancelled(cancel) {
            return None;
        }
        progress(
            &format!("Preference ({}/{})", i + 1, examples.len()),
            base + W_PREFERENCE * (i as f64 / examples.len() as f64),
        );
        // An unscoreable example (empty reply, misaligned scoring) is dropped
        // rather than counted, so it can't drag a mean toward zero.
        if let Some(logprob) = preference::score_example(model, merges, example, layers).await {
            scored.push((example.good, example.prompt.clone(), logprob));
        }
    }

    if scored.is_empty() {
        return Some(None);
    }
    Some(Some(preference::summarize(&scored)))
}

// ── scoring ─────────────────────────────────────────────────────────────────

// One number, on a 0–100 scale, so two checkpoints can be ordered at a glance.
//
// It is a weighted blend of four components, each first mapped onto [0, 1] so they
// are commensurable:
//
//   grammar   (0.40) — accuracy rescaled from its 0.5 chance floor, so a model
//                      that guesses scores 0 rather than 50.
//   corpus    (0.25) — the fraction of the n-gram floor's bits the model removes.
//   words     (0.20) — real-word rate of its own output.
//   variety   (0.15) — distinct-3, which collapses when the model loops.
//
// The weights say grammar is the capability that matters most at this scale and
// repetition matters least, being the easiest to fix with decoding settings. When
// a component is unavailable (no reference corpus installed) its weight is
// redistributed across the rest rather than counted as zero — a missing
// measurement is not a failed one.
fn composite_score(
    corpus: &Option<CorpusMetrics>,
    grammar: &GrammarMetrics,
    generation: &GenerationMetrics,
) -> f64 {
    let mut components: Vec<(f64, f64)> = Vec::new();

    if grammar.accuracy.is_finite() {
        components.push((0.40, ((grammar.accuracy - 0.5) / 0.5).clamp(0.0, 1.0)));
    }
    if let Some(c) = corpus {
        if c.advantage.is_finite() {
            components.push((0.25, c.advantage.clamp(0.0, 1.0)));
        }
    }
    if generation.real_word_rate.is_finite() {
        components.push((0.20, generation.real_word_rate.clamp(0.0, 1.0)));
    }
    if generation.distinct_3.is_finite() {
        components.push((0.15, generation.distinct_3.clamp(0.0, 1.0)));
    }

    let weight: f64 = components.iter().map(|(w, _)| w).sum();
    if weight <= 0.0 {
        return f64::NAN;
    }
    // Dividing by the *present* weight is the redistribution.
    100.0 * components.iter().map(|(w, v)| w * v).sum::<f64>() / weight
}

fn describe(model: &Model) -> ModelInfo {
    let d = model.embedding_table.first().map(|row| row.len()).unwrap_or(0);
    let vocab = model.embedding_table.len();
    let layers = model.layers as usize;
    let heads = model.heads as usize;
    let hd = model.head_dim as usize;
    let ff = model.w1.first().and_then(|l| l.first()).map(|r| r.len()).unwrap_or(0);

    // Everything the optimizer touches, plus the (untrained) positional table, so
    // the figure matches what "model size" usually means.
    let per_layer = 3 * heads * d * hd  // wq, wk, wv
        + d * d                          // wo
        + d * ff + ff                    // w1, b1
        + ff * d + d                     // w2, b2
        + 2 * d;                         // ln gamma, beta
    let parameters = vocab * d              // token embeddings
        + model.positional_table.len() * d  // positional table
        + layers * per_layer
        + d * vocab; // output projection

    ModelInfo {
        dimensions: d,
        layers,
        heads,
        context: model.positional_table.len(),
        vocab,
        parameters,
    }
}

// A rate over no observations is unknown, not zero — NaN serializes to null so the
// frontend can render "—" instead of a confident wrong number.
fn rate(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        return f64::NAN;
    }
    part as f64 / whole as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grammar(accuracy: f64) -> GrammarMetrics {
        GrammarMetrics { accuracy, pairs: 100, paradigms: Vec::new() }
    }

    fn generation(real_word_rate: f64, distinct_3: f64) -> GenerationMetrics {
        GenerationMetrics {
            real_word_rate,
            distinct_1: 1.0,
            distinct_2: 1.0,
            distinct_3,
            repeat_rate: 0.0,
            mean_sentence_words: 10.0,
            marker_validity: 1.0,
            words_scored: 100,
            lexicon_size: 1000,
            samples: Vec::new(),
        }
    }

    fn corpus(advantage: f64) -> Option<CorpusMetrics> {
        Some(CorpusMetrics {
            bits_per_byte: 2.0,
            bits_per_token: 8.0,
            perplexity: 256.0,
            tokens_scored: 1000,
            bytes_scored: 4000,
            baseline_bits_per_byte: 3.0,
            advantage,
            files: 1,
        })
    }

    #[test]
    fn a_perfect_model_scores_one_hundred() {
        let score = composite_score(&corpus(1.0), &grammar(1.0), &generation(1.0, 1.0));
        assert!((score - 100.0).abs() < 1e-9, "got {score}");
    }

    #[test]
    fn a_chance_level_model_scores_zero_on_grammar() {
        // Grammar at chance contributes nothing; the other components are zeroed
        // here so the total isolates it.
        let score = composite_score(&corpus(0.0), &grammar(0.5), &generation(0.0, 0.0));
        assert!(score.abs() < 1e-9, "got {score}");
    }

    #[test]
    fn below_chance_grammar_does_not_go_negative() {
        let score = composite_score(&corpus(0.0), &grammar(0.1), &generation(0.0, 0.0));
        assert!(score >= 0.0, "got {score}");
    }

    #[test]
    fn a_missing_corpus_redistributes_its_weight_rather_than_scoring_zero() {
        // Same model, one measurement unavailable: the score must not drop just
        // because the corpus wasn't installed.
        let with = composite_score(&corpus(1.0), &grammar(1.0), &generation(1.0, 1.0));
        let without = composite_score(&None, &grammar(1.0), &generation(1.0, 1.0));
        assert!((with - without).abs() < 1e-9, "{with} vs {without}");
    }

    #[test]
    fn a_missing_corpus_still_scores_the_other_components() {
        let score = composite_score(&None, &grammar(1.0), &generation(0.0, 0.0));
        // grammar is 0.40 of the 0.75 weight that remains.
        assert!((score - 100.0 * 0.40 / 0.75).abs() < 1e-9, "got {score}");
    }

    #[test]
    fn a_looping_model_is_penalized_through_distinct_3() {
        let varied = composite_score(&corpus(0.5), &grammar(0.8), &generation(0.9, 0.9));
        let looping = composite_score(&corpus(0.5), &grammar(0.8), &generation(0.9, 0.05));
        assert!(varied > looping, "{varied} vs {looping}");
    }

    #[test]
    fn a_report_with_no_usable_components_is_not_a_number() {
        let score = composite_score(&None, &grammar(f64::NAN), &generation(f64::NAN, f64::NAN));
        assert!(score.is_nan());
    }

    #[test]
    fn rate_over_nothing_is_not_a_number() {
        assert!(rate(0, 0).is_nan());
        assert_eq!(rate(1, 2), 0.5);
    }

    #[test]
    fn non_finite_metrics_serialize_as_null() {
        // The frontend relies on this to render "—" for an unavailable metric.
        let value = serde_json::to_value(f64::NAN).unwrap();
        assert!(value.is_null());
    }
}
