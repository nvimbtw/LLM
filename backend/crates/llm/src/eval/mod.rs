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
//   5. **Preference** (`preference`) — does the model assign higher likelihood to
//      the replies people gave a thumbs up than to the ones they gave a thumbs
//      down? Reported but kept out of the composite (see `preference`'s header).
//
// Stages 1–3 and 5 are all sums of `log p(token | prefix)` over supplied text,
// which is why `score` is the only forward path in the module.
//
// **There are two benchmarks here, not one.** Everything above is the *language*
// suite (`run_language`), and every stage of it measures prose. That is deliberate —
// one suite has to work on a base model and a fine-tuned one alike — but it left
// supervised fine-tuning with no instrument at all: SFT moves a model toward
// chat-shaped text, so a *successful* fine-tune showed up here as a regression and
// nowhere as a gain. `markup` (`run_markup`) is the other half, and it owns the
// held-out chat set, `chat`'s bits-per-reply-token included.

pub mod chat;
pub mod corpus;
pub mod generate;
pub mod marker_pairs;
pub mod markup;
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

use crate::chat_template;
use crate::training::infer_session::InferSession;
use crate::training::transformer::Model;

// Default caps on how much reference text a run reads. The held-out cap is the one
// that sets the run's duration — every held-out token costs a forward pass.
//
// Doubled from 128 KiB. The corpus term now carries 45% of the score through a
// logistic that never clamps, so it is worth measuring properly; and with the grammar
// suite tripled the run's wall clock is set by grammar regardless. `corpus::load`
// spreads this across every file in the directory rather than draining the first two.
pub const DEFAULT_MAX_HELDOUT_BYTES: usize = 256 * 1024;
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
    // Held-out conversations in the `raw/sft/*.jsonl` shape — text no fine-tune
    // trained on. Absent until a set is installed, in which case the stage is
    // skipped like `preference`.
    pub chat_dir: PathBuf,
    pub max_heldout_bytes: usize,
    pub max_baseline_bytes: usize,
    pub sampling: generate::SamplingConfig,
    /// Render every held-out conversation under this thinking budget instead of the
    /// one it was written with. `None` keeps each record's own budget, which is what
    /// every run before this did and what the scores on disk were produced under.
    ///
    /// Exactly the chat socket's convention (`queue::ChatRequest::budget` →
    /// `build_prompt`): the caller supplies a marker, absence renders none, and the
    /// suite has no opinion about which. That is deliberate — the marker is a cue the
    /// prompt carries, so "benchmark this model at high" is the same act as "ask this
    /// model a question at high", and it should go through the same door.
    ///
    /// **It reaches only what renders a chat prompt.** The assistant suite is all
    /// held-out conversations, so all of it responds; the markup suite's position,
    /// structure and reply-bits stages respond, but its marker-pair stage carries 55%
    /// of that score and has no conversation to mark, so a markup sweep moves less
    /// than the whole score. The language suite renders no chat prompt anywhere —
    /// corpus, grammar, generation and preference are all raw text — so setting this
    /// changes nothing there, and `run_language` ignores it.
    pub budget: Option<chat_template::ThinkBudget>,
    /// How to render the held-out replies — which must match how the model under
    /// test was *trained*, not be a choice about the benchmark.
    ///
    /// The assistant suite's whole score is bits per assistant-reply token, so
    /// scoring a model trained under `Strip` against replies that still carry their
    /// `<|thinking|>` blocks charges it for tokens it was taught never to emit, and
    /// reports a regression that is purely an artifact of the mismatch. `jobs::queue`
    /// therefore reads this off the model's own `meta.json` rather than exposing it
    /// as a knob on the benchmark form.
    ///
    /// Like `budget`, it reaches only what renders a conversation: the assistant
    /// suite entirely, the markup suite's non-pair stages, and nothing in the
    /// language suite.
    pub reasoning: chat_template::Reasoning,
}

impl EvalConfig {
    pub fn new(model_dir: PathBuf, eval_dir: &Path) -> EvalConfig {
        EvalConfig {
            model_dir,
            corpus_dir: eval_dir.join("corpus"),
            blimp_dir: eval_dir.join("blimp"),
            preference_dir: eval_dir.join("preference"),
            chat_dir: eval_dir.join("chat"),
            max_heldout_bytes: DEFAULT_MAX_HELDOUT_BYTES,
            max_baseline_bytes: DEFAULT_MAX_BASELINE_BYTES,
            sampling: generate::SamplingConfig::default(),
            budget: None,
            // Every score already on disk was measured against unstripped replies.
            reasoning: chat_template::Reasoning::Keep,
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
// by far the most forward passes (two per pair, hundreds of pairs). Preference and
// chat get the smallest slices: one forward pass per example, and there are rarely
// many of either. These affect only the progress bar — `composite_score` carries
// its own, separate weights, and must sum to 1.0.
const STAGE_CORPUS: f64 = 0.36;
const STAGE_GRAMMAR: f64 = 0.40;
const STAGE_GENERATION: f64 = 0.12;
const STAGE_PREFERENCE: f64 = 0.12;

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
    pub floor_progress: f64,
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
    // Duplicated out of `breakdown` because `jobs::queue` lifts it into its own
    // column without parsing the rest of the report.
    pub score: f64,
    // The score's parts and its uncertainty. Carries `version`, which is what makes
    // an old run identifiable as un-comparable rather than merely worse.
    pub breakdown: ScoreBreakdown,
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
    // Tokens the model actually forwarded to produce this report — see
    // `InferSession::tokens_forwarded`. The cost of the run in the unit the model
    // works in, which unlike wall clock is the same number on any machine. All
    // three reports carry it under this name, so the charts read one field.
    pub tokens_used: u64,
}
// **The chat metric deliberately no longer lives here.** It sat in this report for
// one release as a reported-but-unscored field, because the post-train cycle judged
// on `BenchmarkSuite::Language` and had no other way to see the number it was
// actually optimizing. That was a workaround, and it made the language suite a
// slightly different question than its name — "how well does it model English, and
// also here is a chat number".
//
// `run_assistant` is now the answer instead: replies get their own suite, their own
// score and their own version, and this one goes back to measuring language only.

// ── the run ─────────────────────────────────────────────────────────────────

// Runs the whole suite. Blocking and GPU-bound (every stage drives CUDA through
// `forward_seq`), so callers must give it its own OS thread rather than an async
// task — the same constraint training has.
// `Ok(None)` means the caller asked for a cancel and the run stopped early — a
// distinct outcome from `Err`, which is a genuine failure. A partial report is
// never returned: half a benchmark is not a score, and storing one would put a
// meaningless point on the trend chart.
pub fn run_language(
    config: &EvalConfig,
    progress: ProgressCallback,
    cancel: Cancel,
) -> Result<Option<Report>, String> {
    let started = Instant::now();

    let (model, merges) = load(config)?;
    // One session for the whole run: the weights go to the device once, not once
    // per stage. `model` is kept only for `describe`.
    let mut session = InferSession::new(&model, model.ctx_window());

    let split = corpus::load(
        &config.corpus_dir,
        config.max_baseline_bytes,
        config.max_heldout_bytes,
    );

    progress("Scoring held-out text", 0.0);
    let corpus_report = corpus::evaluate(&mut session, &merges, &split);
    progress("Scoring held-out text", STAGE_CORPUS);
    if cancelled(&cancel) {
        return Ok(None);
    }

    let Some(grammar) = run_grammar(&mut session, &merges, config, &progress, &cancel) else {
        return Ok(None);
    };

    let Some(generation) =
        run_generation(&mut session, &merges, config, &split, &progress, &cancel)
    else {
        return Ok(None);
    };

    let Some(preference) = run_preference(&mut session, &merges, config, &progress, &cancel)
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
        floor_progress: c.floor_progress,
        files: c.files,
    });

    let breakdown = composite_score(&corpus_metrics, &grammar, &generation);
    Ok(Some(Report {
        score: breakdown.score,
        breakdown,
        model: describe(&model),
        corpus: corpus_metrics,
        grammar,
        generation,
        preference,
        duration_secs: started.elapsed().as_secs_f64(),
        tokens_used: session.tokens_forwarded(),
    }))
}

/// Stamps the configured budget onto every conversation, in place.
///
/// Applied once at load rather than threaded through the four render sites, because
/// the marker is a property of the prompt being issued and every stage builds its
/// prompt from the conversation it was handed. Doing it here means a stage cannot
/// be added later that quietly forgets to honour the override.
///
/// `None` leaves each record's own budget untouched — the unbudgeted path is
/// byte-identical to what ran before this existed, which is what keeps every score
/// already on disk comparable.
fn apply_budget(conversations: &mut [chat::Conversation], budget: Option<chat_template::ThinkBudget>) {
    let Some(budget) = budget else { return };
    for conversation in conversations {
        conversation.budget = Some(budget);
    }
}

// The weights and the model under test, shared by both suites.
fn load(config: &EvalConfig) -> Result<(Model, Vec<((u32, u32), u32)>), String> {
    let dir = config.model_dir.to_string_lossy().to_string();
    let merges = tokenize::bpe::load_merges(&config.model_dir.join("merges.bin").to_string_lossy())
        .map_err(|e| format!("could not load tokenizer: {e}"))?;
    let model = Model::load(&dir).map_err(|e| format!("could not load model: {e}"))?;
    Ok((model, merges))
}

// ── the markup suite ────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct MarkupReport {
    pub score: f64,
    pub breakdown: markup::MarkupBreakdown,
    pub model: ModelInfo,
    pub markup: markup::MarkupMetrics,
    // Bits per assistant-reply token over the same held-out conversations. Reported,
    // never scored: a base model has never seen the chat scaffold, and folding this in
    // would mark every pretrain down for failing a task it was not trained on. Read it
    // as a paired base-versus-fine-tune comparison. This is also the first thing that
    // actually runs `eval::chat`, which has been dead since it was written because
    // nothing exported `eval/chat/`.
    pub chat: Option<chat::ChatMetrics>,
    pub duration_secs: f64,
    // See `Report::tokens_used`.
    pub tokens_used: u64,
    // The budget every conversation was rendered under, or None when each kept its
    // own. Recorded on the report because a score is only comparable to another one
    // taken at the same budget, and nothing else in the report says which was used.
    pub budget: Option<chat_template::ThinkBudget>,
    // Held-out conversations behind this run. The markup score is only partly built
    // on them — the pair stage is not — but this is what a budget sweep varies, so
    // it is what says whether a point on the curve rests on enough evidence.
    pub conversations: usize,
}

// Stage weights for the markup progress fraction. Pairs dominate for the same reason
// grammar does in the language suite: two forward passes per pair, ~1000 pairs.
const STAGE_PAIRS: f64 = 0.55;
const STAGE_POSITIONS: f64 = 0.20;
const STAGE_STRUCTURE: f64 = 0.20;
const STAGE_CHAT_BITS: f64 = 0.05;

// How many held-out conversations the generation stage draws prompts from. Each costs
// a full sampling loop, which is far more expensive than a scoring pass.
const STRUCTURE_REPLIES: usize = 40;
// Prompt/completion pairs kept in the report so the numbers can be eyeballed against
// what the model actually wrote.
const STRUCTURE_SAMPLES: usize = 6;

// Runs the markup suite. Same contract as `run_language`: blocking, GPU-bound, needs
// its own OS thread, and `Ok(None)` means cancelled rather than failed.
pub fn run_markup(
    config: &EvalConfig,
    progress: ProgressCallback,
    cancel: Cancel,
) -> Result<Option<MarkupReport>, String> {
    let started = Instant::now();
    let (model, merges) = load(config)?;
    let mut session = InferSession::new(&model, model.ctx_window());

    let Some(pairs) = run_marker_pairs(&mut session, &merges, &progress, &cancel) else {
        return Ok(None);
    };

    let mut conversations = markup::load(&config.chat_dir);
    apply_budget(&mut conversations, config.budget);
    if conversations.is_empty() {
        progress("No held-out chat set", 1.0);
    }

    let Some(positions) =
        run_positions(&mut session, &merges, &conversations, config.reasoning, &progress, &cancel)
    else {
        return Ok(None);
    };
    let Some(structure) = run_structure(
        &mut session,
        &merges,
        &conversations,
        config.sampling,
        config.reasoning,
        &progress,
        &cancel,
    )
    else {
        return Ok(None);
    };
    let Some(chat_metrics) = run_chat_bits(
        &mut session,
        &merges,
        &conversations,
        STAGE_PAIRS + STAGE_POSITIONS + STAGE_STRUCTURE,
        STAGE_CHAT_BITS,
        config.reasoning,
        &progress,
        &cancel,
    ) else {
        return Ok(None);
    };
    progress("Done", 1.0);

    // What was *scored*, not what was loaded: a conversation refused for a bad role
    // or a misaligned mask contributes no evidence, and counting it would make a thin
    // point look sound. Bound before the report is built, because moving
    // `chat_metrics` into it would put this read after the move.
    let scored_conversations = chat_metrics
        .as_ref()
        .map(|c| c.conversations)
        .unwrap_or(conversations.len());
    let metrics = markup::MarkupMetrics { pairs, positions, structure };
    let breakdown = markup::composite_score(&metrics);
    Ok(Some(MarkupReport {
        score: breakdown.score,
        breakdown,
        model: describe(&model),
        markup: metrics,
        chat: chat_metrics,
        duration_secs: started.elapsed().as_secs_f64(),
        tokens_used: session.tokens_forwarded(),
        budget: config.budget,
        conversations: scored_conversations,
    }))
}

// ── the assistant suite ─────────────────────────────────────────────────────
//
// How good the model is at being an assistant: bits per assistant-reply token over
// the held-out conversations, mapped to 0–100.
//
// **This is the post-train cycle's objective, and it is the only suite that is.**
// The language suite asks whether the model can model English and the markup suite
// whether it can drive its own harness; both are things a chat fine-tune can trade
// away, and neither goes up when a reply gets better. The instrument that does go up
// has existed since `eval::chat` was written, but it had no suite, so it was carried
// as an unscored field on whichever report happened to run — which meant the cycle
// could only judge chat tuning by parsing a language report.
//
// **Why a whole suite for one number.** Because the number is on a different scale to
// everything else and answers a different question, which is the same reason
// `language` and `markup` are separate. A single blended score across all three would
// let a large reply gain pay for a language collapse and still read as progress; that
// is exactly what `jobs::cycle`'s vector test exists to refuse, and it can only refuse
// it if the three arrive as three numbers.
//
// **Bits per token is a cost — lower is better — and the score inverts it**, so that
// all three suites agree that up is good. Getting that wrong in either direction is
// the failure mode this mapping is written to make impossible: a note reading
// "3.6875 → 3.8796, no gain" was once read as an improvement, because nothing in the
// unit says which way is up.

#[derive(Serialize)]
pub struct AssistantBreakdown {
    pub version: u32,
    // The mapped reply component in [0, 1]. NaN when no held-out set is installed.
    pub replies: f64,
    pub score: f64,
    // Propagated from `bits_per_token_stderr` through the mapping's slope, so
    // `jobs::cycle` can derive its margin here the same way it does for the other two.
    pub score_stderr: f64,
}

#[derive(Serialize)]
pub struct AssistantReport {
    pub score: f64,
    pub breakdown: AssistantBreakdown,
    pub model: ModelInfo,
    // The raw measurement behind the score. Absent when no held-out chat set is
    // installed, which makes `score` NaN rather than zero — an unmeasured model is
    // not a bad one.
    pub chat: Option<chat::ChatMetrics>,
    pub duration_secs: f64,
    // See `Report::tokens_used`.
    pub tokens_used: u64,
    // See `MarkupReport::budget`. The whole assistant score is built on the
    // conversations this marks, so it is the suite a sweep moves furthest.
    pub budget: Option<chat_template::ThinkBudget>,
    // Conversations scored. Small enough to matter: a tier with three held-out
    // conversations behind it carries an error bar several times the differences a
    // sweep is looking for, and the chart marks such a point as provisional.
    pub conversations: usize,
}

// Bump alongside any change to the anchors below, for the reason `SCORE_VERSION`
// gives: it moves every score without a model having changed.
pub const ASSISTANT_SCORE_VERSION: u32 = 1;

// Bits per reply token through a logistic, the same shape and for the same reasons as
// the language suite's `BPB_MID`/`BPB_SCALE`: strictly monotone, smooth, asymptotic at
// both ends, so it never clamps and an improvement always pays.
//
// MID sits near where the current cohort actually lands (3.69–3.88 measured across a
// seed and its post-train rounds), which is where the response is steepest. SCALE
// sets the resolution: peak slope is 100/(4·SCALE) = 41.7 points per bit/token, so the
// 0.19 bits/token a single round has been observed to move is worth ~8 points, against
// a per-run `bits_per_token_stderr` of ~0.05 — i.e. ~2.1 points. Comfortably resolvable,
// which is the property the language composite lacked and why it could not gate this.
const CHAT_MID: f64 = 3.50;
const CHAT_SCALE: f64 = 0.60;

/// Runs the assistant suite. Same contract as the other two: blocking, GPU-bound,
/// needs its own OS thread, and `Ok(None)` means cancelled rather than failed.
///
/// Far cheaper than either — one forward pass per held-out conversation, a few dozen,
/// against the ~1000 the grammar and marker-pair stages each spend. Running all three
/// per round costs barely more than running the language suite alone did.
pub fn run_assistant(
    config: &EvalConfig,
    progress: ProgressCallback,
    cancel: Cancel,
) -> Result<Option<AssistantReport>, String> {
    let started = Instant::now();
    let (model, merges) = load(config)?;
    let mut session = InferSession::new(&model, model.ctx_window());

    let mut conversations = chat::load(&config.chat_dir);
    apply_budget(&mut conversations, config.budget);
    if conversations.is_empty() {
        progress("No held-out chat set", 1.0);
    }

    // The same loader the markup suite reads through — one definition of "held out",
    // so the suites cannot drift into disagreeing about which conversations the model
    // was allowed to see.
    let Some(chat_metrics) =
        run_chat_bits(
            &mut session,
            &merges,
            &conversations,
            0.0,
            1.0,
            config.reasoning,
            &progress,
            &cancel,
        )
    else {
        return Ok(None);
    };
    progress("Done", 1.0);

    // What was *scored*, not what was loaded: a conversation refused for a bad role
    // or a misaligned mask contributes no evidence, and counting it would make a thin
    // point look sound. Bound before the report is built, because moving
    // `chat_metrics` into it would put this read after the move.
    let scored_conversations = chat_metrics
        .as_ref()
        .map(|c| c.conversations)
        .unwrap_or(conversations.len());
    let breakdown = assistant_score(&chat_metrics);
    Ok(Some(AssistantReport {
        score: breakdown.score,
        breakdown,
        model: describe(&model),
        chat: chat_metrics,
        duration_secs: started.elapsed().as_secs_f64(),
        tokens_used: session.tokens_forwarded(),
        budget: config.budget,
        conversations: scored_conversations,
    }))
}

fn replies_component(bits_per_token: f64) -> f64 {
    1.0 / (1.0 + ((bits_per_token - CHAT_MID) / CHAT_SCALE).exp())
}

fn assistant_score(chat: &Option<chat::ChatMetrics>) -> AssistantBreakdown {
    let measured = chat
        .as_ref()
        .map(|c| c.bits_per_token)
        .filter(|b| b.is_finite());

    let Some(bits) = measured else {
        return AssistantBreakdown {
            version: ASSISTANT_SCORE_VERSION,
            replies: f64::NAN,
            score: f64::NAN,
            score_stderr: f64::NAN,
        };
    };

    let replies = replies_component(bits);
    // The logistic's derivative: |d(score)/d(bits)| = 100·σ·(1−σ)/SCALE. A missing
    // error bar on the measurement stays missing here rather than becoming zero — a
    // gate reading zero uncertainty would accept any movement at all.
    let stderr = chat
        .as_ref()
        .map(|c| c.bits_per_token_stderr)
        .filter(|s| s.is_finite())
        .map(|s| 100.0 * replies * (1.0 - replies) / CHAT_SCALE * s)
        .unwrap_or(f64::NAN);

    AssistantBreakdown {
        version: ASSISTANT_SCORE_VERSION,
        replies,
        score: 100.0 * replies,
        score_stderr: stderr,
    }
}

fn run_marker_pairs(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    progress: &ProgressCallback,
    cancel: &Cancel,
) -> Option<markup::PairMetrics> {
    let paradigms = marker_pairs::suite();
    let total: usize = paradigms.iter().map(|p| p.pairs.len()).sum();
    let mut done = 0usize;
    let mut results = Vec::new();

    for paradigm in &paradigms {
        let mut correct = 0usize;
        for pair in &paradigm.pairs {
            if markup::score_pair(session, merges, pair) {
                correct += 1;
            }
            done += 1;
            // Same 25-pair cadence the grammar stage uses: the atomic load would be
            // noise beside two forward passes.
            if done % 25 == 0 && total > 0 {
                if cancelled(cancel) {
                    return None;
                }
                progress(
                    &format!("Markup: {} ({done}/{total})", paradigm.name),
                    STAGE_PAIRS * (done as f64 / total as f64),
                );
            }
        }
        results.push((
            paradigm.name.clone(),
            paradigm.description.clone(),
            correct,
            paradigm.pairs.len(),
        ));
    }

    Some(markup::summarize_pairs(&results))
}

fn run_positions(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    conversations: &[chat::Conversation],
    reasoning: chat_template::Reasoning,
    progress: &ProgressCallback,
    cancel: &Cancel,
) -> Option<Option<markup::PositionMetrics>> {
    if conversations.is_empty() {
        return Some(None);
    }
    let mut tallies = Vec::new();
    for (i, conversation) in conversations.iter().enumerate() {
        // One forward pass per conversation, so cancellation is checked every one.
        if cancelled(cancel) {
            return None;
        }
        progress(
            &format!("Marker positions ({}/{})", i + 1, conversations.len()),
            STAGE_PAIRS + STAGE_POSITIONS * (i as f64 / conversations.len() as f64),
        );
        if let Some(tally) = markup::tally_positions(session, merges, conversation, reasoning) {
            tallies.push(tally);
        }
    }
    Some(markup::summarize_positions(&tallies))
}

fn run_structure(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    conversations: &[chat::Conversation],
    sampling: generate::SamplingConfig,
    reasoning: chat_template::Reasoning,
    progress: &ProgressCallback,
    cancel: &Cancel,
) -> Option<Option<markup::StructureMetrics>> {
    if conversations.is_empty() {
        return Some(None);
    }
    let base = STAGE_PAIRS + STAGE_POSITIONS;
    let ctx = session.ctx_window();
    let chosen: Vec<&chat::Conversation> =
        conversations.iter().take(STRUCTURE_REPLIES).collect();

    let mut replies = Vec::new();
    let mut samples = Vec::new();
    for (i, conversation) in chosen.iter().enumerate() {
        if cancelled(cancel) {
            return None;
        }
        progress(
            &format!("Markup generation ({}/{})", i + 1, chosen.len()),
            base + STAGE_STRUCTURE * (i as f64 / chosen.len() as f64),
        );
        let Some((prompt, user_text)) =
            markup::reply_prompt(conversation, merges, ctx, sampling.max_tokens)
        else {
            continue;
        };
        // Scored over the generated ids, not the decoded text — see `Continuation::ids`.
        let continuation = generate::continue_tokens(session, merges, prompt, sampling);
        replies.push(markup::tally_reply(&continuation.ids, continuation.stopped_on_marker));
        if samples.len() < STRUCTURE_SAMPLES {
            samples.push(markup::Sample { prompt: user_text, completion: continuation.text });
        }
    }

    let reference = markup::reference_density(conversations, merges, reasoning);
    Some(markup::summarize_structure(&replies, reference, samples))
}

// `base` and `weight` are where this stage sits in the caller's progress bar: both
// suites run it, and they run it at different points with different shares of the
// total, so neither can be a constant here.
//
// Eight arguments, and grouping them would be worse than the lint: the session and
// merges are the model, `base`/`weight` are the caller's progress geometry, and
// `reasoning` is a property of how the model was trained. A struct bundling three
// unrelated concerns for one private call site buys nothing.
#[allow(clippy::too_many_arguments)]
fn run_chat_bits(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    conversations: &[chat::Conversation],
    base: f64,
    weight: f64,
    reasoning: chat_template::Reasoning,
    progress: &ProgressCallback,
    cancel: &Cancel,
) -> Option<Option<chat::ChatMetrics>> {
    if conversations.is_empty() {
        return Some(None);
    }
    let mut scored = Vec::new();
    for (i, conversation) in conversations.iter().enumerate() {
        if cancelled(cancel) {
            return None;
        }
        progress(
            &format!("Reply bits ({}/{})", i + 1, conversations.len()),
            base + weight * (i as f64 / conversations.len() as f64),
        );
        if let Some(scores) = chat::reply_logprob(session, merges, conversation, reasoning) {
            scored.push(scores);
        }
    }
    Some(chat::summarize(&scored))
}

// Forced-choice grammar judgements. One pair costs two full-sequence scores; the
// model is correct when it assigns the grammatical sentence the higher total
// log-probability.
// `None` => cancelled partway. Checked on the same 25-pair cadence as the progress
// report rather than every pair, so the atomic load doesn't sit in the inner loop
// beside two forward passes it would be noise against.
fn run_grammar(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    config: &EvalConfig,
    progress: &ProgressCallback,
    cancel: &Cancel,
) -> Option<GrammarMetrics> {
    let mut paradigms = minimal_pairs::suite();
    paradigms.extend(minimal_pairs::load_external(&config.blimp_dir));

    let total_pairs: usize = paradigms.iter().map(|p| p.pairs.len()).sum();
    let mut done = 0usize;

    let mut results = Vec::new();
    let mut correct_total = 0usize;

    for paradigm in &paradigms {
        let mut correct = 0usize;
        for pair in &paradigm.pairs {
            let good = lead_in(&pair.good, merges);
            let bad = lead_in(&pair.bad, merges);
            let good_lp = score::sequence_logprob(session, &good);
            let bad_lp = score::sequence_logprob(session, &bad);
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
                    STAGE_CORPUS + STAGE_GRAMMAR * fraction,
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

// A sentence encoded behind an `<|endoftext|>`, so its own first token is scored.
//
// `score::token_logprobs` treats position 0 as context and never scores it — the
// first token is a prediction from nothing. For most paradigms that costs nothing,
// because both members of the pair open with the same word and the unscored position
// is identical on both sides. But for a paradigm whose contrast *is* the first word —
// `npi_licensing`'s `No student…` against `The student…`, and half of `pronoun_case` —
// the differing token was the one being thrown away, leaving only its effect on what
// followed. Prepending a document boundary makes the contrast a scored position on
// both sides.
//
// Prepended as an id rather than as text: `<|endoftext|>` is a single reserved token
// (see `pretokenize::SPECIAL_BLOCK`), so it adds exactly one position to each side and
// cannot perturb the tokenization of the sentence behind it.
fn lead_in(sentence: &str, merges: &[((u32, u32), u32)]) -> Vec<u32> {
    let mut ids = Vec::new();
    if let Some(bos) = tokenize::pretokenize::special_id('\u{E000}') {
        ids.push(bos);
    }
    ids.extend(tokenize::bpe::encode(sentence, merges));
    ids
}

// Generate from the fixed prompt set and measure the surface properties of what
// came back.
// `None` => cancelled partway, checked once per prompt.
fn run_generation(
    session: &mut InferSession,
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
            STAGE_CORPUS + STAGE_GRAMMAR + STAGE_GENERATION * fraction,
        );
        let completion = generate::continue_prompt(session, merges, prompt, config.sampling);
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
fn run_preference(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    config: &EvalConfig,
    progress: &ProgressCallback,
    cancel: &Cancel,
) -> Option<Option<preference::PreferenceMetrics>> {
    let base = STAGE_CORPUS + STAGE_GRAMMAR + STAGE_GENERATION;
    let examples = preference::load(&config.preference_dir);
    if examples.is_empty() {
        progress("No feedback to score", base + STAGE_PREFERENCE);
        return Some(None);
    }

    let mut scored = Vec::with_capacity(examples.len());

    for (i, example) in examples.iter().enumerate() {
        // One forward pass per example here, against two per pair in the grammar
        // stage, so cancellation is checked every example rather than every 25th.
        if cancelled(cancel) {
            return None;
        }
        progress(
            &format!("Preference ({}/{})", i + 1, examples.len()),
            base + STAGE_PREFERENCE * (i as f64 / examples.len() as f64),
        );
        // An unscoreable example (empty reply, misaligned scoring) is dropped
        // rather than counted, so it can't drag a mean toward zero.
        if let Some(logprob) = preference::score_example(session, merges, example) {
            scored.push((example.good, example.prompt.clone(), logprob));
        }
    }

    if scored.is_empty() {
        return Some(None);
    }
    Some(Some(preference::summarize(&scored)))
}

// ── scoring ─────────────────────────────────────────────────────────────────

// ── the language score ──────────────────────────────────────────────────────
//
// One number, on a 0–100 scale, so two checkpoints can be ordered at a glance.
//
// **Why this was rebuilt.** The previous formula was a weighted blend of four
// components, and across every model this project has benchmarked two of the four
// were constants:
//
//   * `floor_progress` computed to 1.07 / 1.13 / 1.13 and clamped to 1.0 for all
//     three — 25 points wired to a constant, hiding a 1.87 → 1.53 bits-per-byte
//     improvement, the largest real gain in the project's history. (Its predecessor,
//     `advantage`, had clamped to 0.0 on the old corpus for the mirror-image reason.
//     Anchoring the corpus term to the n-gram floor failed in *both* directions,
//     which is why the floor is no longer the score's scale.)
//   * `distinct_3` sat at 0.9896–0.9969 — a further 15 points, near-constant,
//     because any non-degenerate model at temperature 0.8 lands at ~0.99.
//
// So 40% of the score could not move, grammar carried 94% of what did, and a
// fine-tune that improved on bits per byte, real-word rate *and* repetition scored
// 0.10 *below* its base because it lost 2 grammar pairs out of 545. The cycle in
// `jobs::cycle` accepts or rejects rounds on this number; it could not do that job.
//
// The replacement scores three components and gates on the fourth:
//
//   language  (0.45) — a logistic in held-out bits per byte.
//   grammar   (0.40) — accuracy rescaled from its 0.5 chance floor, unchanged.
//   words     (0.15) — real-word rate, read through its *nonsense* complement.
//   variety   (gate) — distinct-3, as a multiplier, not a weighted term.
//
// When a component is unavailable (no reference corpus installed) its weight is
// redistributed across the rest rather than counted as zero — a missing measurement
// is not a failed one. That behaviour is unchanged.

// Bump whenever any scoring constant below moves, or when the pair suite, the prompt
// set or the corpus split changes. Scores are charted across every run ever recorded
// and compared by `jobs::cycle`; a silent change makes two incomparable numbers look
// like a regression, which is the exact failure this rebuild exists to remove.
pub const SCORE_VERSION: u32 = 2;

// Held-out bits per byte, through a logistic: strictly monotone, smooth, and
// asymptotic at both ends, so it never clamps and improvement always pays.
//
// MID sits at the centre of the range these models occupy, which is where the
// response is steepest. SCALE sets the resolution: peak slope is 1/(4·SCALE) per
// bit/byte, so 0.45 buys 0.556 units per bit/byte — 25 composite points — and a 0.1
// bpb gain is worth ~2.5 points, comfortably clear of the reported `score_stderr`.
// It still yields 2.9 points per 0.2 bpb down at 0.8 bpb, well past anything this
// project currently produces, so there is no second re-anchoring waiting.
const BPB_MID: f64 = 1.60;
const BPB_SCALE: f64 = 0.45;

// Real-word rate saturates — 0.909 to 0.952 across the cohort, 4% of a unit span —
// so the *nonsense* rate is what carries information at this end. Every extra 10
// nonsense words per hundred halves the component. Reaches 1.0 only at a perfect
// rate: an endpoint, not a clamp.
const NONSENSE_HALVING: f64 = 0.10;

// The repetition gate. Saturating at 1.0 is the design, not a defect: a healthy model
// must not be ranked on how far past "not looping" it is, which is precisely what
// distinct-3 was doing with 15 points of weight. Below FAIL the model is cycling and
// the other numbers have stopped describing language.
const VARIETY_FAIL: f64 = 0.30;
const VARIETY_PASS: f64 = 0.70;

const W_LANGUAGE: f64 = 0.45;
const W_GRAMMAR: f64 = 0.40;
const W_WORDS: f64 = 0.15;

// The score with its parts shown, so a number can be audited without re-deriving the
// formula by hand — and so a stored report can be re-scored under a later version.
#[derive(Serialize)]
pub struct ScoreBreakdown {
    pub version: u32,
    // Each mapped component, in [0, 1]. NaN when that measurement was unavailable.
    pub language: f64,
    pub grammar: f64,
    pub words: f64,
    // Multiplies the blend. 1.0 for anything that is not looping.
    pub variety_gate: f64,
    pub score: f64,
    // One standard error of `score` as an estimate of the *model's ability* — which
    // is what a cycle gate needs, since it asks whether B beats A, not whether the
    // benchmark reproduces. See `score_stderr`.
    pub score_stderr: f64,
}

fn language_component(bits_per_byte: f64) -> f64 {
    1.0 / (1.0 + ((bits_per_byte - BPB_MID) / BPB_SCALE).exp())
}

fn words_component(real_word_rate: f64) -> f64 {
    2f64.powf(-(1.0 - real_word_rate) / NONSENSE_HALVING)
}

fn variety_gate(distinct_3: f64) -> f64 {
    // A missing measurement is not a failed one: no generated text is no evidence of
    // looping, so the gate opens.
    if !distinct_3.is_finite() {
        return 1.0;
    }
    ((distinct_3 - VARIETY_FAIL) / (VARIETY_PASS - VARIETY_FAIL)).clamp(0.0, 1.0)
}

fn composite_score(
    corpus: &Option<CorpusMetrics>,
    grammar: &GrammarMetrics,
    generation: &GenerationMetrics,
) -> ScoreBreakdown {
    let language = corpus
        .as_ref()
        .map(|c| c.bits_per_byte)
        .filter(|b| b.is_finite())
        .map(language_component)
        .unwrap_or(f64::NAN);
    // Grammar's lower clamp is a verdict, not a saturation: below chance is one
    // thing, not a scale.
    let grammar_value = if grammar.accuracy.is_finite() {
        ((grammar.accuracy - 0.5) / 0.5).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };
    let words = if generation.real_word_rate.is_finite() {
        words_component(generation.real_word_rate)
    } else {
        f64::NAN
    };
    let gate = variety_gate(generation.distinct_3);

    let components = [
        (W_LANGUAGE, language),
        (W_GRAMMAR, grammar_value),
        (W_WORDS, words),
    ];
    let present: f64 = components.iter().filter(|(_, v)| v.is_finite()).map(|(w, _)| w).sum();
    let score = if present <= 0.0 {
        f64::NAN
    } else {
        // Dividing by the *present* weight is the redistribution.
        let blend: f64 = components
            .iter()
            .filter(|(_, v)| v.is_finite())
            .map(|(w, v)| w * v)
            .sum::<f64>()
            / present;
        100.0 * blend * gate
    };

    ScoreBreakdown {
        version: SCORE_VERSION,
        language,
        grammar: grammar_value,
        words,
        variety_gate: gate,
        score,
        score_stderr: score_stderr(grammar, generation, present, gate),
    }
}

// How far the score would move if the benchmark drew a different sample of the same
// size — the quantity `jobs::cycle`'s acceptance threshold should be derived from.
//
// Two independent estimation errors, propagated through their mappings and added in
// quadrature:
//
//   grammar — binomial over the pair set, times the mapping's slope of 2.
//   words   — binomial over the generated words, times the exponential's slope,
//             which is `ln2 / NONSENSE_HALVING` scaled by the component's own value.
//
// **The language term is deliberately absent.** The held-out slice is a fixed set of
// bytes scored deterministically, and it is the *same* text for every model. Its
// sampling error against "English at large" is real but is a constant offset shared
// by every run, so it cancels in a model-to-model comparison. Inventing a number for
// it would inflate the gate and reject real improvements.
fn score_stderr(
    grammar: &GrammarMetrics,
    generation: &GenerationMetrics,
    present_weight: f64,
    gate: f64,
) -> f64 {
    if present_weight <= 0.0 {
        return f64::NAN;
    }
    let mut variance = 0.0;

    if grammar.accuracy.is_finite() && grammar.pairs > 0 {
        let a = grammar.accuracy;
        let se = (a * (1.0 - a) / grammar.pairs as f64).sqrt();
        let points = 100.0 * (W_GRAMMAR / present_weight) * 2.0 * se;
        variance += points * points;
    }
    if generation.real_word_rate.is_finite() && generation.words_scored > 0 {
        let r = generation.real_word_rate;
        let se = (r * (1.0 - r) / generation.words_scored as f64).sqrt();
        let slope = std::f64::consts::LN_2 / NONSENSE_HALVING * words_component(r);
        let points = 100.0 * (W_WORDS / present_weight) * slope * se;
        variance += points * points;
    }

    // The gate scales the score, so it scales the score's uncertainty with it.
    gate * variance.sqrt()
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
        // No positional term: RoPE is parameter-free.
        + layers * per_layer
        + d * vocab; // output projection

    ModelInfo {
        dimensions: d,
        layers,
        heads,
        context: model.ctx_window(),
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

    fn grammar_of(accuracy: f64, pairs: usize) -> GrammarMetrics {
        GrammarMetrics { accuracy, pairs, paradigms: Vec::new() }
    }

    fn grammar(accuracy: f64) -> GrammarMetrics {
        grammar_of(accuracy, 100)
    }

    fn generation_of(real_word_rate: f64, distinct_3: f64, words: usize) -> GenerationMetrics {
        GenerationMetrics {
            real_word_rate,
            distinct_1: 1.0,
            distinct_2: 1.0,
            distinct_3,
            repeat_rate: 0.0,
            mean_sentence_words: 10.0,
            marker_validity: 1.0,
            words_scored: words,
            lexicon_size: 1000,
            samples: Vec::new(),
        }
    }

    fn generation(real_word_rate: f64, distinct_3: f64) -> GenerationMetrics {
        generation_of(real_word_rate, distinct_3, 100)
    }

    // Named for the component the composite actually reads. `floor_progress` and
    // `advantage` are carried alongside at fixed values precisely because they must
    // no longer move the score.
    fn corpus(bits_per_byte: f64) -> Option<CorpusMetrics> {
        Some(CorpusMetrics {
            bits_per_byte,
            bits_per_token: 8.0,
            perplexity: 256.0,
            tokens_scored: 1000,
            bytes_scored: 4000,
            baseline_bits_per_byte: 2.2718792427963264,
            advantage: 0.0,
            floor_progress: 1.0,
            files: 8,
        })
    }

    fn score_of(bpb: f64, accuracy: f64, rwr: f64, d3: f64) -> f64 {
        composite_score(&corpus(bpb), &grammar(accuracy), &generation(rwr, d3)).score
    }

    // ── the archived runs ────────────────────────────────────────────────────
    //
    // The three real reports in `benchmark_runs`, as measured. Hardcoded as *inputs*
    // rather than expected outputs: the properties below must hold for these models,
    // whatever the constants are tuned to.
    //
    // Balanced 1.01 is the supervised fine-tune of Balanced 1. It improved on bits
    // per byte (1.5304 → 1.5172), on real-word rate (0.9430 → 0.9517) and on
    // repetition, and lost 2 grammar pairs out of 545 — and the previous formula
    // scored it 0.10 *below* its base, which halts a post-train cycle. That is the
    // regression this whole rebuild exists to remove.
    #[derive(Clone, Copy)]
    struct Archived {
        name: &'static str,
        bpb: f64,
        accuracy: f64,
        pairs: usize,
        real_word_rate: f64,
        distinct_3: f64,
        words_scored: usize,
    }

    const SHAKEDOWN_1: Archived = Archived {
        name: "Shakedown 1",
        bpb: 1.8721,
        accuracy: 0.6899,
        pairs: 545,
        real_word_rate: 0.9094,
        distinct_3: 0.9969,
        words_scored: 320,
    };
    const BALANCED_1: Archived = Archived {
        name: "Balanced 1",
        bpb: 1.5304,
        accuracy: 0.8037,
        pairs: 545,
        real_word_rate: 0.9430,
        distinct_3: 0.9896,
        words_scored: 386,
    };
    const BALANCED_1_01_SFT: Archived = Archived {
        name: "Balanced 1.01",
        bpb: 1.5172,
        accuracy: 0.8000,
        pairs: 545,
        real_word_rate: 0.9517,
        distinct_3: 0.9909,
        words_scored: 331,
    };

    fn rescore(a: &Archived) -> ScoreBreakdown {
        composite_score(
            &corpus(a.bpb),
            &grammar_of(a.accuracy, a.pairs),
            &generation_of(a.real_word_rate, a.distinct_3, a.words_scored),
        )
    }

    // **The acceptance criterion for the rebuild.** A fine-tune that got better at
    // everything the suite measures except two grammar pairs must not read as a
    // regression, because `jobs::cycle` stops the loop when it does.
    #[test]
    fn the_recorded_sft_run_does_not_score_below_its_base() {
        let base = rescore(&BALANCED_1);
        let sft = rescore(&BALANCED_1_01_SFT);
        assert!(
            sft.score > base.score,
            "SFT {:.3} did not beat base {:.3}",
            sft.score,
            base.score,
        );
    }

    #[test]
    fn a_larger_pretrain_scores_clearly_above_a_smaller_one() {
        let small = rescore(&SHAKEDOWN_1).score;
        let large = rescore(&BALANCED_1).score;
        assert!(large - small > 10.0, "{small:.3} to {large:.3} is not a clear gap");
    }

    // The direct test for the failure being fixed: 40 of the old 100 points could not
    // move because two components were pinned at their bounds.
    #[test]
    fn no_component_clamps_on_any_archived_report() {
        for a in [SHAKEDOWN_1, BALANCED_1, BALANCED_1_01_SFT] {
            let b = rescore(&a);
            for (label, v) in [("language", b.language), ("grammar", b.grammar), ("words", b.words)]
            {
                assert!(v > 0.0 && v < 1.0, "{}: {label} pinned at {v}", a.name);
            }
        }
    }

    // Proves distinct-3 stopped being a weighted term rather than merely being
    // reweighted: every real model is on the flat part of the gate.
    #[test]
    fn the_variety_gate_is_exactly_one_for_every_archived_report() {
        for a in [SHAKEDOWN_1, BALANCED_1, BALANCED_1_01_SFT] {
            assert_eq!(rescore(&a).variety_gate, 1.0, "{}", a.name);
        }
    }

    #[test]
    fn the_three_archived_runs_span_more_than_fifteen_points() {
        // Guards against a future constant change quietly re-compressing the scale
        // back toward the 9.7-point span the old formula produced.
        let worst = rescore(&SHAKEDOWN_1).score;
        let best = rescore(&BALANCED_1_01_SFT).score;
        assert!(best - worst > 15.0, "span is only {:.2} points", best - worst);
    }

    // ── the components ───────────────────────────────────────────────────────

    #[test]
    fn a_flawless_model_approaches_but_does_not_reach_one_hundred() {
        // The asymptote is the point: there is always headroom, so improvement always
        // pays and the scale never needs re-anchoring again.
        let score = score_of(0.0, 1.0, 1.0, 1.0);
        assert!(score > 97.0 && score < 100.0, "got {score}");
    }

    #[test]
    fn a_chance_level_model_scores_zero_on_grammar() {
        let breakdown = composite_score(&None, &grammar(0.5), &generation(f64::NAN, f64::NAN));
        assert_eq!(breakdown.grammar, 0.0);
        assert!(breakdown.score.abs() < 1e-9, "got {}", breakdown.score);
    }

    #[test]
    fn below_chance_grammar_does_not_go_negative() {
        assert!(score_of(2.0, 0.1, 0.5, 1.0) >= 0.0);
    }

    #[test]
    fn language_modelling_is_strictly_monotone_in_bits_per_byte() {
        let mut previous = f64::INFINITY;
        let mut bpb = 0.1;
        while bpb <= 8.0 {
            let v = language_component(bpb);
            assert!(v < previous, "not decreasing at bpb {bpb}");
            previous = v;
            bpb += 0.05;
        }
    }

    #[test]
    fn words_is_strictly_monotone_in_the_real_word_rate() {
        let mut previous = f64::NEG_INFINITY;
        for step in 0..=100 {
            let v = words_component(step as f64 / 100.0);
            assert!(v > previous, "not increasing at rate {}", step as f64 / 100.0);
            previous = v;
        }
        assert!((words_component(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_component_clamps_across_the_operating_range() {
        // The whole range these models plausibly occupy, not just the three measured
        // points — so a better model a year from now does not silently re-pin a term.
        let mut bpb = 1.0;
        while bpb <= 2.0 {
            let mut accuracy = 0.65;
            while accuracy <= 0.95 {
                let mut rwr = 0.85;
                while rwr <= 0.98 {
                    let b = composite_score(
                        &corpus(bpb),
                        &grammar(accuracy),
                        &generation(rwr, 0.99),
                    );
                    for (label, v) in
                        [("language", b.language), ("grammar", b.grammar), ("words", b.words)]
                    {
                        assert!(
                            v > 0.0 && v < 1.0,
                            "{label} clamped to {v} at bpb={bpb} acc={accuracy} rwr={rwr}",
                        );
                    }
                    rwr += 0.01;
                }
                accuracy += 0.05;
            }
            bpb += 0.05;
        }
    }

    #[test]
    fn the_composite_is_strictly_monotone_in_every_scored_component() {
        let base = score_of(1.5, 0.80, 0.94, 0.99);
        assert!(score_of(1.4, 0.80, 0.94, 0.99) > base, "bits per byte");
        assert!(score_of(1.5, 0.82, 0.94, 0.99) > base, "grammar");
        assert!(score_of(1.5, 0.80, 0.95, 0.99) > base, "real words");
    }

    // Ties the resolution requirement to the measured noise rather than to a magic
    // number: a real modelling gain has to outrun the benchmark's own uncertainty.
    #[test]
    fn a_tenth_of_a_bit_per_byte_beats_the_reported_stderr() {
        let at = |bpb| {
            composite_score(
                &corpus(bpb),
                &grammar_of(0.80, 545),
                &generation_of(0.94, 0.99, 386),
            )
        };
        let before = at(1.5);
        let after = at(1.4);
        assert!(
            after.score - before.score > before.score_stderr,
            "gain {:.3} does not clear stderr {:.3}",
            after.score - before.score,
            before.score_stderr,
        );
    }

    // ── the variety gate ─────────────────────────────────────────────────────

    #[test]
    fn a_looping_model_is_gated_to_near_zero() {
        // Good on paper everywhere else; the output is a loop, so the score is not.
        let score = score_of(1.5, 0.80, 0.94, 0.05);
        assert!(score < 5.0, "got {score}");
    }

    #[test]
    fn the_variety_gate_is_not_a_scored_component() {
        // Two models differing only in distinct-3, both comfortably past the gate,
        // must score identically — the property the old 0.15 weight violated.
        let a = score_of(1.5, 0.80, 0.94, 0.99);
        let b = score_of(1.5, 0.80, 0.94, 0.80);
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn a_missing_distinct_3_opens_the_gate() {
        // A missing measurement is not a failed one.
        assert_eq!(variety_gate(f64::NAN), 1.0);
    }

    #[test]
    fn the_gate_closes_gradually_rather_than_at_a_cliff() {
        let mid = variety_gate((VARIETY_FAIL + VARIETY_PASS) / 2.0);
        assert!((mid - 0.5).abs() < 1e-12, "got {mid}");
        assert_eq!(variety_gate(VARIETY_FAIL - 0.01), 0.0);
        assert_eq!(variety_gate(VARIETY_PASS + 0.01), 1.0);
    }

    // ── redistribution ───────────────────────────────────────────────────────

    #[test]
    fn a_missing_corpus_redistributes_its_weight_rather_than_scoring_zero() {
        // Same model, one measurement unavailable: the score must not drop just
        // because the corpus wasn't installed.
        let with = composite_score(&corpus(0.0), &grammar(1.0), &generation(1.0, 1.0));
        let without = composite_score(&None, &grammar(1.0), &generation(1.0, 1.0));
        // `language` is asymptotic, so "perfect" differs slightly; compare the two
        // available components' blend instead of demanding equality.
        assert!(without.score > with.score, "{} vs {}", with.score, without.score);
        assert!(without.language.is_nan());
        assert!((without.score - 100.0).abs() < 1e-9, "got {}", without.score);
    }

    #[test]
    fn a_missing_corpus_still_scores_the_other_components() {
        let breakdown = composite_score(&None, &grammar(1.0), &generation(0.0, 1.0));
        // grammar is 0.40 of the 0.55 weight that remains; words contributes ~0 at a
        // real-word rate of 0.
        let expected = 100.0 * (0.40 + 0.15 * words_component(0.0)) / 0.55;
        assert!((breakdown.score - expected).abs() < 1e-9, "got {}", breakdown.score);
    }

    #[test]
    fn a_report_with_no_usable_components_is_not_a_number() {
        let breakdown =
            composite_score(&None, &grammar(f64::NAN), &generation(f64::NAN, f64::NAN));
        assert!(breakdown.score.is_nan());
        assert!(breakdown.score_stderr.is_nan());
    }

    #[test]
    fn the_breakdown_reproduces_the_score() {
        // Makes the serialized breakdown trustworthy for anything re-deriving a score
        // from a stored report.
        let b = rescore(&BALANCED_1);
        let expected = 100.0
            * (W_LANGUAGE * b.language + W_GRAMMAR * b.grammar + W_WORDS * b.words)
            * b.variety_gate;
        assert!((b.score - expected).abs() < 1e-9, "{} vs {expected}", b.score);
    }

    #[test]
    fn the_score_version_is_recorded() {
        assert_eq!(rescore(&BALANCED_1).version, SCORE_VERSION);
    }

    // ── score_stderr ─────────────────────────────────────────────────────────

    #[test]
    fn score_stderr_shrinks_as_the_square_root_of_the_pair_count() {
        let at = |pairs| {
            composite_score(
                &corpus(1.5),
                &grammar_of(0.80, pairs),
                // Words held at a rate whose binomial term vanishes, isolating grammar.
                &generation_of(1.0, 0.99, 1000),
            )
            .score_stderr
        };
        let ratio = at(545) / at(545 * 4);
        assert!((ratio - 2.0).abs() < 1e-9, "got {ratio}");
    }

    #[test]
    fn score_stderr_is_dominated_by_grammar_at_the_current_sample_sizes() {
        // Documents where extra sampling budget should go.
        let both = rescore(&BALANCED_1).score_stderr;
        let grammar_only = composite_score(
            &corpus(BALANCED_1.bpb),
            &grammar_of(BALANCED_1.accuracy, BALANCED_1.pairs),
            &generation_of(1.0, BALANCED_1.distinct_3, BALANCED_1.words_scored),
        )
        .score_stderr;
        assert!(grammar_only / both > 0.8, "grammar is only {:.2} of it", grammar_only / both);
    }

    #[test]
    fn score_stderr_falls_toward_zero_with_unlimited_samples() {
        let se = composite_score(
            &corpus(1.5),
            &grammar_of(0.80, 100_000_000),
            &generation_of(0.94, 0.99, 100_000_000),
        )
        .score_stderr;
        assert!(se < 0.01, "got {se}");
    }

    #[test]
    fn score_stderr_follows_the_gate() {
        // A model gated to zero has a zero score, so it has zero uncertainty about it.
        let gated = composite_score(
            &corpus(1.5),
            &grammar_of(0.80, 545),
            &generation_of(0.94, 0.0, 386),
        );
        assert_eq!(gated.score, 0.0);
        assert_eq!(gated.score_stderr, 0.0);
    }

    #[test]
    fn score_stderr_is_zero_when_there_is_nothing_to_sample() {
        let se = composite_score(
            &corpus(1.5),
            &grammar_of(f64::NAN, 0),
            &generation_of(f64::NAN, 0.99, 0),
        )
        .score_stderr;
        assert_eq!(se, 0.0);
    }

    // ── the assistant suite ──────────────────────────────────────────────────

    fn chat_of(bits: f64, stderr: f64) -> Option<chat::ChatMetrics> {
        Some(chat::ChatMetrics {
            bits_per_token: bits,
            logprob: -bits * std::f64::consts::LN_2,
            bits_per_token_stderr: stderr,
            conversations: 54,
            tokens_scored: 4000,
        })
    }

    // **The one property the whole suite exists to provide.** Bits per token is a
    // cost, every other suite reports higher-is-better, and a note mixing the two
    // directions was read backwards by a person who then asked why a regression was
    // being blocked. The inversion belongs here, once, not in the reader's head.
    #[test]
    fn fewer_bits_per_reply_token_scores_higher() {
        let better = assistant_score(&chat_of(3.6875, 0.05)).score;
        let worse = assistant_score(&chat_of(3.8796, 0.05)).score;
        assert!(better > worse, "{better} should beat {worse}");
    }

    // The anchors, pinned. These are the numbers `jobs::cycle`'s tests are written
    // against, so a change to `CHAT_MID`/`CHAT_SCALE` has to break here first — and
    // `ASSISTANT_SCORE_VERSION` has to move with it, since every recorded score does.
    #[test]
    fn the_mapping_sits_where_the_current_cohort_does() {
        let at = |bits| (assistant_score(&chat_of(bits, 0.05)).score * 1e4).round() / 1e4;
        assert_eq!(at(CHAT_MID), 50.0);
        assert_eq!(at(3.6875), 42.2505);
        assert_eq!(at(3.8796), 34.6906);
    }

    // The gate derives its margin from this, so it has to be the measurement's own
    // error carried through the mapping's slope — not a constant, and never zero.
    #[test]
    fn the_error_bar_is_propagated_through_the_mapping() {
        let breakdown = assistant_score(&chat_of(3.5, 0.05));
        // Peak slope is 100/(4·SCALE) = 41.67 points per bit, so 0.05 bits is ~2.08.
        assert!((breakdown.score_stderr - 2.0833).abs() < 1e-3, "{}", breakdown.score_stderr);
    }

    // An unmeasured model is not a bad one: with no held-out set the score must be NaN
    // (which serializes to null and reads as "—"), never a real zero that would plot as
    // a catastrophic regression and halt a cycle.
    #[test]
    fn no_held_out_chat_set_scores_nothing_rather_than_zero() {
        let breakdown = assistant_score(&None);
        assert!(breakdown.score.is_nan());
        assert!(breakdown.score_stderr.is_nan());
    }

    // A report with no error bar — fewer than two conversations — must not hand the
    // gate a zero, which would make any movement at all look certain.
    #[test]
    fn a_missing_error_bar_stays_missing() {
        let breakdown = assistant_score(&chat_of(3.5, f64::NAN));
        assert!(breakdown.score.is_finite());
        assert!(breakdown.score_stderr.is_nan());
    }

    // ── odds and ends ────────────────────────────────────────────────────────

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
