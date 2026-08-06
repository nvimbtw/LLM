// Synthetic SFT from a **local** teacher model, generated as thinking-budget
// triples.
//
// The sibling of the crate root's `run`, with two differences that are the whole
// point of the module:
//
//   1. **The model is local**, loaded through `teacher` onto the GPU, so a run
//      costs nothing per call and can produce thousands of conversations rather
//      than the couple of hundred a paid API is worth. Volume is the actual fix
//      for a student that repeats itself — the corpus this replaces was 188
//      conversations, which is far too few regardless of how diverse each one is.
//      Because it holds the device, work driven from here belongs on the Device
//      lane and takes the single GPU line.
//
//   2. **One question, three answers.** Each call asks for a user turn and then
//      three replies *to that same question*, each with a longer thinking block.
//      Sampling budgets independently across separate conversations would give
//      the same marginal distribution of thinking lengths, but the corpus would
//      contain no two records that differ *only* in reasoning depth. The matched
//      triple is what makes "think briefly" and "think hard" learnable as a
//      lever rather than as noise — a contrast the student can actually see.
//
// What this module does not do is decide how the tiers get used at training
// time. It generates, filters and labels; the label is a plain field on the
// record. `training::build_sft_dataset` reads it back and renders it as a
// `ThinkBudget` marker in front of the assistant turn — see `chat_template`.

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use teacher::{SampleOpts, Teacher, TeacherConfig};

// `TOPICS` and `FEATURES` are deliberately not imported: this module has its own
// `DOMAINS` and `MARKUP`, for the reasons documented on each. `SHAPES` is shared —
// the shape of a question is the same whether it is about a loss curve or a soufflé.
use crate::{
    Budget, Conversation, EXEMPLAR_BUDGET, MAX_ASSISTANT_CHARS, Rng, SHAPES, SYSTEM_PROMPTS, Turn,
    load_exemplars, load_openers, pick, pick_exemplars, repair_markup, tally,
    thinking_block, validate_reply, vocabulary,
};

// ── the tiers ───────────────────────────────────────────────────────────────

/// The three thinking budgets.
///
/// Deliberately three and not a continuous number of tokens: the label has to be
/// something a prompt can ask for and a person can read off a file name, and a
/// model asked for "a 173-token thinking block" cannot count its own tokens
/// anyway. The *boundaries* are configurable and measured in tokens; the tiers
/// themselves are fixed.
///
/// It **is** `llm::chat_template::ThinkBudget`, not a twin of it. The label this
/// module writes is the label SFT renders as a marker and the chat socket sends
/// back at inference; three separate enums that happened to serialize the same
/// way would be three places for the tiers to drift apart. What stays here is
/// what is genuinely this module's: the file a tier is written to, and the
/// delimiter the teacher is asked to answer under.
pub use llm::chat_template::ThinkBudget as Tier;

// One file per tier, flat in the SFT directory.
//
// Not a subdirectory, and that is not a style choice: every consumer of
// `raw/sft/` (`training::read_sft_conversations`, `jsonl_files` here,
// `labeling::autolabel`) globs it with a non-recursive `read_dir` filtered on
// the `.jsonl` extension. A directory entry has no extension, so anything
// nested one level down is silently invisible to all three — it would not
// error, the data simply would not exist as far as training is concerned.
pub fn tier_file_name(tier: Tier) -> &'static str {
    match tier {
        Tier::Low => "think-low.jsonl",
        Tier::Med => "think-med.jsonl",
        Tier::High => "think-high.jsonl",
    }
}

// The delimiter that opens this tier's answer in the model's output. Named
// per tier rather than a generic `<<<ANSWER>>>` so the parser knows which
// budget was *requested*, which is what makes the requested-vs-measured
// comparison possible at all.
fn delimiter(tier: Tier) -> &'static str {
    match tier {
        Tier::Low => "<<<SHORT>>>",
        Tier::Med => "<<<MEDIUM>>>",
        Tier::High => "<<<LONG>>>",
    }
}

// ── the request ─────────────────────────────────────────────────────────────

/// What a teaching run was asked for. Stored as the job's `config`, so it is also
/// what the retry button replays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeachRequest {
    /// Which local GGUF to run, and how much of the device to give it.
    pub teacher: TeacherConfig,
    #[serde(default)]
    pub sample: SampleOpts,
    /// How many *triples* to accept before stopping. A triple is one question and
    /// up to three answers, so this is a third of the conversation count.
    pub count: usize,
    /// Upper bound on thinking tokens for the `Low` tier, and for `Med`. Anything
    /// above the second is `High`. Measured in the **student's** tokenizer, not
    /// the teacher's, because the budget is only meaningful at the student's
    /// inference time — which means these do nothing without `id_model`.
    #[serde(default = "default_low_max")]
    pub low_max_tokens: usize,
    #[serde(default = "default_med_max")]
    pub med_max_tokens: usize,
    /// The student whose tokenizer and context window bound the run. Without one
    /// there is no way to measure a token budget, so classification falls back to
    /// the requested tier and says so in the stats.
    #[serde(default)]
    pub id_model: Option<Uuid>,
    /// Gold conversations to imitate. A file name inside the SFT dir, never a
    /// path.
    #[serde(default = "default_exemplar_file")]
    pub exemplar_file: String,
    /// Safety cap on model calls. 0 means "three times the ideal number", the
    /// budget that stops a teacher returning nothing usable from looping forever.
    #[serde(default)]
    pub max_calls: usize,
    #[serde(default)]
    pub seed: Option<u64>,
}

// `0` means "derive from the student's context window", which is the default and
// the only setting that can be right in general — see `derived_bounds`.
fn default_low_max() -> usize {
    0
}
fn default_med_max() -> usize {
    0
}

// The tier boundaries a student can actually afford, in its own tokens.
//
// **The previous constants (140/700) made the top tier arithmetically impossible.**
// A record is filed `high` only when its thinking block exceeds `med_max`, and
// `run` separately discards any conversation that does not fit the student's
// window. At ctx 1024 that asks for more than 700 tokens of reasoning plus a user
// turn plus a real reply inside 1024 — so every high-tier candidate was generated,
// validated, and then dropped by the window check. Measured thinking blocks on the
// long answers ran 982-1056 tokens: over the whole window before the reply exists.
// `think-high.jsonl` could not be created, at any markup quality.
//
// The constants were chosen from word targets alone, with no reference to the
// student. The crate root already had the answer — `Budget::word_capacity` exists
// because "the thinking block has to be derived from that capacity, not requested"
// — and this module never adopted it.
//
// So the bands are fractions of what the window leaves for reasoning:
//
//   * `0.9` of the window, the same slack `word_capacity` holds back for scaffold.
//   * `45%` of that for the thinking block, because the reply has to fit beside it
//     and a reply is the point of the exercise.
//   * the tiers then split that ceiling at 15% and 60%.
//
// At ctx 1024 this gives a ceiling of 414 and bounds of 62/248 — and the long
// answers already observed to survive validation measure 539-589 tokens, so they
// land in `high` instead of being discarded. At ctx 4096 it gives 248/995, i.e.
// the ambitious original numbers, reached by growing the student rather than by
// asserting them.
fn thinking_ceiling(ctx_window: usize) -> usize {
    (ctx_window * 9 / 10) * 45 / 100
}

fn derived_bounds(ctx_window: usize) -> (usize, usize) {
    let ceiling = thinking_ceiling(ctx_window);
    (ceiling * 15 / 100, ceiling * 60 / 100)
}

// What the run actually classifies against: the request's values when it set them,
// otherwise the student's window. With no student there is nothing to measure
// against at all, and the fallback is the old pair — which is wrong for a small
// window, but a run with no `id_model` files by *requested* tier anyway.
fn resolve_bounds(request: &TeachRequest, budget: Option<&Budget>) -> (usize, usize) {
    let derived = match budget {
        Some(b) => derived_bounds(b.ctx_window),
        None => (140, 700),
    };
    match (request.low_max_tokens, request.med_max_tokens) {
        (0, 0) => derived,
        (low, med) => match budget {
            // An explicit bound that cannot be satisfied is refused rather than
            // honoured. Every teach job queued before the boundaries were derived
            // carries 140/700 in its stored config, and the retry button replays
            // that config verbatim — so honouring it would silently reproduce the
            // impossible top tier on every retry of an old run.
            Some(b) if !high_tier_fits(med, b.ctx_window) => derived,
            _ => (low, med),
        },
    }
}

// Whether a record just over `med_max` can exist at all: its thinking block, a
// reply of comparable size, and the user turn, inside the window.
fn high_tier_fits(med_max: usize, ctx_window: usize) -> bool {
    (med_max + 1) * 2 + 40 <= ctx_window
}
// The hand-written general-domain gold set, not `synthetic.jsonl`.
//
// This is the file every call is told to copy the VOICE of, and the previous
// default made that instruction fight itself: `synthetic.jsonl` is Claude Code
// session transcripts — genuinely good data, which is why it stays in the corpus
// — but its register is "engineer auditing a codebase". The prompt separates
// voice from subject matter in as many words, and a 4B teacher copies both
// anyway. `gold-general.jsonl` is the same voice over cooking, money, health,
// travel and the rest, so imitating it pulls in the direction the DOMAINS
// rotation already asks for instead of against it.
//
// Written by `scripts/write_gold_general.py`, which is where to edit it.
fn default_exemplar_file() -> String {
    "gold-general.jsonl".to_string()
}

// ── what a run produces ─────────────────────────────────────────────────────

/// One accepted triple, as the live feed shows it.
#[derive(Debug, Clone, Serialize)]
pub struct AcceptedTriple {
    pub group: String,
    pub opener: String,
    /// Per answer: the tier it was filed under and how many thinking tokens it
    /// actually had. Present in the feed because "did the tiers separate" is the
    /// only question that matters about a run and the counts cannot answer it.
    pub answers: Vec<AnsweredTier>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnsweredTier {
    pub requested: Tier,
    pub filed_as: Tier,
    pub think_tokens: usize,
    pub reply_preview: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TeachReport {
    pub call: usize,
    /// Conversations this call wrote — 0 to 3, one per surviving answer. **Not**
    /// interchangeable with a question count: the two were reported side by side
    /// as if they were the same unit, which is what made "45 triples" sit next to
    /// tier bars summing to 75 and read as a counting bug.
    pub answers_kept: usize,
    /// How many of the three tiers this call's answers covered. 3 is a real
    /// triple; 1 is a lone answer wearing a group id.
    pub tiers_covered: usize,
    pub duplicates: usize,
    pub rejected: Vec<(String, usize)>,
    pub elapsed_ms: u128,
    pub tokens_generated: usize,
    /// Running totals, in the three units the run actually has. `questions` is
    /// what `target` bounds and what the progress bar fills.
    pub total_questions: usize,
    pub total_answers: usize,
    pub total_complete: usize,
    /// Running distribution of questions by tiers covered, so a page attaching
    /// mid-run draws the same figure a finished one does.
    pub total_by_tiers: Vec<usize>,
    pub target: usize,
}

/// What a run produced, in the three units it actually has.
///
/// These were one number called `triples`, which counted a question as a triple as
/// soon as **one** of its three answers survived. A run reporting "45 triples"
/// beside tier bars summing to 75 answers was not double-counting — the two numbers
/// were in different units, and the first one was named after a unit it wasn't
/// measuring. On the corpus that prompted this, 211 questions carried 305 answers
/// and only **5** covered all three tiers.
///
/// The distinction is the whole point of the format: a triple teaches the student
/// that these three lengths are three answers to *the same question*. A question
/// with one surviving answer teaches nothing about length — it is an ordinary SFT
/// conversation that happens to carry a budget marker.
#[derive(Debug, Clone, Serialize)]
pub struct TeachStats {
    /// Questions with at least one answer kept. This is what `TeachRequest::count`
    /// bounds and what the run stops on.
    pub questions: usize,
    /// Questions whose answers covered all three tiers — actual triples.
    pub complete_triples: usize,
    /// Conversations written across the three files. Equals the sum of `per_tier`,
    /// stated separately so the headline figure and the bars share a unit.
    pub answers: usize,
    /// Questions by how many distinct tiers they covered: index 0 is one tier, 1 is
    /// two, 2 is all three. Sums to `questions`, and its last entry is
    /// `complete_triples`.
    ///
    /// Kept as a distribution rather than only its top bucket because the shape is
    /// the diagnosis: a run that is mostly one-tier questions is failing differently
    /// from one that is mostly two-tier, and the second is a nudge from working.
    pub questions_by_tiers: Vec<usize>,
    pub calls: usize,
    pub duplicates: usize,
    pub rejected: Vec<(String, usize)>,
    /// How many answers landed in each tier *after* measurement. The headline
    /// number of a run: three roughly equal counts mean the tiers separated, and
    /// one bucket holding everything means the teacher ignored the instruction
    /// and the data teaches nothing about length.
    pub per_tier: Vec<(String, usize)>,
    /// How often the measured tier disagreed with the requested one. Not an error
    /// — the record is filed by measurement — but a high number means the word
    /// targets in the prompt are mistuned for this teacher.
    pub relabelled: usize,
    /// False when no `id_model` was given, so tiers are the requested ones rather
    /// than measured. Surfaced because it silently changes what the labels mean.
    pub measured: bool,
    pub gpu: bool,
}

pub type ProgressCallback = Box<dyn Fn(&str, f64) + Send>;
pub type ReportCallback = Box<dyn Fn(&TeachReport) + Send>;
pub type TripleCallback = Box<dyn Fn(&AcceptedTriple) + Send>;

// ── the record on disk ──────────────────────────────────────────────────────

/// One tier's answer, written as an ordinary SFT conversation plus the labels.
///
/// `turns` and `system` are exactly what `training::read_sft_conversations`
/// expects, and the three extra fields ride along harmlessly: nothing in the
/// workspace sets `deny_unknown_fields`, and `SftConversation` declares only
/// `system` and `turns`, so every existing reader ignores them.
///
/// `group` is the load-bearing one. `training::chat_destination` hashes a
/// conversation's *content* — assistant text included — to pick its
/// pretrain/SFT/eval split, so the three members of a triple hash to three
/// unrelated values and scatter across all three splits. That destroys the
/// matched contrast the triple exists to create. Carrying the group id now costs
/// nothing and leaves the fix available later: route by group when the field is
/// present, which keeps existing conversations' destinations bit-identical
/// because they have no such field.
#[derive(Debug, Clone, Serialize)]
struct TieredRecord<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    turns: &'a [Turn],
    group: &'a str,
    budget: Tier,
    think_tokens: usize,
}

// ── what the conversations are about ────────────────────────────────────────

// Subject matter, grouped by domain.
//
// **Why this does not reuse the crate root's `TOPICS`.** That list is twenty-six
// entries and every one of them is model training, tokenizers, CUDA or reading a
// codebase — it was written for a corpus about this project. Generating from it
// produced exactly what it describes: the first real run came back with six
// conversations about training logs and loss spikes, and nothing else. A model
// fine-tuned on that answers one kind of question and pattern-matches everything
// else onto it.
//
// The voice stays the same — terse, dense, specific, markup-rich, which is what the
// gold set actually teaches — while the subject matter spans what someone would
// really ask a chat assistant. Software is still here, as one domain of thirteen
// rather than the whole list.
//
// Grouped rather than flat so a run can *guarantee* spread: `pick`ing three entries
// out of one long list would keep landing three software topics by chance, whereas
// rotating the domain per call means a hundred triples cover every domain roughly
// eight times each.
const DOMAINS: &[(&str, &[&str])] = &[
    (
        "software and debugging",
        &[
            "an error or stack trace the user pasted, and what actually caused it",
            "a shell one-liner or script doing the wrong thing",
            "a git problem: a bad merge, a lost commit, messy history",
            "reading an unfamiliar codebase and explaining its layout",
            "a regex or parser that misses an edge case",
            "profiling something slow and finding the real bottleneck",
            "choosing between two implementation approaches under a constraint",
            "a flaky test and what it actually proves",
        ],
    ),
    (
        "machine learning",
        &[
            "a training run that diverged, stalled, or spiked",
            "tokenizer, vocabulary, or BPE behaviour",
            "learning-rate, batch-size, or schedule tuning",
            "explaining an architecture detail (attention, RoPE, MoE, LoRA)",
            "whether a paper's technique is worth trying",
        ],
    ),
    (
        "mathematics and logic",
        &[
            "a probability or combinatorics question with a counterintuitive answer",
            "a unit-conversion or dimensional-analysis problem",
            "an algebra or calculus step the user is stuck on",
            "a statistics question about what a number does and does not show",
            "a puzzle or brainteaser needing a careful case split",
            "compound interest, growth rates, or an amortization figure",
        ],
    ),
    (
        "science and how things work",
        &[
            "why an everyday physical phenomenon happens",
            "a biology or medicine mechanism explained plainly",
            "a chemistry question about a reaction or material",
            "how a common piece of technology works internally",
            "an astronomy or earth-science question of scale",
        ],
    ),
    (
        "history, geography and society",
        &[
            "why a historical event happened, and the popular version being wrong",
            "comparing two periods, places, or systems of government",
            "the origin of a word, custom, border, or institution",
            "a geography question about climate, terrain, or population",
        ],
    ),
    (
        "cooking and food",
        &[
            "a recipe that failed and the technique behind why",
            "substituting an ingredient without ruining the result",
            "food safety, storage, or reheating done properly",
            "why a cooking method works (browning, resting, emulsions, leavening)",
            "scaling a recipe, or converting between weight and volume",
        ],
    ),
    (
        "writing and language",
        &[
            "tightening a paragraph the user pasted, and why the cut works",
            "a grammar or usage question with a real answer, not a superstition",
            "the difference between two words people confuse",
            "structuring a document, email, or argument",
            "translating or explaining an idiom that does not map cleanly",
        ],
    ),
    (
        "money, work and admin",
        &[
            "reading a bill, invoice, statement, or contract clause",
            "budgeting arithmetic, or comparing two offers",
            "a workplace situation needing a specific, sayable sentence",
            "how a tax, insurance, or benefit mechanism works in general terms",
            "a spreadsheet formula doing the wrong thing",
        ],
    ),
    (
        "health, fitness and sleep",
        &[
            "how a training or exercise principle actually works",
            "a nutrition claim and what the evidence supports",
            "sleep, fatigue, or recovery mechanisms",
            "reading a lab or fitness metric and what it means in general",
        ],
    ),
    (
        "home, repair and making",
        &[
            "diagnosing a household appliance or fixture that stopped working",
            "a measurement, material, or tool choice for a small build",
            "why something in the house smells, leaks, drips, or fails",
            "a gardening or plant-care problem with a diagnosable cause",
        ],
    ),
    (
        "travel and logistics",
        &[
            "planning a route or connection under a tight constraint",
            "documents, timing, or rules for crossing a border",
            "packing, luggage, or transport arithmetic",
            "comparing two ways of getting somewhere on cost and time",
        ],
    ),
    (
        "music, art and games",
        &[
            "music theory behind why a progression or tuning works",
            "a technique question about drawing, photography, or mixing",
            "strategy or probability in a specific game",
            "how an instrument, format, or medium physically works",
        ],
    ),
    (
        "everyday reasoning and planning",
        &[
            "breaking a vague goal into a first concrete step",
            "a decision with a real tradeoff and no clean answer",
            "estimating something with no exact data available (Fermi style)",
            "a scheduling or ordering problem with hard constraints",
            "the user's premise is wrong and saying so plainly is the answer",
        ],
    ),
];

// The markup a call is told to exercise, least-used first.
//
// Coverage-driven rather than random, because random is what left `list`, `item`,
// `quote`, `link` and `italic` out of the first run entirely. Every one of these is
// a reserved token with its own embedding row, and a row the corpus never contains
// is a row that never trains — the exact hazard `warm_start_embeddings` exists to
// mitigate after the fact. Asking for the tags used least so far turns coverage
// from something to hope for into something the loop drives.
const MARKUP: &[&str] = &[
    "a bulleted <|list|> of <|item|>s",
    "a <|codeblock|> block (no language attribute — just the code inside)",
    "<|code|> inline code for a path, command, flag, symbol, or number",
    "<|bold|> on the one or two key terms",
    "<|italic|> for light emphasis",
    "sections under a <|heading|> and a <|subheading|>",
    "a <|divider|> between two distinct sections",
    "a <|quote|> block quoting the user's own words or an error string back",
    "a <|warn|> callout for the one caveat that would bite the user if missed",
    "a <|link|> to a plausible reference page",
    "a <|title|> at the very top, for a long structured answer only",
    "<|par|> paragraphs wrapping the prose",
];

// ── prompting ───────────────────────────────────────────────────────────────

// Word targets per tier, as the prompt states them.
//
// Words, not tokens, because a model cannot count its own tokens — the same
// reason `Budget::word_capacity` exists. Tokens are used on the way back in, to
// classify what actually arrived.
// Spread much wider than the first attempt's 15-40 / 90-160 / 220-380, because
// those did not separate. Measured on the first real run, the teacher wrote 15-24
// words for "short" and 22-92 for "medium" — overlapping ranges, so a 24-word block
// landed in `low` and a 22-word block in `med`. The tiers were being decided by
// tokenizer noise rather than by reasoning depth, which is the one thing this whole
// mechanism has to get right.
//
// A ~3 tokens-per-word ratio (measured on that run: 2.3-4.5, noisy because paths
// and numbers tokenize unevenly in a small BPE) puts these at roughly 30-75,
// 330-570 and 960-1500 student tokens — three bands with real gaps between them,
// which is what `default_low_max` and `default_med_max` are set against.
// The token band a tier occupies, given the resolved bounds and the ceiling the
// window allows. `High` is bounded above by the ceiling rather than open-ended:
// an unbounded top tier is what produced 1000-token blocks that could not fit.
fn tier_tokens(tier: Tier, low_max: usize, med_max: usize, ceiling: usize) -> (usize, usize) {
    match tier {
        Tier::Low => (low_max * 30 / 100, low_max * 80 / 100),
        Tier::Med => {
            let span = med_max.saturating_sub(low_max);
            (low_max + span * 30 / 100, low_max + span * 80 / 100)
        }
        Tier::High => (med_max + med_max / 10, ceiling.max(med_max + med_max / 2)),
    }
}

// The same band in words, which is the only unit the teacher can hold itself to —
// it cannot count its own tokens, the reason `Budget::word_capacity` exists.
//
// `words_per_token` is *measured* on the exemplars rather than assumed. The
// crate root learned this the hard way: a hardcoded 3.5 characters per token is
// more than double the truth for a tokenizer trained on a small corpus, and the
// prompt ended up advertising twice the room that existed.
fn tier_words(
    tier: Tier,
    low_max: usize,
    med_max: usize,
    ceiling: usize,
    words_per_token: f64,
) -> (usize, usize) {
    let (lo, hi) = tier_tokens(tier, low_max, med_max, ceiling);
    let to_words = |t: usize| ((t as f64) * words_per_token).round().max(1.0) as usize;
    (to_words(lo), to_words(hi))
}

// Words per student token, measured on the exemplars. Falls back to a third — the
// ratio measured on this corpus — when there is no student to measure with.
fn words_per_token(budget: Option<&Budget>, exemplars: &[&Conversation]) -> f64 {
    let Some(b) = budget else { return 1.0 / 3.0 };
    let usable = (b.ctx_window as f64) * 0.9;
    let capacity = b.word_capacity(exemplars) as f64;
    if usable <= 0.0 || capacity <= 0.0 {
        return 1.0 / 3.0;
    }
    capacity / usable
}

// The instruction block describing all three answers.
//
// The critical constraint is stated explicitly: **all three replies must be
// equally correct and complete**, and only the reasoning varies. Without it a
// teacher writes a deliberately worse answer under the short budget, and the
// student then learns "low budget means give up" rather than "low budget means
// reason briefly". That is the difference between a length lever and a quality
// dial, and it is invisible in the output until you read the low-tier replies
// side by side with the high-tier ones.
fn render_tiers(low_max: usize, med_max: usize, ceiling: usize, wpt: f64) -> String {
    let mut out = String::new();
    out.push_str(
        "\nTHE THREE ANSWERS — this is the core of the task.\n\
         Answer the SAME user question three times. The three replies must be\n\
         equally correct, equally complete and equally useful. What changes\n\
         between them is ONLY how much reasoning the thinking block shows:\n\n",
    );
    for tier in Tier::ALL {
        let (lo, hi) = tier_words(tier, low_max, med_max, ceiling, wpt);
        let guidance = match tier {
            Tier::Low => {
                "state the conclusion and the single fact it rests on. No \
                 alternatives, no exploration. Go straight there. The REPLY is \
                 still fully marked up — a short answer is not a bare sentence"
            }
            Tier::Med => {
                "name the likely cause, weigh one alternative against it, and say \
                 why the alternative loses."
            }
            Tier::High => {
                "work it properly: what the details imply, two or three candidate \
                 explanations weighed against each other with at least one \
                 explicitly rejected and the reason why, an edge case or failure \
                 mode, then a decision about what to say and what to leave out."
            }
        };
        out.push_str(&format!(
            "  {delim}\n    thinking block of roughly {lo}-{hi} words — {guidance}\n",
            delim = delimiter(tier),
            lo = lo,
            hi = hi,
            guidance = guidance,
        ));
    }
    out.push_str(
        "\nEvery one of the three replies starts with its thinking block and then\n\
         gives the answer:\n\n  <|thinking|>...reasoning...<|/thinking|>then the reply\n\n\
         The LONG one is genuinely long — several paragraphs of real working thought,\n\
         not the medium one with more adjectives. If you find yourself restating the\n\
         same point, go deeper instead: what would make this answer wrong, what did\n\
         you consider and drop, what does a specific detail rule out.\n\
         Never write filler (\"Let me think about this carefully\", \"The user wants\n\
         help with X\"). No <|...|> tags inside a thinking block; plain prose only.\n\
         The reply AFTER the block must stand on its own.\n",
    );
    out
}

// An exemplar, rendered in a shape that is **not** the output contract.
//
// `crate::render_exemplar` writes `<<<CONV>>>/<<<USER>>>/<<<ASSISTANT>>>`, which is
// exactly right for the crate root — there the exemplars and the requested output
// are the same format, so showing one teaches the other. Here they are different
// formats, and reusing it was actively teaching the wrong one: the prompt showed
// six to nine conversations ending in `<<<ASSISTANT>>>` and then asked, once, for
// three answers under `<<<SHORT>>>`, `<<<MEDIUM>>>` and `<<<LONG>>>`. A 4B model
// follows the pattern it saw nine times.
//
// Measured: 2 of 4 probe calls emitted no tier delimiter at all, one of them
// answering under a literal `<<<ASSISTANT>>>` it was never given. Those calls
// produce nothing — which is most of the loss left after the markup repair.
//
// So the examples use no `<<<…>>>` token of any kind, leaving the tier delimiters
// as the only things in the prompt wearing that shape.
fn render_plain_exemplar(conv: &Conversation, n: usize) -> String {
    let mut out = format!("--- EXAMPLE {n} ---\n");
    for turn in &conv.turns {
        let who = if turn.role == "user" { "QUESTION" } else { "ANSWER" };
        out.push_str(&format!("{who}:\n{}\n", turn.text.trim()));
    }
    out
}

// A worked example of the three-answer format, with the escalation visible.
//
// The tier instructions describe the long block as "several paragraphs of real
// working thought" and, before this, never showed one. Telling a small model to
// write 320-500 words of reasoning is far weaker than showing it the shape once —
// and the same example doubles as the only place in the prompt where the exact
// output contract appears filled in rather than described.
//
// Deliberately about a boiled egg. It has to demonstrate format and escalation
// without contributing subject matter, and anything technical here would land
// straight back in the corpus as another software conversation.
const WORKED_EXAMPLE: &str = "\
Here is the shape, filled in, for the question \"why did my boiled egg crack\":\n\
\n\
<<<SHORT>>>\n\
<|thinking|>Cracking on entry means thermal shock — cold egg into boiling water.<|/thinking|>\
<|par|>The egg went in cold. A fridge-cold shell hits boiling water and the outside \
expands faster than the inside, so it splits.<|/par|><|par|>Start them in cold water, or \
let them sit out 10 minutes first.<|/par|>\n\
\n\
<<<MEDIUM>>>\n\
<|thinking|>Two candidates. Thermal shock from a cold egg meeting boiling water is the \
common one and fits a crack that appears immediately. The other is the air cell at the \
blunt end expanding as it warms with nowhere to go, which fits a crack that appears a \
minute or two in. Thermal shock is the better bet unless they noticed it happening \
late.<|/thinking|>\
<|par|>Almost certainly thermal shock. A fridge-cold shell dropped into boiling water \
expands unevenly and splits.<|/par|>\
<|list|><|item|>Start the eggs in cold water and bring it up together.<|/item|>\
<|item|>Or let them come to room temperature for 10 minutes first.<|/item|>\
<|item|>Or pierce the blunt end, which vents the air cell.<|/item|><|/list|>\
<|par|>If the crack appeared a minute or two <|italic|>after<|/italic|> they went in, it \
was the air cell rather than the shock — same fix, different cause.<|/par|>\n\
\n\
<<<LONG>>>\n\
<|thinking|>Three mechanisms crack a shell, and they are told apart by timing, which is \
the useful thing to give them. Thermal shock: the shell conducts heat poorly, so a \
fridge-cold egg in boiling water expands outside-first and splits within seconds. The air \
cell: the pocket at the blunt end grows as an egg ages, and heating expands it with \
nowhere to vent, which takes a minute or two — so older eggs crack more, cutting against \
the usual advice to use them for easy peeling. Mechanical: knocked against the pan or \
each other by a rolling boil, cracking at one point. Reject the salt claim; at kitchen \
quantities it moves the boiling point a fraction of a degree. Lead with the timing test, \
name the cold egg as the likely cause rather than hedging, then fix each.<|/thinking|>\
<|par|>Three things crack a boiled egg, and <|bold|>when<|/bold|> it cracked tells you \
which one you had.<|/par|>\
<|subheading|>Within seconds — thermal shock<|/subheading|>\
<|par|>The most likely one. A shell conducts heat poorly, so a fridge-cold egg dropped \
into boiling water expands on the outside before the inside catches up, and splits along \
whatever flaw was already there.<|/par|>\
<|subheading|>A minute or two in — the air cell<|/subheading|>\
<|par|>Every egg has an air pocket at the blunt end, and it grows as the egg ages. Heat \
expands the gas with nowhere to vent.<|/par|>\
<|warn|>This is why older eggs crack more — which cuts against the usual advice to use \
older eggs because they peel more easily.<|/warn|>\
<|subheading|>At one spot on the shell — mechanical<|/subheading|>\
<|par|>Dropped onto the pan base, or knocked together by a hard rolling boil.<|/par|>\
<|divider|>\
<|par|>All three are fixed the same way: start them in cold water, bring it up together, \
and hold it at a bare simmer rather than a rolling boil. Piercing the blunt end handles \
the air cell specifically.<|/par|>\n\
\n\
Notice that all three answers are equally correct and equally useful. Only the \
depth of the reasoning changes.\n";

// Takes no `TeachRequest`, deliberately. Everything on it is either ours to act on
// rather than the teacher's to know (`count`, `max_calls`, the tier thresholds — a
// budget measured in the *student's* tokens is not a number this model could hold
// itself to) or already spent choosing the exemplars. The word targets it does
// state come from `tier_words`.
//
// `call` drives the domain: the index rotates through `DOMAINS` so a run covers all
// of them evenly instead of hoping random sampling does. `coverage` names the tags
// used least so far, for the same reason.
#[allow(clippy::too_many_arguments)]
fn build_prompt(
    call: usize,
    exemplars: &[&Conversation],
    avoid: &[String],
    coverage: &Coverage,
    bounds: (usize, usize, usize, f64),
    rng: &mut Rng,
) -> String {
    // Round-robin, not random. With thirteen domains and a hundred calls this gives
    // each one about eight conversations; sampling would leave some domains empty and
    // others tripled, which is how the first run came back entirely about training
    // logs.
    let (domain, topics) = DOMAINS[call % DOMAINS.len()];
    let topic = topics[rng.below(topics.len())];
    let shapes = pick(SHAPES, 2, rng).join("; ");
    let markup = coverage.least_used(3).join("; ");

    // Rendered without `<<<…>>>` delimiters — see `render_plain_exemplar`.
    let exemplar_text: Vec<String> = exemplars
        .iter()
        .enumerate()
        .map(|(n, c)| render_plain_exemplar(c, n + 1))
        .collect();

    let avoid_block = if avoid.is_empty() {
        String::new()
    } else {
        let mut out = String::from(
            "\nQuestions already in the corpus — write about something ELSE, not a\n\
             reworded version of any of these:\n",
        );
        for opener in avoid {
            out.push_str(&format!("  - {opener}\n"));
        }
        out
    };

    format!(
        "You are generating supervised training data for a small general-purpose chat\n\
         assistant that answers in a WEB CHAT APP. It is asked about anything — code,\n\
         cooking, history, maths, money, health, repairs — and it answers like a sharp,\n\
         well-read expert on that subject: dense, specific, no filler, no hedging.\n\
         \n\
         Its only output channel is the markup vocabulary below, which the web UI\n\
         renders. Using that vocabulary well IS part of what you are teaching it.\n\
         \n\
         Below are REAL answers from the target dataset. Copy their VOICE, length,\n\
         density and markup style exactly — and do NOT copy their subject matter, which\n\
         is given to you separately below. They are written as QUESTION/ANSWER pairs;\n\
         that is NOT the format you output. Your output format is given after them.\n\
         \n\
         {exemplars}\n\
         --- END EXAMPLES ---\n\
         \n\
         Output EXACTLY ONE block in this format, and nothing else — no prose, no\n\
         markdown fences, no commentary before or after:\n\
         \n\
         {conv}\n\
         {user}\n\
         the user's message, as many lines as it needs\n\
         {short}\n\
         the first reply\n\
         {medium}\n\
         the second reply\n\
         {long}\n\
         the third reply\n\
         {end}\n\
         \n\
         Every delimiter sits ALONE on its own line, with nothing else on that line.\n\
         Write turn text literally — real newlines, real quotes, no escaping of any\n\
         kind. Nothing is JSON here.\n\
         {tiers}\
         \n\
         {worked}\
         \n\
         Hard rules (breaking any of these throws the whole block away):\n\
         - the user turn is PLAIN text and realistic: sometimes a pasted brief with\n\
           file paths, logs or requirements; sometimes three terse words with a typo.\n\
           NEVER any <|...|> tokens in the user turn,\n\
         - each reply is under {max_chars} characters, written in the voice of the\n\
           examples: lead with the answer, no \"Great question!\", no restating the ask,\n\
         - reply markup uses ONLY this vocabulary:\n\
             <|par|>a paragraph<|/par|>\n\
             <|list|><|item|>bullet<|/item|><|item|>bullet<|/item|><|/list|>\n\
             <|codeblock|>code here<|/codeblock|>\n\
             <|code|>inline code<|/code|>  <|bold|>bold<|/bold|>  <|italic|>italic<|/italic|>\n\
             <|heading|>section heading<|/heading|>  <|subheading|>smaller heading<|/subheading|>\n\
             <|quote|>a quoted line<|/quote|>  <|divider|>\n\
             <|link|>example.com/page<|/link|>  <|title|>a document title<|/title|>\n\
           Markup is sprinkled rather than total — plain prose between tags is fine —\n\
           but every reply must use at least some of it, and every opened tag must be\n\
           closed. Write every tag BARE, with no attributes. Closing tags are spelled\n\
           <|/par|>: the slash goes INSIDE, after the bar. </|par|> is wrong.\n\
           Never use <|reply|>, <|user|>, <|assistant|>, <|system|> or <|endoftext|>.\n\
           NEVER invent a tag. If the vocabulary above has no tag for what you want,\n\
           write plain prose. In particular no HTML: no <p>, <div>, <b>, <li>, <br>,\n\
           and no <|p|>, <|div|>, <|li|>, <|pipeline|>. Long answers are where this\n\
           slips — the longer the answer, the more carefully every tag must come\n\
           from the list.\n\
         - QUOTE THE USER EXACTLY. Every number, filename, path, timestamp, version and\n\
           identifier the user wrote must appear in your answer CHARACTER FOR CHARACTER,\n\
           or not at all. If they wrote step 1200, it is step 1200 everywhere — not 1202,\n\
           not 1100. If they wrote run_007.jsonl, it is never run_006.jsonl. Changing one\n\
           digit of the user's own data throws the block away. Copy, do not paraphrase.\n\
         \n\
         Content for this block:\n\
         - The subject is {domain}. Specifically: {topic}.\n\
           Write it the way someone in that world would ask, with their vocabulary and\n\
           their details — NOT as a software question wearing a costume.\n\
         - Use one of these shapes for the question: {shapes}.\n\
         - This batch must exercise, where they genuinely fit: {markup}. These are the\n\
           tags the corpus has used least, so prefer them over the ones you would reach\n\
           for by habit. Never force one — a terse question still gets a terse answer.\n\
         - Content must be CORRECT and self-contained. Invent plausible names, places,\n\
           paths and numbers; no real personal data about real people, no current events.\n\
         {avoid}\
         \n\
         Output the block now, starting with {conv}:",
        exemplars = exemplar_text.join("\n\n"),
        conv = crate::D_CONV,
        user = crate::D_USER,
        short = delimiter(Tier::Low),
        medium = delimiter(Tier::Med),
        long = delimiter(Tier::High),
        end = crate::D_END,
        tiers = render_tiers(bounds.0, bounds.1, bounds.2, bounds.3),
        worked = WORKED_EXAMPLE,
        max_chars = MAX_ASSISTANT_CHARS,
        domain = domain,
        topic = topic,
        shapes = shapes,
        markup = markup,
        avoid = avoid_block,
    )
}

// How many times each markup instruction has been asked for, so a call can ask for
// the ones that have not been.
//
// A count per `MARKUP` entry rather than per tag: the entries are what the prompt
// says, and `<|list|>`/`<|item|>` travel together. Ties break toward the earlier
// entry, which keeps a run deterministic under a fixed seed.
#[derive(Default)]
struct Coverage {
    asked: Vec<usize>,
}

impl Coverage {
    fn new() -> Coverage {
        Coverage {
            asked: vec![0; MARKUP.len()],
        }
    }

    // The `k` least-asked-for instructions, and records that they were asked.
    fn least_used(&self, k: usize) -> Vec<&'static str> {
        let mut order: Vec<usize> = (0..MARKUP.len()).collect();
        order.sort_by_key(|&i| self.asked[i]);
        order.into_iter().take(k).map(|i| MARKUP[i]).collect()
    }

    // Counted on what actually came back, not on what was requested — the point is
    // coverage of the corpus, and a tag the teacher was asked for and ignored has
    // not been covered.
    fn record(&mut self, reply: &str) {
        for (i, entry) in MARKUP.iter().enumerate() {
            // The instruction text carries the tags it is about, so the tags to look
            // for are read off the entry itself rather than listed a second time.
            let mut rest = *entry;
            while let Some(start) = rest.find("<|") {
                let after = &rest[start + 2..];
                let Some(end) = after.find("|>") else { break };
                let tag = &after[..end];
                if !tag.starts_with('/') && reply.contains(&format!("<|{tag}|>")) {
                    self.asked[i] += 1;
                    break;
                }
                rest = &after[end + 2..];
            }
        }
    }
}

// The system prompt for the teacher itself — not the student's persona, just an
// instruction to obey the format. Picked from the same list the crate root uses
// so the two paths do not drift.
fn teacher_system(rng: &mut Rng) -> String {
    format!(
        "{} You follow output formats exactly and emit no commentary outside them.",
        SYSTEM_PROMPTS[rng.below(SYSTEM_PROMPTS.len())]
    )
}

// ── parsing ─────────────────────────────────────────────────────────────────

/// A parsed triple, before validation.
struct RawTriple {
    user: String,
    // Requested tier → reply text, in the order the delimiters appeared.
    answers: Vec<(Tier, String)>,
}

// Pulls the one expected block out of the teacher's response.
//
// Tolerant of the two things a small local model reliably gets wrong — leading
// commentary before the first delimiter, and a missing final `<<<END>>>` when the
// response was truncated — because both are recoverable and rejecting them
// outright would throw away most of a run's output.
fn split_triple(text: &str) -> Result<RawTriple, String> {
    let start = text
        .find(crate::D_CONV)
        .ok_or_else(|| "no conversation block in the response".to_string())?;
    let body = &text[start + crate::D_CONV.len()..];
    let body = match body.find(crate::D_END) {
        Some(i) => &body[..i],
        None => body,
    };

    let user_at = body
        .find(crate::D_USER)
        .ok_or_else(|| "no user turn in the block".to_string())?;
    let after_user = &body[user_at + crate::D_USER.len()..];

    // Find each tier delimiter and cut at the earliest one that follows.
    let mut marks: Vec<(usize, Tier)> = Vec::new();
    for tier in Tier::ALL {
        if let Some(i) = after_user.find(delimiter(tier)) {
            marks.push((i, tier));
        }
    }
    if marks.is_empty() {
        return Err("no tier answers in the block".to_string());
    }
    marks.sort_by_key(|(i, _)| *i);

    let user = after_user[..marks[0].0].trim().to_string();
    if user.is_empty() {
        return Err("empty user turn".to_string());
    }

    let mut answers = Vec::new();
    for (n, (at, tier)) in marks.iter().enumerate() {
        let from = at + delimiter(*tier).len();
        let to = marks.get(n + 1).map(|(i, _)| *i).unwrap_or(after_user.len());
        if to < from {
            continue;
        }
        let reply = after_user[from..to].trim();
        if !reply.is_empty() {
            answers.push((*tier, repair_markup(reply)));
        }
    }
    if answers.is_empty() {
        return Err("every tier answer was empty".to_string());
    }

    Ok(RawTriple { user, answers })
}

// ── the teacher must not rewrite the user's own facts ───────────────────────

// Tokens in a user turn that the reply is not allowed to alter: paths, filenames,
// timestamps, and numbers of three digits or more.
//
// Short numbers are excluded deliberately. "2" and "16" appear in prose for a
// hundred unrelated reasons and matching on them would flag every reply that
// mentions a batch size; three digits is where a number starts being an identifier
// the user typed rather than a quantity the answer reasoned to.
fn salient_tokens(text: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for raw in text.split_whitespace() {
        let token = raw.trim_matches(|c: char| {
            c != '/' && c != '.' && c != '_' && c != ':' && c != '-' && !c.is_alphanumeric()
        });
        if token.is_empty() {
            continue;
        }
        let digits = token.chars().filter(char::is_ascii_digit).count();
        let looks_like_path = token.contains('/') || token.contains('_') || token.contains('.');
        if digits >= 3 || (digits > 0 && looks_like_path) {
            out.insert(token.to_string());
        }
    }
    out
}

// Whether two tokens are close enough that one is plausibly a corruption of the
// other rather than a different fact.
//
// **Equal length is required**, and that is the rule doing most of the work. A
// misquote is a substitution — `1200` → `1202`, `run_007` → `run_006`, `14:23:17` →
// `14:22:17` — whereas a genuinely different number the answer reasoned to is
// usually a different length. Without this gate, `1200` and `500` come out two edits
// apart and every reply that mentions a round number gets thrown away; the test
// `inventing_unrelated_numbers_is_allowed` is that false positive, caught.
//
// The allowance then scales with length: one edit on a short numeric token, two on
// something path-shaped where there is more to get wrong. Deliberately tuned for
// precision over recall — `1420` → `2120` (two edits, four characters) slips
// through, and that is the right way to be wrong here. A false rejection costs one
// generation; a false acceptance is silent, and the whole point is that this class of
// defect does not announce itself.
const LONG_TOKEN: usize = 6;

fn is_near_miss(a: &str, b: &str) -> bool {
    if a == b || a.len() != b.len() {
        return false;
    }
    let allowed = if a.len() >= LONG_TOKEN { 2 } else { 1 };
    // Levenshtein, iterative single-row.
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let distance = prev[b.len()];
    distance > 0 && distance <= allowed
}

/// Rejects a reply that quotes one of the user's identifiers back *wrong*.
///
/// This is the failure the first real run produced on every single record: asked
/// about "step 1200" in `run_007.jsonl`, the teacher answered about step 1202 in
/// `run_006.jsonl`; asked about 14:23:17, it answered about 14:14:30. A small model
/// treats a number in its context as a thing to paraphrase rather than a thing to
/// copy.
///
/// It cannot be fixed by prompting alone, and it is the one defect that makes the
/// data worse than useless: an SFT corpus of near-miss quotations teaches the model
/// that the numbers in its own context window are approximate. So it is a
/// structural check, next to the tag balance.
///
/// Deliberately one-directional — a reply may introduce identifiers of its own (it
/// invents paths and figures, which is fine) and may omit any of the user's. Only a
/// token that is *almost* one of the user's, and is not any of them, is a rewrite.
fn check_quoted_facts(user: &str, reply: &str) -> Result<(), String> {
    let theirs = salient_tokens(user);
    if theirs.is_empty() {
        return Ok(());
    }
    for candidate in salient_tokens(reply) {
        if theirs.contains(&candidate) {
            continue;
        }
        if let Some(original) = theirs.iter().find(|t| is_near_miss(t, &candidate)) {
            return Err(format!(
                "changed the user's own {original} into {candidate}"
            ));
        }
    }
    Ok(())
}

// ── classification ──────────────────────────────────────────────────────────

/// Which tier a reply's thinking block actually lands in.
///
/// Filed by **measurement**, not by what was asked for. A teacher told to write
/// 15-40 words routinely writes 70, and discarding those would throw away good
/// data to defend a label that is ours to assign — so the record is relabelled to
/// wherever it fell and the disagreement is counted. The label then describes the
/// content, which is the only thing that makes it trainable.
fn classify(think_tokens: usize, low_max: usize, med_max: usize) -> Tier {
    if think_tokens <= low_max {
        Tier::Low
    } else if think_tokens <= med_max {
        Tier::Med
    } else {
        Tier::High
    }
}

// Thinking tokens in the *student's* tokenizer.
//
// `thinking_block` returns a word count, which is the right unit for instructing
// the teacher and the wrong one for a budget: a token budget is what the student
// spends at inference, and this tokenizer is trained on a small corpus with short
// merges, so its token-per-word ratio is nothing like a large tokenizer's. With
// no student named there is nothing to measure with, and the caller falls back to
// the requested tier.
fn think_tokens(reply: &str, budget: Option<&Budget>) -> Option<usize> {
    let (words, body_start) = thinking_block(reply);
    if body_start == 0 || words == 0 {
        return Some(0);
    }
    // The open marker is located rather than assumed to sit at byte 0:
    // `thinking_block` tolerates leading whitespace before it, so slicing from a
    // fixed offset would take the marker's own bytes into the block on any reply
    // that starts with a newline.
    let open_at = reply.find(crate::THINK_OPEN)? + crate::THINK_OPEN.len();
    let close_at = body_start.checked_sub(crate::THINK_CLOSE.len())?;
    let inner = reply.get(open_at..close_at)?;
    budget.map(|b| b.text_tokens(inner))
}

// ── the run ─────────────────────────────────────────────────────────────────

/// Generates thinking-budget triples with a local teacher until `count` are
/// accepted, the call budget runs out, or the job is cancelled.
///
/// Blocking, and deliberately so: llama.cpp decoding is a synchronous CUDA loop,
/// exactly like `llm::train`, and the queue runs those on a plain OS thread that
/// never touches the pool. Progress leaves through the callbacks.
pub fn run(
    request: &TeachRequest,
    on_progress: ProgressCallback,
    on_report: ReportCallback,
    on_triple: TripleCallback,
    cancel: Arc<AtomicBool>,
) -> Result<Option<TeachStats>, String> {
    let dir = storage::sft_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("could not open the SFT directory: {e}"))?;

    let budget = match request.id_model {
        Some(id) => Some(Budget::load(&storage::models_dir().join(id.to_string()))?),
        None => None,
    };

    on_progress("Reading the gold set", 0.0);
    let pool = load_exemplars(&dir.join(&request.exemplar_file), EXEMPLAR_BUDGET);
    if pool.is_empty() {
        return Err(format!(
            "no usable exemplars in {} — generation copies the style of real \
             conversations, and with none to show it produces generic chat data",
            request.exemplar_file
        ));
    }

    on_progress("Loading the teacher onto the GPU", 0.0);
    let teacher = Teacher::load(&request.teacher)?;
    let gpu = teacher.supports_gpu();
    if !gpu {
        // Not fatal — it still works — but a CPU run is ~100x slower, which reads
        // as a hang rather than a misconfiguration, so it is said out loud.
        on_progress(
            "WARNING: this build cannot offload to the GPU; running on the CPU",
            0.0,
        );
    }
    if request.teacher.n_ctx > teacher.train_ctx() {
        return Err(format!(
            "n_ctx is {} but {} was trained for {} — a longer window degrades \
             quality silently rather than erroring",
            request.teacher.n_ctx,
            teacher.path().display(),
            teacher.train_ctx(),
        ));
    }
    let mut session = teacher.session(&request.teacher)?;

    // Dedup against every SFT file, so a generated question never echoes the gold
    // set or an earlier tier file either.
    let mut seen = load_openers(&dir);
    let (paired, void) = vocabulary();

    // Resolved once: the tier boundaries, the ceiling the window allows, and the
    // measured words-per-token the prompt states its targets in.
    let (low_max, med_max) = resolve_bounds(request, budget.as_ref());
    let ceiling = budget.as_ref().map(|b| thinking_ceiling(b.ctx_window)).unwrap_or(med_max * 2);
    let sample: Vec<&Conversation> = pool.iter().take(8).collect();
    let wpt = words_per_token(budget.as_ref(), &sample);
    let bounds = (low_max, med_max, ceiling, wpt);
    // Say which bounds are in force, and say so *before* generating: this is the
    // number that decides whether the top tier can exist, and when it was wrong it
    // was invisible for two rounds of investigation.
    let overridden = (request.low_max_tokens, request.med_max_tokens) != (0, 0)
        && (request.low_max_tokens, request.med_max_tokens) != (low_max, med_max);
    on_progress(
        &format!(
            "Thinking budgets for this student: low <= {low_max}, med <= {med_max}, \
             high up to {ceiling} tokens{}",
            if overridden {
                format!(
                    " (the requested {}/{} was refused — its top tier cannot fit a \
                     {}-token window)",
                    request.low_max_tokens,
                    request.med_max_tokens,
                    budget.as_ref().map(|b| b.ctx_window).unwrap_or(0),
                )
            } else {
                String::new()
            },
        ),
        0.0,
    );

    let mut rng = Rng::new(request.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5eed)
    }));

    let max_calls = if request.max_calls == 0 {
        request.count.saturating_mul(3).max(1)
    } else {
        request.max_calls
    };

    let mut triples = 0usize;
    // Tracked apart from `triples` because they are different questions about the
    // same run: how many prompts yielded anything, and how many yielded the matched
    // set of three the format exists for.
    let mut complete_triples = 0usize;
    let mut answers_total = 0usize;
    // Indexed by (tiers covered - 1), so a question is always counted somewhere.
    let mut by_tiers = [0usize; 3];
    let mut calls = 0usize;
    let mut duplicates = 0usize;
    let mut relabelled = 0usize;
    let mut rejected: Vec<(String, usize)> = Vec::new();
    let mut per_tier: Vec<(Tier, usize)> = Tier::ALL.iter().map(|t| (*t, 0)).collect();
    let mut coverage = Coverage::new();

    while triples < request.count && calls < max_calls {
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }

        calls += 1;
        let started = Instant::now();

        let exemplars = pick_exemplars(&pool, EXEMPLAR_BUDGET, &mut rng);
        let avoid = recent_openers(&seen, &mut rng);
        let user_prompt = build_prompt(calls, &exemplars, &avoid, &coverage, bounds, &mut rng);
        let system = teacher_system(&mut rng);
        let prompt = teacher.chat_prompt(Some(&system), &user_prompt)?;

        // A fresh seed per call, or every call in a run samples the same answer
        // from the same prompt shape.
        let opts = SampleOpts {
            seed: rng.below(u32::MAX as usize) as u32,
            ..request.sample.clone()
        };

        // `&*cancel`, not `&cancel`: the parameter is `Option<&AtomicBool>` and
        // deref coercion does not reach inside an `Option`, so `&cancel` would be
        // an `Option<&Arc<AtomicBool>>`.
        let generated = session.generate(&prompt, &opts, Some(&*cancel), |_| {})?;
        if generated.cancelled {
            return Ok(None);
        }
        // Truncation is tallied rather than ignored, and it is not a generic
        // failure: the answers are emitted shortest-first, so running out of budget
        // costs the LONG tier specifically. Left unreported this shows up as a high
        // tier that will not fill, with nothing pointing at the cause — the fix is
        // a larger `max_tokens` or `n_ctx`, which is only obvious once named.
        if generated.truncated {
            tally(
                &mut rejected,
                "response hit the token budget (the long answer is cut)".to_string(),
            );
        }

        let mut kept_answers = 0usize;
        let mut tiers_covered = 0usize;

        match split_triple(&generated.text) {
            Err(reason) => tally(&mut rejected, reason),
            Ok(raw) => {
                let opener = raw.user.split_whitespace().collect::<Vec<_>>().join(" ");
                let key = opener.to_lowercase();
                if seen.contains(&key) {
                    duplicates += 1;
                } else if raw.user.contains("<|") || raw.user.contains("|>") {
                    tally(&mut rejected, "markup in the user turn".to_string());
                } else {
                    let group = Uuid::new_v4().to_string();
                    let mut answers = Vec::new();

                    for (requested, reply) in &raw.answers {
                        // The tier's own floor is not enforced here: a thinking
                        // block shorter than asked for is a *relabel*, not a
                        // discard. Passing 0 keeps the structural checks — tags
                        // balanced, block in the right place, some markup present
                        // — without the length rejection the crate root applies.
                        if let Err(reason) = validate_reply(reply, 0, &paired, &void) {
                            tally(&mut rejected, reason);
                            continue;
                        }
                        if reply.len() > MAX_ASSISTANT_CHARS {
                            tally(&mut rejected, "reply too long".to_string());
                            continue;
                        }
                        // Checked over the whole reply, thinking block included: the
                        // scratchpad misquoted the user's numbers as often as the
                        // answer did, and reasoning from a wrong figure is the part
                        // that teaches the habit.
                        if let Err(reason) = check_quoted_facts(&raw.user, reply) {
                            tally(&mut rejected, reason);
                            continue;
                        }

                        let measured = think_tokens(reply, budget.as_ref());
                        let filed_as = match measured {
                            Some(n) => classify(n, low_max, med_max),
                            None => *requested,
                        };
                        if filed_as != *requested {
                            relabelled += 1;
                        }

                        let turns = vec![
                            Turn {
                                role: "user".to_string(),
                                text: raw.user.clone(),
                            },
                            Turn {
                                role: "assistant".to_string(),
                                text: reply.clone(),
                            },
                        ];

                        // Length against the student's window, the same check the
                        // crate root makes: a conversation that will not fit is
                        // rejected at generation rather than discovered later as a
                        // window with no assistant token in it.
                        if let Some(b) = budget.as_ref() {
                            let conv = Conversation {
                                system: None,
                                turns: turns.clone(),
                            };
                            if b.tokens(&conv) > b.ctx_window {
                                tally(&mut rejected, "does not fit the context window".to_string());
                                continue;
                            }
                        }

                        let think = measured.unwrap_or(0);
                        append_record(
                            &dir.join(tier_file_name(filed_as)),
                            &TieredRecord {
                                system: None,
                                turns: &turns,
                                group: &group,
                                budget: filed_as,
                                think_tokens: think,
                            },
                        )?;

                        if let Some(slot) = per_tier.iter_mut().find(|(t, _)| *t == filed_as) {
                            slot.1 += 1;
                        }
                        coverage.record(reply);
                        kept_answers += 1;
                        answers.push(AnsweredTier {
                            requested: *requested,
                            filed_as,
                            think_tokens: think,
                            reply_preview: preview(reply),
                        });
                    }

                    // Distinct tiers this question ended up covering. Counted from
                    // `filed_as` rather than from what was requested, because a
                    // relabelled answer lands in the file it measured into — three
                    // answers that all measure `med` are three copies of one budget,
                    // not a triple, and the old counter called them one anyway.
                    tiers_covered = Tier::ALL
                        .iter()
                        .filter(|t| answers.iter().any(|a| a.filed_as == **t))
                        .count();
                    if !answers.is_empty() {
                        seen.insert(key);
                        triples += 1;
                        answers_total += answers.len();
                        if tiers_covered == Tier::ALL.len() {
                            complete_triples += 1;
                        }
                        if tiers_covered >= 1 && tiers_covered <= by_tiers.len() {
                            by_tiers[tiers_covered - 1] += 1;
                        }
                        on_triple(&AcceptedTriple {
                            group,
                            opener,
                            answers,
                        });
                    }
                }
            }
        }

        on_report(&TeachReport {
            call: calls,
            answers_kept: kept_answers,
            tiers_covered,
            duplicates,
            rejected: rejected.clone(),
            elapsed_ms: started.elapsed().as_millis(),
            tokens_generated: generated.tokens,
            total_questions: triples,
            total_answers: answers_total,
            total_complete: complete_triples,
            total_by_tiers: by_tiers.to_vec(),
            target: request.count,
        });
        // Says which unit it is counting, and shows the one that was invisible.
        // "45/250 triples" beside three tier bars summing to 75 is not a number
        // anyone can reconcile, and the reconciliation is the interesting part:
        // most questions are not producing three answers.
        on_progress(
            &format!(
                "{triples}/{} questions · {answers_total} answers · \
                 {complete_triples} full triples",
                request.count
            ),
            triples as f64 / request.count.max(1) as f64,
        );
    }

    Ok(Some(TeachStats {
        questions: triples,
        complete_triples,
        answers: answers_total,
        questions_by_tiers: by_tiers.to_vec(),
        calls,
        duplicates,
        rejected,
        per_tier: per_tier
            .into_iter()
            .map(|(t, n)| (t.as_str().to_string(), n))
            .collect(),
        relabelled,
        measured: budget.is_some(),
        gpu,
    }))
}

// One JSON object per line, appended. Opened per record rather than held open for
// the run: a teaching run is long, and a crash partway through should leave the
// accepted triples on disk rather than in a lost buffer.
fn append_record(path: &Path, record: &TieredRecord) -> Result<(), String> {
    let line = serde_json::to_string(record)
        .map_err(|e| format!("could not serialize a conversation: {e}"))?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("could not open {}: {e}", path.display()))?;
    writeln!(file, "{line}").map_err(|e| format!("could not write {}: {e}", path.display()))
}

// How many previous questions to show, and how much of each. Same reasoning as
// the crate root's avoid-list: the opening clause carries the repetition, and a
// pasted brief in full would cost more of the window than the exemplars do.
const AVOID_OPENERS: usize = 10;
const OPENER_PREVIEW_CHARS: usize = 90;

fn recent_openers(seen: &HashSet<String>, rng: &mut Rng) -> Vec<String> {
    if seen.is_empty() {
        return Vec::new();
    }
    // A sample rather than the newest, because the corpus runs to thousands and a
    // fixed slice of it would steer every call away from the same ten questions
    // while leaving the rest of the space just as crowded.
    let all: Vec<&String> = seen.iter().collect();
    let mut out = Vec::new();
    let mut used = HashSet::new();
    for _ in 0..AVOID_OPENERS.min(all.len()) {
        let i = rng.below(all.len());
        if used.insert(i) {
            out.push(truncate(all[i], OPENER_PREVIEW_CHARS));
        }
    }
    out
}

fn preview(reply: &str) -> String {
    let (_, body_start) = thinking_block(reply);
    truncate(reply[body_start..].trim(), 120)
}

// Truncates on a character boundary — `&s[..n]` panics mid-codepoint, and model
// output is full of box-drawing characters and arrows.
fn truncate(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &flat[..i]),
        None => flat,
    }
}

// ── the probe ───────────────────────────────────────────────────────────────

/// Runs `calls` real teacher calls and reports, per call, **where each of the
/// three answers went** — parsed or not, and if parsed, which check rejected it.
///
/// This exists because a run's only record of that is the `rejected` tally on
/// `TeachStats`, which is broadcast to the live feed and never persisted: once
/// the job row is written, the reason a tier stayed empty is gone. Diagnosing
/// "the high tier will not fill" from the accepted output alone is impossible —
/// the accepted output is, by definition, the answers that did not fail.
///
/// Returns one line per call for the caller to print. Deliberately allocates no
/// files and writes nothing to `raw/sft/`: it is a measurement, not a run.
pub fn probe(request: &TeachRequest, calls: usize, dump_to: Option<&Path>) -> Result<Vec<String>, String> {
    let dir = storage::sft_dir();
    let budget = match request.id_model {
        Some(id) => Some(Budget::load(&storage::models_dir().join(id.to_string()))?),
        None => None,
    };
    let pool = load_exemplars(&dir.join(&request.exemplar_file), EXEMPLAR_BUDGET);
    if pool.is_empty() {
        return Err(format!("no usable exemplars in {}", request.exemplar_file));
    }

    let teacher = Teacher::load(&request.teacher)?;
    let mut session = teacher.session(&request.teacher)?;
    let (paired, void) = vocabulary();
    let mut rng = Rng::new(request.seed.unwrap_or(0x5eed));
    let coverage = Coverage::new();
    let cancel = Arc::new(AtomicBool::new(false));

    let mut out = Vec::new();
    for call in 1..=calls {
        let exemplars = pick_exemplars(&pool, EXEMPLAR_BUDGET, &mut rng);
        let sample: Vec<&Conversation> = pool.iter().take(8).collect();
        let (low_max, med_max) = resolve_bounds(request, budget.as_ref());
        let ceiling = budget
            .as_ref()
            .map(|b| thinking_ceiling(b.ctx_window))
            .unwrap_or(med_max * 2);
        let bounds = (low_max, med_max, ceiling, words_per_token(budget.as_ref(), &sample));
        let user_prompt = build_prompt(call, &exemplars, &[], &coverage, bounds, &mut rng);
        let system = teacher_system(&mut rng);
        let prompt = teacher.chat_prompt(Some(&system), &user_prompt)?;
        let opts = SampleOpts {
            seed: rng.below(u32::MAX as usize) as u32,
            ..request.sample.clone()
        };

        let started = Instant::now();
        let generated = session.generate(&prompt, &opts, Some(&*cancel), |_| {})?;
        // The raw response, so the exact malformed construct can be read rather
        // than inferred from a rejection reason.
        if let Some(dir) = dump_to {
            let _ = fs::create_dir_all(dir);
            let _ = fs::write(dir.join(format!("call{call}.txt")), &generated.text);
        }

        let mut line = format!(
            "call {call}: prompt {} chars, generated {} tokens in {:?}{}\n",
            prompt.len(),
            generated.tokens,
            started.elapsed(),
            if generated.truncated { "  ** TRUNCATED **" } else { "" },
        );
        // Which delimiters the model actually emitted — the first thing that
        // decides whether three answers even exist to be judged.
        for tier in Tier::ALL {
            let d = delimiter(tier);
            line.push_str(&format!(
                "    {d}: {}\n",
                if generated.text.contains(d) { "present" } else { "MISSING" },
            ));
        }

        line.push_str(&verdicts(&generated.text, request, budget.as_ref(), &paired, &void));
        out.push(line);
    }
    Ok(out)
}

/// Re-parses raw responses saved by a previous `probe` run, without a GPU.
///
/// This is what makes a parser change measurable: the same bytes through the old
/// and new code is a true A/B, where re-generating would confound the parser with
/// fresh sampling. Reads `call*.txt` from `dir` in name order.
pub fn replay(request: &TeachRequest, dir: &Path) -> Result<Vec<String>, String> {
    let budget = match request.id_model {
        Some(id) => Some(Budget::load(&storage::models_dir().join(id.to_string()))?),
        None => None,
    };
    let (paired, void) = vocabulary();

    let mut files: Vec<_> = fs::read_dir(dir)
        .map_err(|e| format!("could not read {}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("txt"))
        .collect();
    files.sort();

    let mut out = Vec::new();
    for path in files {
        let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut line = format!("{}:\n", path.file_name().unwrap_or_default().to_string_lossy());
        line.push_str(&verdicts(&text, request, budget.as_ref(), &paired, &void));
        out.push(line);
    }
    Ok(out)
}

// The first `<|…|>` in a reply that is not a well-formed, known tag, with a little
// context either side. Best-effort: it re-walks the reply the way `tags_in` does
// and reports the first thing that would stop it.
fn offending_fragment(reply: &str) -> String {
    let names = crate::vocabulary();
    let mut rest = reply;
    let mut at = 0usize;
    while let Some(start) = rest.find("<|") {
        let abs = at + start;
        match rest[start + 2..].find("|>") {
            Some(end) => {
                let tag = &rest[start + 2..start + 2 + end];
                let known = names.0.contains(tag)
                    || names.1.contains(tag)
                    || tag.strip_prefix('/').is_some_and(|n| names.0.contains(n))
                    || tag.contains("thinking");
                if !known || tag.contains('|') || tag.contains('=') {
                    return snippet(reply, abs);
                }
                at = abs + 4 + end;
                rest = &reply[at..];
            }
            None => return snippet(reply, abs),
        }
    }
    "(no single tag identified)".to_string()
}

// The rendered conversation's token count when it busts the student's window, or
// `None` when it fits (or when there is no student to measure against).
fn over_window(user: &str, reply: &str, budget: Option<&Budget>) -> Option<usize> {
    let b = budget?;
    let conv = Conversation {
        system: None,
        turns: vec![
            Turn { role: "user".to_string(), text: user.to_string() },
            Turn { role: "assistant".to_string(), text: reply.to_string() },
        ],
    };
    let tokens = b.tokens(&conv);
    (tokens > b.ctx_window).then_some(tokens)
}

fn snippet(reply: &str, at: usize) -> String {
    let lo = reply[..at].char_indices().rev().nth(30).map(|(i, _)| i).unwrap_or(0);
    let hi = reply[at..]
        .char_indices()
        .nth(70)
        .map(|(i, _)| at + i)
        .unwrap_or(reply.len());
    format!("{:?}", &reply[lo..hi])
}

// One response's worth of per-answer verdicts, shared by `probe` and `replay` so
// a live call and a replayed one are judged by identical code.
fn verdicts(
    text: &str,
    request: &TeachRequest,
    budget: Option<&Budget>,
    paired: &HashSet<&'static str>,
    void: &HashSet<&'static str>,
) -> String {
    let mut line = String::new();
    match split_triple(text) {
        Err(reason) => line.push_str(&format!("    split failed: {reason}\n")),
        Ok(raw) => {
            line.push_str(&format!("    parsed {} answers\n", raw.answers.len()));
            for (requested, reply) in &raw.answers {
                let (words, _) = thinking_block(reply);
                let measured = think_tokens(reply, budget).unwrap_or(0);
                let verdict = if let Err(e) = validate_reply(reply, 0, paired, void) {
                    // The offending fragment, not just the reason. A rejection
                    // reason names a *class* of defect; fixing one needs the
                    // actual bytes, and these are post-repair so they show what
                    // the repair did not reach.
                    format!("REJECTED (markup: {e}) near {}", offending_fragment(reply))
                } else if reply.len() > MAX_ASSISTANT_CHARS {
                    format!("REJECTED (too long: {} chars)", reply.len())
                } else if let Err(e) = check_quoted_facts(&raw.user, reply) {
                    format!("REJECTED (quoted facts: {e})")
                } else if let Some(over) = over_window(&raw.user, reply, budget) {
                    // The check `run` applies and this instrument did not. Omitting
                    // it made the probe report answers as kept that the real
                    // pipeline discards — which is exactly how a tier that could
                    // never be written looked like a markup problem.
                    format!("REJECTED (does not fit the {} window: {over} tokens)", 
                        budget.map(|b| b.ctx_window).unwrap_or(0))
                } else {
                    let (low_max, med_max) = resolve_bounds(request, budget);
                    format!("kept, filed as {}", classify(measured, low_max, med_max).as_str())
                };
                line.push_str(&format!(
                    "      {:6} think {words:4} words / {measured:4} tokens, reply {:5} chars — {verdict}\n",
                    requested.as_str(),
                    reply.len(),
                ));
            }
        }
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> TeachRequest {
        TeachRequest {
            teacher: TeacherConfig::default(),
            sample: SampleOpts::default(),
            count: 1,
            low_max_tokens: 64,
            med_max_tokens: 256,
            id_model: None,
            exemplar_file: default_exemplar_file(),
            max_calls: 0,
            seed: Some(1),
        }
    }

    #[test]
    fn splits_a_well_formed_triple() {
        let text = "\
<<<CONV>>>
<<<USER>>>
why is my loss spiking
<<<SHORT>>>
<|thinking|>lr too high<|/thinking|><|par|>Lower the <|code|>lr<|/code|>.<|/par|>
<<<MEDIUM>>>
<|thinking|>could be lr, could be data<|/thinking|><|par|>Lower it.<|/par|>
<<<LONG>>>
<|thinking|>work it through properly here<|/thinking|><|par|>Lower it.<|/par|>
<<<END>>>";
        let raw = split_triple(text).unwrap();
        assert_eq!(raw.user, "why is my loss spiking");
        assert_eq!(raw.answers.len(), 3);
        assert_eq!(raw.answers[0].0, Tier::Low);
        assert_eq!(raw.answers[2].0, Tier::High);
        assert!(raw.answers[0].1.contains("lr too high"), "{:?}", raw.answers[0].1);
    }

    #[test]
    fn tolerates_commentary_and_a_missing_end() {
        // Both of these are what a small local model actually does, and both are
        // recoverable — rejecting them would discard most of a run.
        let text = "Sure! Here is the block:\n\
<<<CONV>>>
<<<USER>>>
where is the file
<<<SHORT>>>
<|thinking|>it is in src<|/thinking|><|par|>In <|code|>src/</|code|>.<|/par|>";
        let raw = split_triple(text).unwrap();
        assert_eq!(raw.user, "where is the file");
        assert_eq!(raw.answers.len(), 1);
    }

    // The tier delimiters must be the ONLY `<<<…>>>` tokens in the prompt.
    //
    // Reusing the crate root's exemplar rendering put six to nine `<<<ASSISTANT>>>`
    // in front of the one place the output contract was stated, and the teacher
    // answered under `<<<ASSISTANT>>>` — producing a response with no tier in it at
    // all. Half the probe calls were lost this way.
    #[test]
    fn the_examples_never_wear_the_output_delimiters() {
        let conv = Conversation {
            system: None,
            turns: vec![
                Turn { role: "user".into(), text: "why is the sky blue".into() },
                Turn { role: "assistant".into(), text: "<|par|>Rayleigh.<|/par|>".into() },
            ],
        };
        let rendered = render_plain_exemplar(&conv, 1);
        assert!(!rendered.contains("<<<"), "{rendered}");
        assert!(rendered.contains("why is the sky blue"));
        assert!(rendered.contains("<|par|>Rayleigh.<|/par|>"), "the markup must survive");

        // And the assembled prompt: every `<<<…>>>` in it is part of the contract.
        let mut rng = Rng::new(1);
        let prompt = build_prompt(1, &[&conv], &[], &Coverage::new(), (62, 248, 414, 1.0 / 3.0), &mut rng);
        let contract = [
            crate::D_CONV,
            crate::D_USER,
            crate::D_END,
            delimiter(Tier::Low),
            delimiter(Tier::Med),
            delimiter(Tier::High),
        ];
        for found in prompt.match_indices("<<<") {
            let tail = &prompt[found.0..];
            assert!(
                contract.iter().any(|d| tail.starts_with(d)),
                "prompt contains a delimiter outside the output contract: {:?}",
                &tail[..tail.len().min(20)],
            );
        }
    }

    // The worked example is the only place the format appears filled in, so it has
    // to be valid under the rules it illustrates — an invalid one teaches the
    // teacher to produce invalid output.
    #[test]
    fn the_worked_example_is_itself_well_formed() {
        let (paired, void) = vocabulary();
        let raw = split_triple(&format!(
            "{}
{}
why did my boiled egg crack
{}",
            crate::D_CONV,
            crate::D_USER,
            WORKED_EXAMPLE
                .split_once("<<<SHORT>>>")
                .map(|(_, rest)| format!("<<<SHORT>>>{rest}"))
                .unwrap(),
        ))
        .expect("the worked example must parse as a triple");
        assert_eq!(raw.answers.len(), 3, "all three tiers must be shown");
        for (tier, reply) in &raw.answers {
            validate_reply(reply, 0, &paired, &void)
                .unwrap_or_else(|e| panic!("the {} example is invalid: {e}", tier.as_str()));
        }
        // The escalation has to be visible, or it illustrates nothing.
        let words: Vec<usize> = raw.answers.iter().map(|(_, r)| thinking_block(r).0).collect();
        assert!(words[0] < words[1] && words[1] < words[2], "{words:?}");
        // Inside the band, not merely above its floor. An example four times the
        // stated target teaches the teacher to write blocks the student's window
        // cannot hold — the exact failure the derived bounds exist to stop.
        let (low, med) = derived_bounds(1024);
        let (lo, hi) = tier_words(Tier::High, low, med, thinking_ceiling(1024), 1.0 / 3.0);
        assert!(
            words[2] >= lo && words[2] <= hi,
            "the long example is {} words, outside its own {lo}-{hi} target",
            words[2],
        );
    }

    #[test]
    fn a_block_with_no_tier_answers_is_refused() {
        let text = "<<<CONV>>>\n<<<USER>>>\nhello\n<<<END>>>";
        assert!(split_triple(text).is_err());
    }

    #[test]
    fn tiers_are_assigned_by_measurement() {
        assert_eq!(classify(0, 64, 256), Tier::Low);
        assert_eq!(classify(64, 64, 256), Tier::Low);
        assert_eq!(classify(65, 64, 256), Tier::Med);
        assert_eq!(classify(256, 64, 256), Tier::Med);
        assert_eq!(classify(257, 64, 256), Tier::High);
    }

    // The bug that made `think-high.jsonl` impossible: a top tier whose *floor*
    // costs more than the student's whole context window, so every candidate was
    // generated, validated, and then discarded by the window check.
    //
    // The property is that a minimal high-tier record has to fit: its thinking
    // block just over `med_max`, plus a reply of similar size, plus the user turn,
    // inside the window. The old 700-token boundary failed this at ctx 1024 by
    // more than 2x, and nothing tested it.
    #[test]
    fn a_high_tier_record_can_actually_fit_the_student() {
        for ctx in [512, 1024, 2048, 4096] {
            let (low, med) = derived_bounds(ctx);
            let ceiling = thinking_ceiling(ctx);
            assert!(low < med, "ctx {ctx}: bounds {low}/{med} do not order");
            assert!(med < ceiling, "ctx {ctx}: no room above med for a high tier");

            // A high-tier record: thinking just over `med`, a reply about as long
            // as the thinking, and a user turn. All of it inside the window.
            let smallest_high = med + 1;
            let plausible_record = smallest_high * 2 + 40;
            assert!(
                plausible_record <= ctx,
                "ctx {ctx}: the smallest high-tier record needs ~{plausible_record}                  tokens against a {ctx} window — the tier cannot be written",
            );
        }
        // And the concrete case that was broken.
        let (low, med) = derived_bounds(1024);
        assert!(med < 700, "ctx 1024 must not ask for the old 700-token boundary");
        assert!(low > 0 && med > low);
    }

    // Old stored configs carry 140/700 and the retry button replays them verbatim.
    // Honouring a bound whose top tier cannot fit would reproduce the original bug
    // on every retry, so it is refused in favour of the derived pair.
    #[test]
    fn an_impossible_explicit_bound_is_refused() {
        assert!(!high_tier_fits(700, 1024), "the config that caused all this");
        assert!(high_tier_fits(248, 1024));
        assert!(high_tier_fits(700, 2048));
    }

    #[test]
    fn the_bounds_follow_the_student_and_zero_means_derive() {
        // Wider student, wider tiers — the ambitious original numbers are reached
        // by growing the model rather than by asserting them.
        let (l1, m1) = derived_bounds(1024);
        let (l4, m4) = derived_bounds(4096);
        assert!(l4 > l1 && m4 > m1);
        assert!(m4 >= 700, "at ctx 4096 the old 700 boundary becomes reachable");

        // An explicit setting is honoured; 0/0 defers to the student.
        let mut r = request();
        r.low_max_tokens = 11;
        r.med_max_tokens = 22;
        assert_eq!(resolve_bounds(&r, None), (11, 22));
        r.low_max_tokens = 0;
        r.med_max_tokens = 0;
        assert_eq!(resolve_bounds(&r, None), (140, 700), "no student, no measurement");
    }

    // The word targets the prompt states must describe the same bands the
    // classifier files against, or the teacher is asked for lengths that are then
    // relabelled — which is how the tiers stopped meaning anything the first time.
    #[test]
    fn the_stated_word_targets_match_the_token_bands() {
        let (low, med) = derived_bounds(1024);
        let ceiling = thinking_ceiling(1024);
        let wpt = 1.0 / 3.0;
        let bands: Vec<(usize, usize)> = Tier::ALL
            .iter()
            .map(|t| tier_words(*t, low, med, ceiling, wpt))
            .collect();
        for (i, (lo, hi)) in bands.iter().enumerate() {
            assert!(lo < hi, "tier {i} band {lo}-{hi} is inverted");
        }
        assert!(bands[0].1 < bands[1].0, "low and med word targets overlap");
        assert!(bands[1].1 < bands[2].0, "med and high word targets overlap");
        // The high band, converted back to tokens, has to clear `med_max` — else
        // the teacher is asked for a length that files as medium.
        let high_floor_tokens = (bands[2].0 as f64 / wpt) as usize;
        assert!(
            high_floor_tokens > med,
            "asking for {} words ({high_floor_tokens} tokens) files as med, not high",
            bands[2].0,
        );
    }

    #[test]
    fn each_tier_writes_a_distinct_flat_file() {
        // Flat, and one per tier: a nested directory is invisible to every
        // consumer of raw/sft/, which all use a non-recursive read_dir.
        let names: Vec<&str> = Tier::ALL.iter().map(|t| tier_file_name(*t)).collect();
        assert_eq!(names.len(), 3);
        assert_eq!(names.iter().collect::<HashSet<_>>().len(), 3);
        for name in names {
            assert!(name.ends_with(".jsonl"), "{name}");
            assert!(!name.contains('/'), "{name}");
        }
    }

    #[test]
    fn a_rewritten_number_is_refused() {
        // The exact failure the first real run produced on every record.
        let user = "why did the training loss spike at step 1200 in run_007.jsonl?";
        assert!(check_quoted_facts(user, "spike at step 1200, clear enough").is_ok());
        let err = check_quoted_facts(user, "gradient norm blew up at step 1202").unwrap_err();
        assert!(err.contains("1200"), "{err}");
        assert!(check_quoted_facts(user, "see run_006.jsonl").is_err());
        assert!(check_quoted_facts(user, "see run_007.jsonl").is_ok());
    }

    #[test]
    fn a_rewritten_timestamp_is_refused() {
        let user = "error rate spiked at 14:23:17, find the cause";
        assert!(check_quoted_facts(user, "at 14:23:17 the pool was exhausted").is_ok());
        assert!(check_quoted_facts(user, "the restart at 14:22:17 did it").is_err());
    }

    #[test]
    fn inventing_unrelated_numbers_is_allowed() {
        // One-directional on purpose: a reply invents its own paths and figures, and
        // only a token that is *almost* one of the user's is treated as a rewrite. A
        // guard that flagged every new number would reject every useful answer.
        let user = "why did the loss spike at step 1200?";
        assert!(check_quoted_facts(user, "set warmup to 500 over 8 epochs").is_ok());
        assert!(check_quoted_facts(user, "check /var/log/train_9999.log").is_ok());
        assert!(check_quoted_facts(user, "try 64 or 128 instead").is_ok());
    }

    #[test]
    fn a_nearby_number_of_the_same_length_is_refused_too() {
        // `1500` against the user's `1200` is a single substitution — structurally
        // identical to the `1202` corruption, and no heuristic on the token alone can
        // tell "misquoted their number" from "reasoned to a different one".
        //
        // So it errs toward rejection, and that asymmetry is deliberate: the teacher
        // is local and free, so a discarded answer costs a few seconds of a GPU that
        // would otherwise be idle. An accepted corruption costs a corpus that quietly
        // teaches the model its own context is approximate — and nothing downstream
        // would ever surface it.
        let user = "why did the loss spike at step 1200?";
        assert!(check_quoted_facts(user, "warm up over 1500 steps").is_err());
    }

    #[test]
    fn short_numbers_are_not_treated_as_identifiers() {
        // "16" and "32" are quantities an answer reasons to, not identifiers the user
        // typed — matching on them would flag every reply mentioning a batch size.
        let user = "my batch size is 16, should I raise it?";
        assert!(check_quoted_facts(user, "go to 32, then 64").is_ok());
    }

    #[test]
    fn every_markup_instruction_gets_asked_for() {
        // Coverage is the whole point: the first run left list, item, quote, link and
        // italic out of the corpus entirely, and an untrained embedding row is a
        // marker the model cannot use.
        let mut coverage = Coverage::new();
        let mut seen: HashSet<&'static str> = HashSet::new();
        // Ask repeatedly, recording nothing — the least-used set must rotate rather
        // than return the same three forever.
        for _ in 0..MARKUP.len() {
            for entry in coverage.least_used(3) {
                seen.insert(entry);
            }
            // Simulate the batch landing, so those entries stop being least-used.
            for entry in coverage.least_used(3) {
                let i = MARKUP.iter().position(|m| *m == entry).unwrap();
                coverage.asked[i] += 1;
            }
        }
        assert_eq!(
            seen.len(),
            MARKUP.len(),
            "every markup instruction must come up; missed {:?}",
            MARKUP.iter().filter(|m| !seen.contains(*m)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn coverage_counts_only_tags_that_actually_appeared() {
        let mut coverage = Coverage::new();
        coverage.record("<|list|><|item|>a<|/item|><|/list|>");
        let list_at = MARKUP.iter().position(|m| m.contains("<|list|>")).unwrap();
        let quote_at = MARKUP.iter().position(|m| m.contains("<|quote|>")).unwrap();
        assert_eq!(coverage.asked[list_at], 1);
        assert_eq!(coverage.asked[quote_at], 0);
    }

    #[test]
    fn the_domains_are_broad_rather_than_all_software() {
        // The first run came back entirely about training logs because every topic in
        // the shared list was about this project. Guard the property, not the list.
        assert!(DOMAINS.len() >= 10, "{} domains", DOMAINS.len());
        let technical = DOMAINS
            .iter()
            .filter(|(name, _)| name.contains("software") || name.contains("machine learning"))
            .count();
        assert!(
            technical * 3 <= DOMAINS.len(),
            "software and ML are {technical} of {} domains — too concentrated",
            DOMAINS.len()
        );
        for (name, topics) in DOMAINS {
            assert!(!topics.is_empty(), "{name} has no topics");
        }
    }

    #[test]
    fn a_reply_with_no_thinking_block_measures_zero() {
        assert_eq!(think_tokens("<|par|>just an answer<|/par|>", None), Some(0));
    }

    #[test]
    fn truncate_does_not_split_a_codepoint() {
        let s = "→→→→→→→→→→";
        let out = truncate(s, 3);
        assert_eq!(out, "→→→…");
    }
}
