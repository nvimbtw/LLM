// Synthetic SFT conversations from a cheap hosted model.
//
// This is the Rust half of `scripts/generate_sft_llm.py`, and the reason to move it
// here is not the UI. It is that the script validates against copies:
//
//   * a hand-copied list of the markup tags, which drifts from
//     `tokenize::pretokenize::SPECIAL_TOKENS` the moment a marker is added or pruned
//     (E008–E00B already came and went), and
//   * a character-count stand-in for length, which cannot answer the only length
//     question that matters — does the rendered conversation fit the context window
//     of the model that will train on it.
//
// Here both are answered by the real thing: the tag vocabulary is *derived* from
// `SPECIAL_TOKENS`, and the budget check renders the conversation through
// `llm::chat_template::build_training_example` with the base model's own merges and
// counts the tokens SFT will actually chunk. A conversation that would not fit is
// rejected at generation rather than discovered later as a window full of context
// with no assistant token in it.
//
// **This never touches the GPU**, which is why its job runs on the network lane
// (see `db::job::JobLane`). It is a sequence of HTTP calls and some parsing.

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;


// ── the request ─────────────────────────────────────────────────────────────

// What a generation run was asked for. Stored as the job's `config`, so it is also
// what the retry button replays.
//
// The API key is deliberately *not* here. This struct is persisted in the jobs
// table and echoed back to the browser with every board refetch; a key in it would
// be a key in the database and in the network tab. Only the *name* of the
// environment variable travels, and the value is read on the server at call time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthRequest {
    // An OpenAI-compatible API base, e.g. https://openrouter.ai/api/v1 or
    // http://localhost:11434/v1 for a local Ollama.
    pub base_url: String,
    pub model: String,
    // How many valid conversations to add before stopping.
    pub count: usize,
    // Conversations requested per model call. Small on purpose: a batch of ten with
    // long thinking blocks runs past the response cap and arrives truncated, and a
    // truncated batch loses every conversation after the cut.
    #[serde(default = "default_batch")]
    pub batch: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    // Thinking blocks shorter than this are rejected. The block is the whole point
    // of generating rather than hand-writing — the gold set has almost none.
    #[serde(default = "default_min_think_words")]
    pub min_think_words: usize,
    #[serde(default = "default_think_frac")]
    pub think_frac: f64,
    #[serde(default = "default_system_frac")]
    pub system_frac: f64,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    // Safety cap on calls. 0 means "three times the ideal number", which is the
    // budget that stops a model returning nothing usable from looping forever.
    #[serde(default)]
    pub max_calls: usize,
    // The model whose tokenizer and context window define the length budget. When
    // absent, conversations are accepted without a token check — which is the old
    // Python behaviour and the reason the corpus ended up mostly unfittable.
    #[serde(default)]
    pub id_model: Option<Uuid>,
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    // Gold conversations to imitate, and where the output goes. Both are file names
    // inside the SFT dir, never paths — a run cannot be pointed at the rest of the
    // disk from the browser.
    #[serde(default = "default_exemplar_file")]
    pub exemplar_file: String,
    #[serde(default = "default_out_file")]
    pub out_file: String,
    // Seeds the topic/shape/exemplar sampling, so a run can be repeated exactly.
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_batch() -> usize {
    2
}
fn default_temperature() -> f32 {
    1.0
}
fn default_max_tokens() -> usize {
    8000
}
fn default_min_think_words() -> usize {
    120
}
fn default_think_frac() -> f64 {
    0.75
}
fn default_system_frac() -> f64 {
    0.15
}
fn default_timeout() -> u64 {
    600
}
fn default_api_key_env() -> String {
    "SFT_API_KEY".to_string()
}
fn default_exemplar_file() -> String {
    "synthetic.jsonl".to_string()
}
fn default_out_file() -> String {
    "synthetic_llm.jsonl".to_string()
}

// ── what a run produces ─────────────────────────────────────────────────────

// One accepted conversation, as the live feed shows it. The text is carried
// because "what did it actually write" is the question the counts cannot answer —
// a run can add two hundred conversations and every one of them be the same shape.
#[derive(Debug, Clone, Serialize)]
pub struct Accepted {
    pub opener: String,
    pub reply: String,
    pub turns: usize,
    pub think_words: usize,
    pub tokens: Option<usize>,
    pub user_chars: usize,
    pub assistant_chars: usize,
}

// One model call, start to finish. The per-call record is what makes a run
// diagnosable: a call that returns nothing, a call whose output is all rejected for
// one reason, and a call that is fine but duplicates are three different problems
// with three different fixes.
#[derive(Debug, Clone, Serialize)]
pub struct CallReport {
    pub call: usize,
    pub kept: usize,
    pub duplicates: usize,
    // Reason → how many, for everything this call's output lost.
    pub rejected: Vec<(String, usize)>,
    pub elapsed_ms: u128,
    pub response_chars: usize,
    // Running totals after this call, so a reattaching page can draw the curve
    // without replaying every frame.
    pub total: usize,
    pub target: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SynthStats {
    pub added: usize,
    pub calls: usize,
    pub duplicates: usize,
    pub with_thinking: usize,
    pub rejected: Vec<(String, usize)>,
    pub out_file: String,
    // What the file holds now, gold and generated together — the number the
    // question "how many do we have" actually asks.
    pub corpus_total: usize,
}

pub type ProgressCallback = Box<dyn Fn(&str, f64) + Send>;
pub type CallCallback = Box<dyn Fn(&CallReport) + Send>;
pub type AcceptedCallback = Box<dyn Fn(&Accepted) + Send>;

// ── the conversation shape ──────────────────────────────────────────────────

// The SFT line format, as `training::build_sft_dataset` reads it back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub turns: Vec<Turn>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub role: String,
    pub text: String,
}

impl Conversation {
    fn assistant_chars(&self) -> usize {
        self.turns
            .iter()
            .filter(|t| t.role == "assistant")
            .map(|t| t.text.len())
            .sum()
    }

    fn user_chars(&self) -> usize {
        self.turns
            .iter()
            .filter(|t| t.role == "user")
            .map(|t| t.text.len())
            .sum()
    }

    // Normalized first user message. Near-duplicate conversations almost always
    // share their opening question verbatim, so this is the dedup key.
    fn opener_key(&self) -> String {
        self.turns
            .first()
            .map(|t| t.text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase())
            .unwrap_or_default()
    }
}

// ── the markup vocabulary, derived rather than copied ───────────────────────

// Markers that are scaffold rather than reply markup. `user`/`assistant`/`system`
// are supplied by the chat template itself, `endoftext` separates examples, and
// `thinking` is handled separately (it is allowed, but only as a leading block).
// `title` renders as a document title, which a chat reply has no business emitting.
//
// `reply` is excluded for a different reason: its content is a message id, not
// prose. A generated conversation has no real ids to point at, so training on it
// would teach the model to emit plausible-looking but meaningless uuids — and the
// id text itself is high-entropy noise no model at this scale can predict. The
// token stays reserved so the frontend renders it; only generated data avoids it.
const NOT_REPLY_MARKUP: &[&str] = &[
    "user",
    "assistant",
    "system",
    "endoftext",
    "thinking",
    "title",
    "reply",
];

// Every tag name a reply may use, split by whether it closes.
//
// Derived from `SPECIAL_TOKENS` at first use rather than listed here: a name is
// paired when the vocabulary holds both `<|name|>` and `<|/name|>`, and void when
// it holds only the opening form. Add a marker to the tokenizer and this follows
// automatically — which is the drift the Python copy could not avoid.
fn vocabulary() -> (HashSet<&'static str>, HashSet<&'static str>) {
    let names: Vec<&str> = tokenize::pretokenize::SPECIAL_TOKENS
        .iter()
        .map(|(token, _)| token.trim_start_matches("<|").trim_end_matches("|>"))
        .collect();

    let mut paired = HashSet::new();
    let mut void = HashSet::new();
    for name in &names {
        if name.starts_with('/') {
            continue;
        }
        if NOT_REPLY_MARKUP.contains(name) {
            continue;
        }
        let closer = format!("/{name}");
        if names.iter().any(|n| *n == closer) {
            paired.insert(*name);
        } else {
            void.insert(*name);
        }
    }
    (paired, void)
}

// ── validation ──────────────────────────────────────────────────────────────

// Assistant replies in the gold set top out around 4.7k chars. Beyond this a small
// model is padding, and the token budget below will reject it anyway when a base
// model is named — this is the cheap check that runs first.
const MAX_ASSISTANT_CHARS: usize = 6000;
const MAX_USER_CHARS: usize = 30000;

// The closer this vocabulary uses is `<|/par|>`. Models reliably reach for the
// XML-shaped `</|par|>` instead — it is the form every other markup language uses,
// and no amount of showing the right one in the prompt fully suppresses it. The
// rewrite is unambiguous and lossless, so repairing beats discarding an otherwise
// good conversation over a transposed slash.
fn repair_markup(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if text[i..].starts_with("</|") {
            // Only a well-formed `</|word|>` is rewritten; anything else is left
            // alone so a genuine oddity still fails validation rather than being
            // quietly reshaped into something that passes.
            if let Some(end) = text[i + 3..].find("|>") {
                let name = &text[i + 3..i + 3 + end];
                if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric()) {
                    out.push_str("<|/");
                    out.push_str(name);
                    out.push_str("|>");
                    i += 3 + end + 2;
                    continue;
                }
            }
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

const THINK_OPEN: &str = "<|thinking|>";
const THINK_CLOSE: &str = "<|/thinking|>";

// Word count of a leading thinking block, and where the body after it starts.
fn thinking_block(text: &str) -> (usize, usize) {
    let trimmed_start = text.len() - text.trim_start().len();
    if !text[trimmed_start..].starts_with(THINK_OPEN) {
        return (0, 0);
    }
    let open_end = trimmed_start + THINK_OPEN.len();
    match text[open_end..].find(THINK_CLOSE) {
        Some(offset) => {
            let inner = &text[open_end..open_end + offset];
            (
                inner.split_whitespace().count(),
                open_end + offset + THINK_CLOSE.len(),
            )
        }
        None => (0, 0),
    }
}

// Why a conversation was thrown away, in the words the tally shows. `Ok(())` is a
// conversation worth writing.
fn validate(conv: &Conversation, min_think_words: usize) -> Result<(), String> {
    if conv.turns.is_empty() {
        return Err("no turns".to_string());
    }
    if let Some(system) = &conv.system {
        if system.contains("<|") {
            return Err("markup in the system prompt".to_string());
        }
    }
    let (paired, void) = vocabulary();

    for (i, turn) in conv.turns.iter().enumerate() {
        let expected = if i % 2 == 0 { "user" } else { "assistant" };
        if turn.role != expected {
            return Err("roles out of order".to_string());
        }
        if turn.text.trim().is_empty() {
            return Err("empty turn".to_string());
        }
        if turn.role == "user" {
            if turn.text.contains("<|") || turn.text.contains("|>") {
                return Err("markup in a user turn".to_string());
            }
            if turn.text.len() > MAX_USER_CHARS {
                return Err("user turn too long".to_string());
            }
            continue;
        }
        if turn.text.len() > MAX_ASSISTANT_CHARS {
            return Err("assistant turn too long".to_string());
        }
        validate_reply(&turn.text, min_think_words, &paired, &void)?;
    }

    if conv.turns.last().map(|t| t.role.as_str()) != Some("assistant") {
        return Err("does not end on assistant".to_string());
    }
    Ok(())
}

// Well-formed reply markup: an optional leading thinking block, then balanced tags
// drawn from the derived vocabulary.
//
// Bare prose between tags is allowed — the parser renders root-level text and the
// gold replies rely on it — but a reply with no markup at all is rejected, because
// SFT is the only place the model learns the vocabulary exists.
fn validate_reply(
    text: &str,
    min_think_words: usize,
    paired: &HashSet<&'static str>,
    void: &HashSet<&'static str>,
) -> Result<(), String> {
    let (think_words, body_start) = thinking_block(text);
    let body = &text[body_start..];

    if body.contains(THINK_OPEN) || body.contains("/thinking") {
        return Err("thinking block out of place".to_string());
    }
    if body_start > 0 {
        if think_words == 0 {
            return Err("empty thinking block".to_string());
        }
        if think_words < min_think_words {
            return Err(format!("thinking block under {min_think_words} words"));
        }
        if body.trim().is_empty() {
            return Err("reasoned, then said nothing".to_string());
        }
    }

    let tags = tags_in(body)?;
    if tags.is_empty() {
        return Err("no markup at all".to_string());
    }

    let mut stack: Vec<&str> = Vec::new();
    for tag in &tags {
        // Bare tags only. An attribute form like `codeblock|lang='python'` no longer
        // tokenizes to a single id — it byte-shreds into the raw markup — so it is
        // rejected rather than repaired.
        if tag.contains('|') || tag.contains('=') {
            return Err("tag with attributes".to_string());
        }
        match tag.strip_prefix('/') {
            Some(name) => match stack.pop() {
                Some(open) if open == name => {}
                _ => return Err("unbalanced markup".to_string()),
            },
            None if void.contains(tag.as_str()) => {}
            None if paired.contains(tag.as_str()) => {
                // Leaked through `paired` as a &'static str so the stack can hold a
                // borrow rather than a copy of every tag name.
                stack.push(paired.get(tag.as_str()).unwrap());
            }
            None => return Err("unknown tag".to_string()),
        }
    }
    if !stack.is_empty() {
        return Err("unbalanced markup".to_string());
    }
    Ok(())
}

// The inner text of every `<|…|>` in `text`. Errs when a `<|` or `|>` appears that
// is not part of a well-formed tag, which is what catches a dangling delimiter.
fn tags_in(text: &str) -> Result<Vec<String>, String> {
    let mut tags = Vec::new();
    let mut rest = text;
    let mut opens = 0usize;
    while let Some(start) = rest.find("<|") {
        opens += 1;
        let after = &rest[start + 2..];
        match after.find("|>") {
            Some(end) => {
                tags.push(after[..end].to_string());
                rest = &after[end + 2..];
            }
            None => return Err("dangling markup".to_string()),
        }
    }
    // Every `|>` must have belonged to one of the tags just read; a leftover means
    // a closer with no opener.
    if text.matches("|>").count() != opens {
        return Err("dangling markup".to_string());
    }
    Ok(tags)
}

// ── the token budget ────────────────────────────────────────────────────────

// The tokenizer and context window a conversation has to fit, loaded from the base
// model that will train on it. This is the check the Python could not do.
pub struct Budget {
    merges: Vec<((u32, u32), u32)>,
    pub ctx_window: usize,
}

impl Budget {
    // `merges.bin` sits next to the weights; the context window is the `ctx_window`
    // the run was configured with, read from the model's own meta.json rather than
    // by loading the weights — this needs one number, not 100M parameters.
    pub fn load(model_dir: &Path) -> Result<Budget, String> {
        let merges = tokenize::bpe::load_merges(&model_dir.join("merges.bin").to_string_lossy())
            .map_err(|e| format!("could not read the model's tokenizer: {e}"))?;

        let meta = fs::read_to_string(model_dir.join("meta.json"))
            .map_err(|e| format!("could not read the model's meta.json: {e}"))?;
        let parsed: serde_json::Value = serde_json::from_str(&meta)
            .map_err(|e| format!("the model's meta.json is unreadable: {e}"))?;
        let ctx_window = parsed
            .get("config")
            .and_then(|c| c.get("ctx_window"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "the model's meta.json names no context window".to_string())?;

        Ok(Budget {
            merges,
            ctx_window: ctx_window as usize,
        })
    }

    /// How many **words** of conversation the window holds, measured on real text.
    ///
    /// The prompt has to state the limit in words, because that is the only unit a
    /// model can hold itself to: it cannot count its own tokens, and asking it for
    /// a character count gets a guess. So the conversion has to be measured rather
    /// than assumed — and assuming it is exactly what went wrong here. A hardcoded
    /// 3.5 characters per token (a reasonable figure for a large general tokenizer)
    /// is more than double the truth for ours, which is trained on a small corpus
    /// and so has few merges and short tokens. The prompt was advertising roughly
    /// twice the room that existed, and the overflow it caused was read as the
    /// model ignoring instructions.
    ///
    /// Measured against the exemplars specifically, since they are drawn from the
    /// same corpus as the output and carry the same markup density — the tags cost
    /// one token each regardless of how many characters they spell.
    pub fn word_capacity(&self, exemplars: &[&Conversation]) -> usize {
        let mut words = 0usize;
        let mut tokens = 0usize;
        for conv in exemplars {
            words += conv.turns.iter().map(|t| t.text.split_whitespace().count()).sum::<usize>();
            tokens += self.tokens(conv);
        }
        if words == 0 || tokens == 0 {
            return 0;
        }

        // Held back for the scaffold and for the fact that a rendered conversation
        // is checked against the window with no slack: landing exactly on the limit
        // is a discard, so the target has to sit under it.
        let usable = (self.ctx_window as f64 * 0.9) as usize;
        usable * words / tokens
    }

    // How many tokens this conversation occupies once rendered for training —
    // exactly what `build_sft_dataset` will produce for it, because it is the same
    // call with the same merges.
    pub fn tokens(&self, conv: &Conversation) -> usize {
        let turns: Vec<llm::chat_template::Turn> = conv
            .turns
            .iter()
            .map(|t| llm::chat_template::Turn {
                role: if t.role == "user" {
                    llm::chat_template::Role::User
                } else {
                    llm::chat_template::Role::Assistant
                },
                text: t.text.clone(),
            })
            .collect();
        let (ids, _) = llm::chat_template::build_training_example(
            conv.system.as_deref(),
            &turns,
            &self.merges,
        );
        ids.len()
    }
}

// ── parsing the model's output ──────────────────────────────────────────────

const D_CONV: &str = "<<<CONV>>>";
const D_SYSTEM: &str = "<<<SYSTEM>>>";
const D_USER: &str = "<<<USER>>>";
const D_ASSISTANT: &str = "<<<ASSISTANT>>>";
const D_END: &str = "<<<END>>>";

// Conversations read out of delimiter-formatted output.
//
// The model is asked for delimiters rather than JSON because a conversation has to
// survive newlines, quotes and backslashes inside every turn, and a small model
// gets that escaping wrong often enough that most of a batch is lost — on output
// whose *content* was fine. Delimiters remove the problem instead of policing it.
//
// Tolerant by design: a banner, a stray sentence or a markdown fence between
// conversations is ignored, because only a line that is exactly a delimiter carries
// structure. A trailing conversation with no `<<<END>>>` (the response ran out of
// tokens) is still returned, and is dropped later only if it fails validation —
// which a half-written reply usually will.
/// Drops a reasoning preamble, so only the model's actual answer is parsed.
///
/// Reasoning models emit their scratch work first — Gemma 4 in `<thought>`, the
/// R1-derived models in `<think>`. That is not decoration the parser can ignore:
/// the scratch work *rehearses the output format*, so a single call arrives holding
/// several draft `<<<CONV>>>` markers before the real ones. `split_delimited` scans
/// for markers anywhere, so without this it parses the rehearsal — yielding
/// conversations built from bullet points and half-written sentences that look
/// structurally valid enough to reach the file.
///
/// Only a preamble is removed. A tag appearing later is left alone, because by then
/// it is far more likely to be a conversation *about* markup than a real block, and
/// silently cutting the middle out of a reply would be worse than keeping it.
///
/// An unterminated block means the response was truncated mid-thought and never
/// reached an answer, so nothing is returned.
fn strip_reasoning(raw: &str) -> &str {
    const BLOCKS: [(&str, &str); 2] = [("<thought>", "</thought>"), ("<think>", "</think>")];

    let trimmed = raw.trim_start();
    for (open, close) in BLOCKS {
        if let Some(rest) = trimmed.strip_prefix(open) {
            return match rest.find(close) {
                Some(end) => rest[end + close.len()..].trim_start(),
                None => "",
            };
        }
    }
    raw
}

fn split_delimited(text: &str) -> Vec<Conversation> {
    let mut out: Vec<Conversation> = Vec::new();
    let mut current: Option<Conversation> = None;
    let mut role: Option<&str> = None;
    let mut buf: Vec<&str> = Vec::new();

    // Closes the run of lines collected for the current delimiter.
    fn flush(current: &mut Option<Conversation>, role: Option<&str>, buf: &mut Vec<&str>) {
        let body = buf.join("\n").trim().to_string();
        buf.clear();
        let (Some(conv), Some(role)) = (current.as_mut(), role) else {
            return;
        };
        if body.is_empty() {
            return;
        }
        if role == "system" {
            conv.system = Some(body);
        } else {
            conv.turns.push(Turn {
                role: role.to_string(),
                text: body,
            });
        }
    }

    for line in text.lines() {
        let marker = line.trim();
        let is_delimiter = matches!(marker, D_CONV | D_SYSTEM | D_USER | D_ASSISTANT | D_END);
        if !is_delimiter {
            if role.is_some() {
                buf.push(line);
            }
            continue;
        }

        flush(&mut current, role, &mut buf);
        match marker {
            D_CONV => {
                if let Some(conv) = current.take() {
                    if !conv.turns.is_empty() {
                        out.push(conv);
                    }
                }
                current = Some(Conversation {
                    system: None,
                    turns: Vec::new(),
                });
                role = None;
            }
            D_END => {
                if let Some(conv) = current.take() {
                    if !conv.turns.is_empty() {
                        out.push(conv);
                    }
                }
                role = None;
            }
            other => {
                // A role delimiter with no <<<CONV>>> before it: the model started
                // straight in. Treated as an implicit start rather than discarding
                // everything that follows.
                if current.is_none() {
                    current = Some(Conversation {
                        system: None,
                        turns: Vec::new(),
                    });
                }
                role = Some(match other {
                    D_SYSTEM => "system",
                    D_USER => "user",
                    _ => "assistant",
                });
            }
        }
    }

    flush(&mut current, role, &mut buf);
    if let Some(conv) = current {
        if !conv.turns.is_empty() {
            out.push(conv);
        }
    }
    out
}

// ── the prompt ──────────────────────────────────────────────────────────────

// Drawn from the gold set's actual domain: an agentic coding CLI driving model
// training runs, codebase work, and tool use.
const TOPICS: &[&str] = &[
    "debugging a stalled or crashing model training run",
    "reading an unfamiliar codebase and explaining its layout",
    "a CUDA/GPU out-of-memory error and how to cut memory use",
    "tokenizer, vocabulary, or BPE merge behaviour",
    "benchmark or leaderboard numbers that moved unexpectedly",
    "a dataset pipeline producing wrong or truncated records",
    "an agent CLI misusing its tools or looping on a tool call",
    "a TUI/terminal rendering bug (scroll, layout, redraw)",
    "checkpointing, resuming, or merging model weights",
    "learning-rate, batch-size, or schedule tuning",
    "an audit for bugs in a file the user names, read-only",
    "a git problem: bad merge, lost commit, messy history",
    "a shell one-liner or script that does the wrong thing",
    "profiling something slow and finding the real bottleneck",
    "a flaky or wrong test and what it actually proves",
    "refactoring a module without changing behaviour",
    "an API/library call returning an error the user pasted",
    "explaining a model-architecture detail (attention, RoPE, MoE, LoRA)",
    "evaluating whether a paper's technique is worth trying here",
    "a data-format question: JSONL schema, encoding, escaping",
    "concurrency: a race, a deadlock, or a worker pool stalling",
    "deciding between two implementation approaches under a constraint",
    "a regex or parser that misses an edge case",
    "reading logs or a stack trace the user pasted and locating the cause",
    "distillation, SFT, or RL fine-tuning strategy questions",
    "disk/IO, caching, or a run directory filling up",
];

const SHAPES: &[&str] = &[
    "a long pasted task brief (file paths, constraints, logs) answered with a short plan and a first concrete step",
    "a terse one-line question ('where is the file?', 'why a G?') answered in two or three dense sentences, no preamble",
    "a question, then a terse follow-up ('but why?', 'and if it fails?') that the assistant answers without re-explaining what it already said",
    "the user pastes an error or log; the assistant names the cause and the fix",
    "the user's premise is wrong; the assistant says so plainly and corrects it",
    "a multi-turn investigation: assistant states what it will check, user says 'continue', assistant reports what it found",
    "a request to compare two approaches, ending on a recommendation",
    "a request for a short code change, answered with a codeblock and one note",
    "a vague request; the assistant asks one sharp clarifying question, the user answers, the assistant then delivers",
];

const FEATURES: &[&str] = &[
    "a bulleted <|list|> of <|item|>s",
    "a <|codeblock|> code block (no language attribute — just the code inside)",
    "<|code|> inline code for a path, command, flag, or symbol",
    "<|bold|> on the one or two key terms",
    "<|italic|> for light emphasis",
    "an answer in two parts under <|heading|> or <|subheading|> headings",
    "a <|divider|> between two distinct sections",
    "a <|quote|> block quoting the user's own words or an error string back",
    "a <|warn|> callout for the one caveat that would bite the user if missed",
    "plain prose with only inline <|code|> and <|bold|>, no block structure",
];

// The gold set has no system prompts; these mirror the implicit persona in it.
const SYSTEM_PROMPTS: &[&str] = &[
    "You are a coding copilot embedded in a terminal CLI. Be terse and technical.",
    "You are an ML engineering assistant. Answer with specifics, no filler.",
    "You are a senior engineer pair-programming over a shared repo. Be direct.",
];

// What the previous call taught this one. Empty on the first call of a run.
//
// The single biggest lever on yield here: a run makes a dozen-odd independent calls
// with the same prompt, so without this, a model that misspells its closing tags
// misspells them in every call and the run burns its whole budget on the same
// mistake. Feeding the last call's *actual* discard reasons back turns a run into
// something that converges instead of repeating.
#[derive(Default)]
struct Feedback {
    // Discard reasons from the last call, most frequent first, already phrased the
    // way `validate` phrases them.
    reasons: Vec<(String, usize)>,
    // How many of the last call's conversations were thrown away as repeats of an
    // opener already in the corpus.
    duplicates: usize,
    // Openers already in the corpus, so the model can be told what not to write
    // again. A sample, not the lot: the corpus runs to thousands and the prompt has
    // a budget.
    used_openers: Vec<String>,
}

// How many previous openers to show. Enough to convey the range already covered
// without spending more of the window on them than on the exemplars.
const AVOID_OPENERS: usize = 12;
// Openers are truncated to this many characters — the opening clause is what
// carries the repetition, and a pasted 400-line brief in the avoid-list would cost
// more than the exemplars do.
const OPENER_PREVIEW_CHARS: usize = 90;

// The corrections section, or an empty string on the first call.
fn render_feedback(feedback: &Feedback) -> String {
    let mut out = String::new();

    if !feedback.reasons.is_empty() {
        out.push_str(
            "\nYOUR LAST BATCH — the previous response to this exact prompt was parsed\n\
             and validated, and these conversations were DISCARDED for these reasons:\n",
        );
        for (reason, n) in feedback.reasons.iter().take(5) {
            out.push_str(&format!("  - {n}× {reason}\n"));
        }
        out.push_str(
            "Fix these specifically in this batch. They are the actual output of the\n\
             validator, not hypotheticals — whatever you did last time, do it differently.\n",
        );
    }

    if feedback.duplicates > 0 {
        out.push_str(&format!(
            "\n{} conversation(s) from your last batch were discarded as REPEATS: their\n\
             opening user message was one already in the corpus. Openers are deduplicated\n\
             verbatim, so a new phrasing of the same question still counts as new — but the\n\
             same question asked the same way is wasted output.\n",
            feedback.duplicates
        ));
    }

    if !feedback.used_openers.is_empty() {
        out.push_str(
            "\nAlready in the corpus — do NOT open a conversation with any of these, or a\n\
             close variant of one:\n",
        );
        for opener in &feedback.used_openers {
            out.push_str(&format!("  - {opener}\n"));
        }
    }

    out
}

// A one-line preview of an opener for the avoid-list: collapsed whitespace, cut to
// its first clause, so a pasted brief contributes a line rather than a page.
fn opener_preview(opener: &str) -> String {
    let flat: String = opener.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(OPENER_PREVIEW_CHARS) {
        Some((cut, _)) => format!("{}…", &flat[..cut]),
        None => flat,
    }
}

// A gold conversation in the delimiter format the model must reproduce.
//
// Shown in the *output* format rather than as JSON: demonstrating one format while
// asking for another is most of why the JSON version of this failed.
fn render_exemplar(conv: &Conversation) -> String {
    let mut parts = vec![D_CONV.to_string()];
    if let Some(system) = conv.system.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        parts.push(D_SYSTEM.to_string());
        parts.push(system.to_string());
    }
    for turn in &conv.turns {
        parts.push(if turn.role == "user" { D_USER } else { D_ASSISTANT }.to_string());
        parts.push(turn.text.clone());
    }
    parts.push(D_END.to_string());
    parts.join("\n")
}

// `capacity` is the word budget from `Budget::word_capacity`, or 0 when no model was
// named and there is therefore no limit to state. Passed in rather than derived here
// so this stays a pure string-builder: measuring needs a tokenizer, phrasing does not.
fn build_prompt(
    request: &SynthRequest,
    exemplars: &[&Conversation],
    capacity: usize,
    feedback: &Feedback,
    rng: &mut Rng,
) -> String {
    let topics = pick(TOPICS, 5, rng).join("; ");
    let shapes = pick(SHAPES, 3, rng).join("; ");
    let features = pick(FEATURES, 4, rng).join("; ");
    // The thinking block has to be derived from that capacity, not requested
    // independently of it. Asking for "at least 120 words, typically 180-340" inside
    // a window holding ~149 words total is not a demanding instruction, it is an
    // impossible one — and an impossible instruction does not produce a shorter
    // answer, it produces a long one that gets discarded. Below the floor the block
    // is dropped entirely rather than shrunk into a token-burning stub, because a
    // 30-word scratchpad teaches the model to open a thinking block and say nothing.
    const THINK_FLOOR: usize = 60;
    let (min_words, think_pct) = match capacity {
        0 => (
            request.min_think_words,
            (request.think_frac * 100.0).round().clamp(0.0, 100.0) as u32,
        ),
        // Half the budget to reasoning, half to the reply that has to stand alone.
        cap => {
            let allowed = request.min_think_words.min(cap / 2);
            if allowed < THINK_FLOOR {
                (0, 0)
            } else {
                (allowed, (request.think_frac * 100.0).round().clamp(0.0, 100.0) as u32)
            }
        }
    };
    let typical = format!("{}-{}", min_words, min_words + min_words / 2);

    let system_note = if rng.next_f64() < request.system_frac {
        format!(
            "\nOn ONE conversation only, put a {D_SYSTEM} block immediately after {D_CONV}, \
             before the first {D_USER}, e.g. \"{}\".\n",
            SYSTEM_PROMPTS[rng.below(SYSTEM_PROMPTS.len())]
        )
    } else {
        String::new()
    };

    // The length rule is stated in tokens *and* in characters, because the model
    // cannot count the former. Present only when a base model was named — without
    // one there is no budget to state, and inventing a number would be worse than
    // saying nothing.
    let length_rule = if capacity == 0 {
        String::new()
    } else {
        format!(
            "\n- LENGTH — READ THIS TWICE. The ENTIRE conversation, every turn added \
             together, must come to AT MOST {capacity} WORDS. Not per turn: per \
             conversation. That is short — roughly {sentences} sentences in total, so a \
             single exchange of a brief question and a focused answer. Anything longer \
             is discarded outright, and this is by far the most common reason output is \
             thrown away. Count as you write. Prefer ONE exchange; if a topic cannot be \
             answered well within {capacity} words, pick a narrower topic rather than \
             writing a longer conversation.",
            capacity = capacity,
            // ~15 words a sentence, the figure the exemplars sit near.
            sentences = (capacity / 15).max(1),
        )
    };

    // Omitted entirely when the window cannot hold a real thinking block. Leaving it
    // in with a smaller number would still spend most of the budget on reasoning and
    // still overflow; the honest thing at this size is conversations without one.
    let thinking_rule = if think_pct == 0 {
        String::new()
    } else {
        format!(
"         THINKING BLOCKS — the most important part of this task.\n\
         For about {think_pct}% of conversations, the FINAL assistant reply must START\n\
         with a hidden reasoning block, before any other tag:\n\
         \n\
           <|thinking|>...<|/thinking|>\n\
         \n\
         That block must be LONG and SUBSTANTIVE: at least {min_words} words, typically\n\
         {typical}. It is the model's private scratchpad, so write it as real working\n\
         thought, not a summary of the answer. Work through, in order:\n\
           1. what the user is actually asking for, and what they are not,\n\
           2. what the concrete details (paths, numbers, error strings, versions) imply,\n\
           3. two or three candidate explanations or approaches, weighed against each\n\
              other, with at least one explicitly rejected and the reason why,\n\
           4. edge cases, failure modes, or what could make the chosen answer wrong,\n\
           5. a decision about what to actually say and what to leave out.\n\
         Reason about THIS specific question — cite the user's own numbers, paths and\n\
         symbols. Change your mind mid-block where it is natural to. Never write generic\n\
         filler (\"Let me think about this carefully\", \"The user wants help with X\").\n\
         No <|...|> tags inside the thinking block; plain prose only.\n\
         The reply AFTER the block must stand on its own and stay tight — the length\n\
         goes in the thinking, not the answer.\n\
         \n"
        )
    };

    let exemplar_text: Vec<String> = exemplars.iter().map(|c| render_exemplar(c)).collect();

    format!(
        "You are generating supervised training data for a small chat model that acts as\n\
         a CODING AGENT in a terminal CLI: it reads codebases, drives model-training runs,\n\
         uses tools, and answers like a senior engineer — dense, specific, no filler.\n\
         \n\
         Below are {n_ex} REAL conversations from the target dataset. Study their voice,\n\
         length, density and markup style closely. Your output must look like it came\n\
         from the same corpus.\n\
         \n\
         --- REAL EXAMPLES ---\n\
         {exemplars}\n\
         --- END EXAMPLES ---\n\
         \n\
         Output exactly {batch} conversations in the SAME delimiter format as the examples\n\
         above. No prose, no markdown fences, no commentary before or after.\n\
         \n\
         Every delimiter sits ALONE on its own line, with nothing else on that line:\n\
         \n\
         {conv}\n\
         {user}\n\
         the user's message, as many lines as it needs\n\
         {assistant}\n\
         the reply, as many lines as it needs\n\
         {end}\n\
         {system_note}\
         Write turn text literally — real newlines, real quotes, no escaping of any kind.\n\
         Nothing is JSON here.\n\
         \n\
         Hard rules (a conversation breaking any of these is discarded):\n\
         - roles alternate user/assistant, start with \"user\", END on \"assistant\",\n\
         - 1 to 4 exchanges per conversation,\n\
         - user turns are PLAIN text and realistic: sometimes a pasted brief with file\n\
           paths, logs, or requirements; sometimes three terse words with a typo.\n\
           Never any <|...|> tokens in a user turn,\n\
         - assistant turns are under {max_chars} characters, written in the voice of the\n\
           examples: lead with the answer, no \"Great question!\", no restating the ask,\n\
         - assistant markup uses ONLY this vocabulary:\n\
             <|par|>a paragraph<|/par|>\n\
             <|list|><|item|>bullet<|/item|><|item|>bullet<|/item|><|/list|>\n\
             <|codeblock|>code here<|/codeblock|>\n\
             <|code|>inline code<|/code|>  <|bold|>bold<|/bold|>  <|italic|>italic<|/italic|>\n\
             <|heading|>section heading<|/heading|>  <|subheading|>smaller heading<|/subheading|>\n\
             <|quote|>a quoted line<|/quote|>  <|divider|>\n\
             <|link|>example.com/page<|/link|>\n\
           As in the examples, markup is sprinkled rather than total — plain prose and\n\
           markdown-style lines between tags are fine — but every reply must use at least\n\
           some of it, and every opened tag must be closed.\n\
           Never use <|user|>, <|assistant|>, <|system|>, <|endoftext|>, or any tag not\n\
           listed above. Write every tag BARE, with no attributes: <|codeblock|> (not\n\
           <|codeblock|lang='python'|>). A tag containing '=' or a second '|' is rejected.\n\
           Closing tags are spelled <|/par|> — the slash goes INSIDE, after the bar.\n\
           </|par|> is wrong.{length_rule}\n\
         \n\
         {thinking_rule}\
         Content requirements for this batch:\n\
         - Spread the conversations across these topics: {topics}.\n\
         - Use these conversation shapes across the batch: {shapes}.\n\
         - Where they genuinely fit, exercise: {features}. Never force a feature — a\n\
           terse question still gets a terse answer.\n\
         - Technical content must be CORRECT and self-contained. Invent plausible file\n\
           paths, run directories, and numbers; no real personal data, no current events.\n\
         - Every conversation in this batch opens on a DIFFERENT question from the others,\n\
           in a different voice. Two conversations that differ only in which file path\n\
           they name are one conversation.\n\
         {feedback}\
         \n\
         Before you emit each conversation, check it against the four things that\n\
         actually get output thrown away, in this order:\n\
           1. every {open}tag{close} you opened is closed, and closers are spelled\n\
              {open}/tag{close} — the slash INSIDE, after the bar,\n\
           2. the last turn is an assistant turn, and the roles strictly alternate\n\
              starting from user,\n\
           3. no {open}…{close} of any kind appears in a user turn,\n\
           4. the assistant turn uses at least one markup tag from the vocabulary.\n\
         \n\
         Output the {batch} conversations now, starting with {conv}:",
        n_ex = exemplars.len(),
        exemplars = exemplar_text.join("\n\n"),
        batch = request.batch,
        conv = D_CONV,
        user = D_USER,
        assistant = D_ASSISTANT,
        end = D_END,
        system_note = system_note,
        max_chars = MAX_ASSISTANT_CHARS,
        length_rule = length_rule,
        thinking_rule = thinking_rule,
        topics = topics,
        shapes = shapes,
        features = features,
        feedback = render_feedback(feedback),
        open = "<|",
        close = "|>",
    )
}

// ── exemplars ───────────────────────────────────────────────────────────────

// Gold lines that are transcript noise rather than training targets: bare API
// errors, model-fallback blobs, empty tool-result echoes. Showing these to the
// generator would teach it to emit error strings as answers.
fn is_junk(text: &str) -> bool {
    const NEEDLES: &[&str] = &[
        "API Error",
        "does not support tools",
        "may not exist or you may not have access",
        "\"type\": \"fallback\"",
        "{\"text\": \"\"",
    ];
    NEEDLES.iter().any(|needle| text.contains(needle))
}

// Gold conversations worth imitating, small enough to fit in a prompt.
fn load_exemplars(path: &Path, max_chars: usize) -> Vec<Conversation> {
    let Ok(content) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut pool = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(conv) = serde_json::from_str::<Conversation>(line) else {
            continue;
        };
        if line.len() > max_chars {
            continue; // a 100k pasted-log conversation will not fit a prompt
        }
        let assistant: String = conv
            .turns
            .iter()
            .filter(|t| t.role == "assistant")
            .map(|t| t.text.as_str())
            .collect();
        if assistant.len() < 150 || is_junk(&assistant) {
            continue;
        }
        pool.push(conv);
    }
    pool
}

// Two or three exemplars fitting a combined character budget.
fn pick_exemplars<'a>(pool: &'a [Conversation], budget: usize, rng: &mut Rng) -> Vec<&'a Conversation> {
    let mut order: Vec<usize> = (0..pool.len()).collect();
    rng.shuffle(&mut order);

    let mut picks = Vec::new();
    let mut used = 0usize;
    for i in order {
        let size = pool[i].user_chars() + pool[i].assistant_chars();
        if used + size > budget {
            continue;
        }
        picks.push(&pool[i]);
        used += size;
        if picks.len() >= 3 {
            break;
        }
    }
    // Everything in the pool is bigger than the budget: show the smallest one
    // anyway rather than send a prompt with no examples in it, which is the one
    // thing that reliably produces off-style output.
    if picks.is_empty() {
        if let Some(smallest) = pool
            .iter()
            .min_by_key(|c| c.user_chars() + c.assistant_chars())
        {
            picks.push(smallest);
        }
    }
    picks
}

// ── the model call ──────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    max_tokens: usize,
    stream: bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatContent,
}

#[derive(Deserialize)]
struct ChatContent {
    content: Option<String>,
}

// One completion from an OpenAI-compatible `/chat/completions` endpoint.
//
// That request shape is the de-facto standard — OpenRouter, DeepSeek, Groq,
// Together, Google's compatibility layer and a local Ollama or llama.cpp server all
// speak it — so a base URL and a model name are enough to move a run between
// providers without touching this code.
//
// A failed call returns `Err` and the run continues: one bad call out of thirty is
// a rate limit or a provider hiccup, not a reason to throw away the conversations
// already written.
/// A failed call, and whether trying again could ever go differently.
///
/// The distinction is the whole point: a timeout or a rate limit is worth another
/// call, and a rejected credential is not. Without it a misconfigured key spends the
/// entire call budget re-learning the same 401.
struct CallError {
    message: String,
    fatal: bool,
}

impl CallError {
    fn retryable(message: String) -> Self {
        Self {
            message,
            fatal: false,
        }
    }
}

async fn call_model(
    client: &reqwest::Client,
    request: &SynthRequest,
    api_key: Option<&str>,
    prompt: &str,
) -> Result<String, CallError> {
    let url = format!("{}/chat/completions", request.base_url.trim_end_matches('/'));
    let body = ChatRequest {
        model: &request.model,
        messages: vec![ChatMessage {
            role: "user",
            content: prompt,
        }],
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        stream: false,
    };

    let mut post = client.post(&url).json(&body);
    if let Some(key) = api_key {
        post = post.bearer_auth(key);
    }

    let response = post.send().await.map_err(|e| {
        // The URL is safe to show (it came from the form); the key never appears in
        // it, because it travels in a header.
        CallError::retryable(format!("request to {url} failed: {e}"))
    })?;

    let status = response.status();
    if !status.is_success() {
        let detail = response.text().await.unwrap_or_default();
        let detail: String = detail.chars().take(300).collect();
        let mut message = format!("HTTP {status} from the provider: {detail}");

        // A rejected credential is the same on every remaining call, so recognising
        // it is what stops a run from spending its whole budget re-learning one
        // mistake. Status alone does not identify it: OpenRouter answers a bad key
        // with 401, Google's OpenAI-compatible endpoint with **400 INVALID_ARGUMENT**
        // ("Invalid Auth key."). So a 400 counts only when the body says the problem
        // was the credential — an unqualified 400 is a malformed request, and the
        // next prompt is a different one.
        let auth_shaped = {
            let lower = detail.to_ascii_lowercase();
            ["auth", "api key", "api_key", "credential", "unauthorized", "permission denied"]
                .iter()
                .any(|needle| lower.contains(needle))
        };
        let fatal = matches!(status.as_u16(), 401 | 403) || (status == 400 && auth_shaped);
        if fatal {
            let host = request
                .base_url
                .split('/')
                .nth(2)
                .unwrap_or(request.base_url.as_str());
            message.push_str(&match api_key {
                Some(_) => format!(
                    " — {host} rejected the key in ${}. This is what a key from a \
                     different provider looks like, whatever the message says; \
                     check that the key and the base URL belong together.",
                    request.api_key_env
                ),
                None => format!(
                    " — no key was sent, because ${} is unset on the server.",
                    request.api_key_env
                ),
            });
        }

        return Err(CallError { message, fatal });
    }

    let parsed: ChatResponse = response
        .json()
        .await
        .map_err(|e| CallError::retryable(format!("unexpected response shape: {e}")))?;
    Ok(parsed
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .unwrap_or_default())
}

// Runs `work` unless the cancel flag goes up first, in which case it returns
// `None` and the future is dropped.
//
// Polled rather than woken by a channel because the flag is an `AtomicBool` shared
// with the queue's own cancellation path, and a quarter of a second of latency on a
// job whose steps are measured in minutes is not worth a second mechanism.
async fn cancellable<T>(
    cancel: &AtomicBool,
    work: impl std::future::Future<Output = T>,
) -> Option<T> {
    tokio::pin!(work);
    loop {
        tokio::select! {
            biased;
            result = &mut work => return Some(result),
            _ = tokio::time::sleep(Duration::from_millis(250)) => {
                if cancel.load(Ordering::SeqCst) {
                    return None;
                }
            }
        }
    }
}

// ── the run ─────────────────────────────────────────────────────────────────

// Generates until `count` conversations have been added or the call budget runs
// out. `Ok(None)` means cancelled — a distinct outcome from an error, and from a
// run that simply fell short.
//
// Everything written is appended as it is accepted rather than buffered to the end,
// so a cancelled or crashed run keeps what it had already produced. That is the
// same reasoning as the auto-labeller's append-only catalogue: generated examples
// are immutable once judged, and there is nothing to reconcile.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    request: &SynthRequest,
    on_progress: ProgressCallback,
    on_call: CallCallback,
    on_accepted: AcceptedCallback,
    cancel: Arc<AtomicBool>,
) -> Result<Option<SynthStats>, String> {
    let dir = storage::sft_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("could not open the SFT directory: {e}"))?;
    let exemplar_path = dir.join(&request.exemplar_file);
    let out_path = dir.join(&request.out_file);

    let budget = match request.id_model {
        Some(id) => Some(Budget::load(&storage::models_dir().join(id.to_string()))?),
        None => None,
    };

    on_progress("Reading the gold set", 0.0);
    let pool = load_exemplars(&exemplar_path, EXEMPLAR_BUDGET);
    if pool.is_empty() {
        return Err(format!(
            "no usable exemplars in {} — generation copies the style of real \
             conversations, and with none to show it produces generic chat data",
            request.exemplar_file
        ));
    }

    // Dedup against every SFT file, not just the output one, so generated openers
    // never echo the gold set either.
    let mut seen = load_openers(&dir);

    let api_key = std::env::var(&request.api_key_env).ok().filter(|k| !k.is_empty());
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(request.timeout_secs))
        .build()
        .map_err(|e| format!("could not build the HTTP client: {e}"))?;

    // Measured once, over the whole gold set rather than the two or three exemplars
    // a given call happens to draw — the ratio is a property of the tokenizer and the
    // corpus, so a wider sample is both steadier and cheaper than re-measuring per
    // call. Zero when no model was named, which means no length rule is stated.
    let pool_refs: Vec<&Conversation> = pool.iter().collect();
    let capacity = budget
        .as_ref()
        .map(|b| b.word_capacity(&pool_refs))
        .unwrap_or(0);
    if capacity > 0 {
        eprintln!(
            "[synth] window holds ~{capacity} words per conversation{}",
            if capacity < 120 { " — too small for thinking blocks, omitting them" } else { "" }
        );
    }

    let mut rng = Rng::new(request.seed.unwrap_or_else(|| Uuid::new_v4().as_u128() as u64));
    let max_calls = if request.max_calls > 0 {
        request.max_calls
    } else {
        (request.count / request.batch.max(1) + 1) * 3
    };

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .map_err(|e| format!("could not open {} for writing: {e}", request.out_file))?;

    let mut added = 0usize;
    let mut calls = 0usize;
    let mut duplicates = 0usize;
    let mut with_thinking = 0usize;
    let mut rejected: Vec<(String, usize)> = Vec::new();
    // Backstop for the failures that are not auth and still will not fix themselves:
    // a model name the provider does not have, a base URL with no endpoint behind
    // it, a service that is down. One failure is noise and worth retrying; several
    // in a row with nothing in between is a wall, and continuing to call it just
    // makes the user wait longer for the same error.
    let mut consecutive_failures = 0usize;
    const FAILURE_LIMIT: usize = 3;

    // Carries the previous call's verdicts into the next prompt. Seeded with the
    // openers already on disk, so even the first call knows what not to write again.
    let mut feedback = Feedback {
        used_openers: seen.iter().take(AVOID_OPENERS).map(|o| opener_preview(o)).collect(),
        ..Feedback::default()
    };

    while added < request.count && calls < max_calls {
        if cancel.load(Ordering::SeqCst) {
            return Ok(None);
        }
        calls += 1;

        let fraction = added as f64 / request.count as f64;
        on_progress(&format!("Call {calls} · {added}/{}", request.count), fraction);

        let exemplars = pick_exemplars(&pool, EXEMPLAR_BUDGET, &mut rng);
        let prompt = build_prompt(request, &exemplars, capacity, &feedback, &mut rng);

        let started = std::time::Instant::now();
        // Raced against the cancel flag rather than simply checked between calls: a
        // request can legitimately take minutes, and a stop button that waits for
        // somebody else's server before it does anything is not a stop button. The
        // in-flight response is dropped, which is the whole cost of stopping here —
        // everything accepted so far is already on disk.
        let call = cancellable(&cancel, call_model(&client, request, api_key.as_deref(), &prompt));
        let Some(result) = call.await else {
            return Ok(None);
        };
        let raw = match result {
            Ok(raw) => raw,
            Err(e) => {
                // Reported as a call with nothing kept rather than swallowed: a run
                // whose calls are all failing looks identical to a weak model unless
                // the reason is on screen.
                eprintln!("[synth] call {calls}: {}", e.message);
                // A rejected credential will be rejected identically by every call
                // left, so the run ends here and the message becomes the job's
                // error. Anything already accepted is on disk and stays there.
                if e.fatal {
                    return Err(e.message);
                }
                consecutive_failures += 1;
                if consecutive_failures >= FAILURE_LIMIT {
                    return Err(format!(
                        "{} calls in a row failed, so the run stopped rather than \
                         work through the rest of its budget. Last failure: {}",
                        consecutive_failures, e.message
                    ));
                }
                let report = CallReport {
                    call: calls,
                    kept: 0,
                    duplicates: 0,
                    rejected: vec![(e.message, 1)],
                    elapsed_ms: started.elapsed().as_millis(),
                    response_chars: 0,
                    total: added,
                    target: request.count,
                };
                on_call(&report);
                continue;
            }
        };

        // The provider answered, so whatever the run does with the content, the
        // wall is not there. Counting only *consecutive* failures keeps an
        // occasional rate limit from ending a long run.
        consecutive_failures = 0;

        let mut call_rejected: Vec<(String, usize)> = Vec::new();
        let mut kept = 0usize;
        let mut call_duplicates = 0usize;
        // What this call opened its conversations with, for the next call's
        // avoid-list. The most useful entries there are the freshest ones: the
        // repetition worth naming is the model's own last answer to the same prompt.
        let mut call_openers: Vec<String> = Vec::new();

        for mut conv in split_delimited(strip_reasoning(&raw)) {
            if added >= request.count {
                break;
            }
            for turn in &mut conv.turns {
                // Only assistant turns: markup in a user turn is a rule violation,
                // not a typo, and repairing it would hide that.
                if turn.role == "assistant" {
                    turn.text = repair_markup(&turn.text);
                }
            }

            if let Err(reason) = validate(&conv, request.min_think_words) {
                tally(&mut call_rejected, reason);
                continue;
            }

            let tokens = budget.as_ref().map(|b| b.tokens(&conv));
            if let (Some(tokens), Some(budget)) = (tokens, budget.as_ref()) {
                if tokens > budget.ctx_window {
                    tally(
                        &mut call_rejected,
                        format!("over the {} token window", budget.ctx_window),
                    );
                    continue;
                }
            }

            let key = conv.opener_key();
            if !seen.insert(key) {
                call_duplicates += 1;
                continue;
            }

            let line = serde_json::to_string(&conv)
                .map_err(|e| format!("could not encode a conversation: {e}"))?;
            writeln!(file, "{line}")
                .map_err(|e| format!("could not write to {}: {e}", request.out_file))?;

            let think_words = conv
                .turns
                .iter()
                .map(|t| thinking_block(&t.text).0)
                .max()
                .unwrap_or(0);
            if think_words > 0 {
                with_thinking += 1;
            }
            added += 1;
            kept += 1;
            call_openers.push(opener_preview(&conv.turns[0].text));

            on_accepted(&Accepted {
                opener: conv.turns[0].text.clone(),
                reply: conv
                    .turns
                    .last()
                    .map(|t| t.text.clone())
                    .unwrap_or_default(),
                turns: conv.turns.len(),
                think_words,
                tokens,
                user_chars: conv.user_chars(),
                assistant_chars: conv.assistant_chars(),
            });
        }

        // Flushed per call rather than per line: a cancelled run should keep its
        // work, and a call is the granularity at which that matters.
        let _ = file.flush();
        duplicates += call_duplicates;
        for (reason, n) in &call_rejected {
            tally_n(&mut rejected, reason.clone(), *n);
        }

        // What the next call is told about this one. The avoid-list leads with this
        // call's own openers and fills from the corpus behind them, so it always
        // names the repetition that just happened before older ones.
        let mut reasons = call_rejected.clone();
        reasons.sort_by(|a, b| b.1.cmp(&a.1));
        let mut used_openers = call_openers;
        for older in seen.iter().map(|o| opener_preview(o)) {
            if used_openers.len() >= AVOID_OPENERS {
                break;
            }
            if !used_openers.contains(&older) {
                used_openers.push(older);
            }
        }
        used_openers.truncate(AVOID_OPENERS);
        feedback = Feedback {
            reasons,
            duplicates: call_duplicates,
            used_openers,
        };

        on_call(&CallReport {
            call: calls,
            kept,
            duplicates: call_duplicates,
            rejected: call_rejected,
            elapsed_ms: started.elapsed().as_millis(),
            response_chars: raw.len(),
            total: added,
            target: request.count,
        });
    }

    on_progress("Done", 1.0);
    rejected.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(Some(SynthStats {
        added,
        calls,
        duplicates,
        with_thinking,
        rejected,
        out_file: request.out_file.clone(),
        corpus_total: count_conversations(&dir),
    }))
}

// Chars of exemplars embedded per prompt. Enough for two or three real
// conversations; past this the prompt starts costing more than the output.
const EXEMPLAR_BUDGET: usize = 9000;

fn tally(counts: &mut Vec<(String, usize)>, reason: String) {
    tally_n(counts, reason, 1);
}

fn tally_n(counts: &mut Vec<(String, usize)>, reason: String, n: usize) {
    match counts.iter_mut().find(|(r, _)| *r == reason) {
        Some((_, count)) => *count += n,
        None => counts.push((reason, n)),
    }
}

// Openers already present in any SFT file. A `HashSet` of normalized first user
// messages is the whole dedup index — cheap, and it is the key near-duplicates
// actually share.
fn load_openers(dir: &Path) -> HashSet<String> {
    let mut seen = HashSet::new();
    for path in jsonl_files(dir) {
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        for line in content.lines() {
            if let Ok(conv) = serde_json::from_str::<Conversation>(line) {
                if !conv.turns.is_empty() {
                    seen.insert(conv.opener_key());
                }
            }
        }
    }
    seen
}

pub fn jsonl_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("jsonl"))
        .collect();
    paths.sort();
    paths
}

fn count_conversations(dir: &Path) -> usize {
    jsonl_files(dir)
        .iter()
        .filter_map(|p| fs::read_to_string(p).ok())
        .map(|content| content.lines().filter(|l| !l.trim().is_empty()).count())
        .sum()
}

// ── corpus stats ────────────────────────────────────────────────────────────

// What one SFT file holds. Computed by reading it and rendering every conversation
// through the real tokenizer, which is why this lives here rather than being
// estimated in the browser from character counts.
#[derive(Debug, Clone, Serialize)]
pub struct FileStats {
    pub file: String,
    pub conversations: usize,
    pub turns: usize,
    pub with_thinking: usize,
    // Token lengths of every conversation, so the page can draw the distribution
    // against the context window rather than just a mean.
    pub tokens: Vec<usize>,
    pub over_budget: usize,
    pub user_chars_median: usize,
    pub assistant_chars_median: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorpusStats {
    pub files: Vec<FileStats>,
    pub conversations: usize,
    // Present only when a model was named; without one there is no window to
    // measure against and the page says so rather than inventing a number.
    pub ctx_window: Option<usize>,
    pub over_budget: usize,
}

// Reads the whole SFT corpus. Blocking (file IO plus BPE encoding over every
// conversation), so the route runs it on a blocking thread.
pub fn corpus_stats(id_model: Option<Uuid>) -> Result<CorpusStats, String> {
    let dir = storage::sft_dir();
    let budget = match id_model {
        Some(id) => Some(Budget::load(&storage::models_dir().join(id.to_string()))?),
        None => None,
    };

    let mut files = Vec::new();
    let mut conversations = 0usize;
    let mut over_budget = 0usize;

    for path in jsonl_files(&dir) {
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut stats = FileStats {
            file: name,
            conversations: 0,
            turns: 0,
            with_thinking: 0,
            tokens: Vec::new(),
            over_budget: 0,
            user_chars_median: 0,
            assistant_chars_median: 0,
        };
        let mut user_chars = Vec::new();
        let mut assistant_chars = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(conv) = serde_json::from_str::<Conversation>(line) else {
                continue;
            };
            stats.conversations += 1;
            stats.turns += conv.turns.len();
            if conv
                .turns
                .iter()
                .any(|t| thinking_block(&t.text).0 > 0)
            {
                stats.with_thinking += 1;
            }
            for turn in &conv.turns {
                if turn.role == "user" {
                    user_chars.push(turn.text.len());
                } else {
                    assistant_chars.push(turn.text.len());
                }
            }
            if let Some(budget) = budget.as_ref() {
                let tokens = budget.tokens(&conv);
                stats.tokens.push(tokens);
                if tokens > budget.ctx_window {
                    stats.over_budget += 1;
                }
            }
        }

        stats.user_chars_median = median(&mut user_chars);
        stats.assistant_chars_median = median(&mut assistant_chars);
        conversations += stats.conversations;
        over_budget += stats.over_budget;
        files.push(stats);
    }

    Ok(CorpusStats {
        files,
        conversations,
        ctx_window: budget.as_ref().map(|b| b.ctx_window),
        over_budget,
    })
}

fn median(values: &mut [usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

// ── a small seedable PRNG ───────────────────────────────────────────────────

// xorshift64*. Sampling topics and exemplars needs variety, not cryptography, and a
// seedable generator means a run can be repeated exactly from its stored config —
// which a thread-local RNG could not offer.
pub struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        // Zero is the one state xorshift cannot leave.
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_u64() % n as u64) as usize }
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    // Fisher-Yates, so every ordering is equally likely.
    fn shuffle(&mut self, items: &mut [usize]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }
}

// `k` distinct entries of `items`, in a random order.
fn pick(items: &[&'static str], k: usize, rng: &mut Rng) -> Vec<&'static str> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    rng.shuffle(&mut order);
    order.into_iter().take(k.min(items.len())).map(|i| items[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(turns: &[(&str, &str)]) -> Conversation {
        Conversation {
            system: None,
            turns: turns
                .iter()
                .map(|(role, text)| Turn {
                    role: role.to_string(),
                    text: text.to_string(),
                })
                .collect(),
        }
    }

    #[test]
    fn the_tag_vocabulary_comes_from_the_tokenizer() {
        let (paired, void) = vocabulary();
        // Present because the tokenizer has both halves.
        assert!(paired.contains("par"));
        assert!(paired.contains("codeblock"));
        // Present because the tokenizer has only the opening form.
        assert!(void.contains("divider"));
        // Scaffold the chat template supplies itself: never valid in a reply.
        assert!(!paired.contains("assistant"));
        assert!(!paired.contains("system"));
        // Handled separately — allowed, but only as a leading block.
        assert!(!paired.contains("thinking"));
    }

    #[test]
    fn a_well_formed_reply_passes() {
        let c = conv(&[("user", "why is it slow?"), ("assistant", "<|par|>Disk.<|/par|>")]);
        assert!(validate(&c, 0).is_ok());
    }

    #[test]
    fn markup_in_a_user_turn_is_a_rule_violation_not_a_typo() {
        let c = conv(&[
            ("user", "<|par|>why<|/par|>"),
            ("assistant", "<|par|>because<|/par|>"),
        ]);
        assert_eq!(validate(&c, 0).unwrap_err(), "markup in a user turn");
    }

    #[test]
    fn the_xml_shaped_closer_is_repaired_rather_than_discarded() {
        // Models reach for `</|par|>` constantly; the rewrite is lossless, so a
        // conversation that is otherwise good survives it.
        assert_eq!(repair_markup("<|par|>hi</|par|>"), "<|par|>hi<|/par|>");
        let c = conv(&[
            ("user", "hi"),
            ("assistant", &repair_markup("<|par|>hi</|par|>")),
        ]);
        assert!(validate(&c, 0).is_ok());
    }

    #[test]
    fn a_reply_with_no_markup_at_all_is_rejected() {
        // SFT is the only place the model learns the vocabulary exists, so a corpus
        // of unmarked prose teaches it that markup is optional.
        let c = conv(&[("user", "why?"), ("assistant", "just because")]);
        assert_eq!(validate(&c, 0).unwrap_err(), "no markup at all");
    }

    #[test]
    fn unbalanced_and_attributed_tags_are_rejected() {
        let unbalanced = conv(&[("user", "hi"), ("assistant", "<|par|>hi")]);
        assert_eq!(validate(&unbalanced, 0).unwrap_err(), "unbalanced markup");

        // An attribute form byte-shreds instead of tokenizing to one id.
        let attributed = conv(&[
            ("user", "hi"),
            ("assistant", "<|codeblock|lang='py'|>x<|/codeblock|>"),
        ]);
        assert_eq!(validate(&attributed, 0).unwrap_err(), "tag with attributes");
    }

    #[test]
    fn a_short_thinking_block_is_rejected_but_none_at_all_is_fine() {
        let token_gesture = conv(&[
            ("user", "why?"),
            ("assistant", "<|thinking|>hmm ok<|/thinking|><|par|>Disk.<|/par|>"),
        ]);
        assert!(validate(&token_gesture, 10).is_err());

        let none = conv(&[("user", "why?"), ("assistant", "<|par|>Disk.<|/par|>")]);
        assert!(validate(&none, 10).is_ok());
    }

    #[test]
    fn roles_must_alternate_and_end_on_the_assistant() {
        let out_of_order = conv(&[("assistant", "<|par|>hi<|/par|>"), ("user", "hi")]);
        assert_eq!(validate(&out_of_order, 0).unwrap_err(), "roles out of order");

        let unanswered = conv(&[("user", "hi")]);
        assert_eq!(
            validate(&unanswered, 0).unwrap_err(),
            "does not end on assistant"
        );
    }

    #[test]
    fn a_small_window_drops_the_thinking_block_instead_of_demanding_the_impossible() {
        // The failure this pins: at ctx 512 our tokenizer leaves roughly 150 words for
        // a whole conversation, while the request asked for a 120-word *minimum*
        // thinking block, "typically 180-340". Both cannot hold. A model given
        // contradictory instructions does not produce a shorter conversation — it
        // produces a long one that fails the budget check, which is exactly what
        // happened: six generated, six discarded, four of them for length.
        let gold = Conversation {
            system: None,
            turns: vec![
                Turn { role: "user".into(), text: "why is the run stalling?".into() },
                Turn { role: "assistant".into(), text: "<|par|>The loader is single threaded.<|/par|>".into() },
            ],
        };
        let refs = vec![&gold];
        let mut request = SynthRequest {
            base_url: "http://x".into(), model: "m".into(), count: 1, batch: 1,
            temperature: 1.0, max_tokens: 100, min_think_words: 120, think_frac: 1.0,
            system_frac: 0.0, timeout_secs: 10, max_calls: 1, id_model: None,
            api_key_env: "X".into(), exemplar_file: "g.jsonl".into(),
            out_file: "o.jsonl".into(), seed: Some(1),
        };
        let mut rng = Rng::new(1);

        // No model named: nothing was measured, so no limit is invented and the
        // thinking rule stands as configured.
        let prompt = build_prompt(&request, &refs, 0, &Feedback::default(), &mut rng);
        assert!(prompt.contains("THINKING BLOCKS"));
        assert!(!prompt.contains("WORDS"), "no budget means no length rule");

        // A window too small for a real block drops the section outright rather than
        // asking for a stub — a 30-word scratchpad would teach the model to open a
        // thinking block and say nothing.
        let prompt = build_prompt(&request, &refs, 80, &Feedback::default(), &mut rng);
        assert!(!prompt.contains("THINKING BLOCKS"), "an impossible demand must be dropped");
        assert!(prompt.contains("80 WORDS"), "the real limit must be stated");

        // A window with room keeps it, clamped to half the budget so the visible
        // reply still has somewhere to live.
        let prompt = build_prompt(&request, &refs, 600, &Feedback::default(), &mut rng);
        assert!(prompt.contains("THINKING BLOCKS"));
        assert!(prompt.contains("600 WORDS"));
        assert!(prompt.contains("at least 120 words"), "the request's own floor fits here");

        // ...and clamped when the request asks for more than half the window.
        request.min_think_words = 400;
        let prompt = build_prompt(&request, &refs, 600, &Feedback::default(), &mut rng);
        assert!(prompt.contains("at least 300 words"), "clamped to half of 600");
    }

    #[test]
    fn a_reasoning_preamble_is_dropped_before_the_output_is_parsed() {
        // Shaped after a real Gemma 4 response: the thought rehearses the format,
        // so the draft markers arrive before the genuine ones.
        let raw = "<thought>Format: `<<<CONV>>>`, `<<<USER>>>`.\n\
                   *   `<<<CONV>>>`\n*   `<<<USER>>>`\n*   draft question\n\
                   </thought>\n\
                   <<<CONV>>>\n<<<USER>>>\nwhat is a slice?\n<<<ASSISTANT>>>\n\
                   <|par|>A view into a buffer.<|/par|>\n<<<END>>>";

        let convs = split_delimited(strip_reasoning(raw));
        assert_eq!(convs.len(), 1, "the rehearsal must not become a conversation");
        assert_eq!(convs[0].turns[0].text, "what is a slice?");
    }

    #[test]
    fn a_truncated_thought_yields_nothing_rather_than_scratch_work() {
        // Hitting the token ceiling mid-thought: the draft markers are all there is.
        let raw = "<thought>*   `<<<CONV>>>`\n*   `<<<USER>>>`\n*   half a quest";
        assert_eq!(split_delimited(strip_reasoning(raw)).len(), 0);
    }

    #[test]
    fn a_reasoning_tag_that_is_not_a_preamble_is_left_alone() {
        // A conversation *about* markup keeps its text; only a preamble is cut.
        let raw = "<<<CONV>>>\n<<<USER>>>\nwhat does <think> mean?\n\
                   <<<ASSISTANT>>>\n<|par|>It marks reasoning.<|/par|>\n<<<END>>>";
        let convs = split_delimited(strip_reasoning(raw));
        assert_eq!(convs.len(), 1);
        assert!(convs[0].turns[0].text.contains("<think>"));
    }

    #[test]
    fn delimited_output_survives_commentary_around_it() {
        // Real output from a small model: a banner before, a fence after, and no
        // trailing <<<END>>> because the response hit its token cap.
        let raw = "Sure! Here are the conversations:\n\
                   <<<CONV>>>\n<<<USER>>>\nwhy is it slow?\n<<<ASSISTANT>>>\n\
                   <|par|>Disk.<|/par|>\n<<<END>>>\n```\n\
                   <<<CONV>>>\n<<<USER>>>\nand now?\n<<<ASSISTANT>>>\n<|par|>Better.<|/par|>";
        let convs = split_delimited(raw);
        assert_eq!(convs.len(), 2);
        assert_eq!(convs[0].turns[0].text, "why is it slow?");
        assert_eq!(convs[1].turns[1].text, "<|par|>Better.<|/par|>");
    }

    #[test]
    fn a_multi_line_turn_keeps_its_newlines() {
        // The whole reason for the delimiter format: turn text is written literally,
        // so a pasted log survives without any escaping to get wrong.
        let raw = "<<<CONV>>>\n<<<USER>>>\nline one\nline two\n<<<ASSISTANT>>>\n\
                   <|par|>ok<|/par|>\n<<<END>>>";
        let convs = split_delimited(raw);
        assert_eq!(convs[0].turns[0].text, "line one\nline two");
    }

    #[test]
    fn a_system_block_is_read_and_kept_off_the_turn_list() {
        let raw = "<<<CONV>>>\n<<<SYSTEM>>>\nBe terse.\n<<<USER>>>\nhi\n\
                   <<<ASSISTANT>>>\n<|par|>hi<|/par|>\n<<<END>>>";
        let convs = split_delimited(raw);
        assert_eq!(convs[0].system.as_deref(), Some("Be terse."));
        assert_eq!(convs[0].turns.len(), 2);
    }

    #[test]
    fn the_opener_key_ignores_whitespace_and_case() {
        let a = conv(&[("user", "Why  is\nit slow?"), ("assistant", "x")]);
        let b = conv(&[("user", "why is it slow?"), ("assistant", "y")]);
        assert_eq!(a.opener_key(), b.opener_key());
    }

    #[test]
    fn the_rng_is_reproducible_from_its_seed() {
        // The reason a run stores its seed: the same config generates the same
        // prompts, which is what makes two runs comparable.
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        assert_eq!(a.next_u64(), b.next_u64());
        assert_eq!(a.below(100), b.below(100));
    }
}


#[cfg(test)]
mod e2e {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    /// Held for the length of any test that sets `DATA_DIR`.
    ///
    /// `storage::sft_dir()` reads that variable per call, so it is process-wide
    /// state and cargo runs these tests on threads of one process — two of them
    /// pointing it at different directories at once means one reads the other's.
    /// Poisoning is ignored deliberately: a panicking test has already failed, and
    /// taking the rest of the module down with it hides which one it was.
    /// Shared with `training`'s tests, which also repoint `DATA_DIR` — this module
    /// and that one each used to hold a private lock over the same process-global,
    /// so neither excluded the other and a `synth` test could move `DATA_DIR` out
    /// from under a `training` build mid-run.
    fn exclusive_data_dir() -> std::sync::MutexGuard<'static, ()> {
        storage::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    // A one-shot OpenAI-compatible endpoint returning canned delimiter output.
    fn mock_provider(body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(4) {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut len = 0usize;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if let Some(v) = line.strip_prefix("content-length: ")
                        .or_else(|| line.strip_prefix("Content-Length: "))
                    { len = v.trim().parse().unwrap(); }
                    if line == "\r\n" || line.is_empty() { break; }
                }
                let mut buf = vec![0u8; len];
                std::io::Read::read_exact(&mut reader, &mut buf).unwrap();
                let payload = serde_json::json!({
                    "choices": [{"message": {"content": body}}]
                }).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                    payload.len(), payload);
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        format!("http://127.0.0.1:{port}/v1")
    }

    #[tokio::test]
    async fn a_run_writes_only_what_survives_validation() {
        let _guard = exclusive_data_dir();
        let dir = std::env::temp_dir().join(format!("synth-e2e-{}", Uuid::new_v4()));
        let sft = dir.join("raw/sft");
        fs::create_dir_all(&sft).unwrap();
        unsafe { std::env::set_var("DATA_DIR", &dir) };

        fs::write(sft.join("gold.jsonl"), serde_json::json!({
            "turns": [
                {"role":"user","text":"why is the run stalling?"},
                {"role":"assistant","text":"<|par|>The loader is single threaded, so the GPU waits on disk for most of every step and the device sits idle between batches. Raise the worker count to four and re-run; if the step time does not move, the bottleneck is the collate function rather than the read.<|/par|>"}
            ]
        }).to_string() + "\n").unwrap();

        // Two conversations: one valid, one with markup in the user turn.
        let base_url = mock_provider(
            "<<<CONV>>>\n<<<USER>>>\nwhere is the tokenizer?\n<<<ASSISTANT>>>\n\
             <|par|>In <|code|>src/tokenize<|/code|>.<|/par|>\n<<<END>>>\n\
             <<<CONV>>>\n<<<USER>>>\n<|par|>bad<|/par|>\n<<<ASSISTANT>>>\n<|par|>x<|/par|>\n<<<END>>>",
        );

        let request = SynthRequest {
            // Asks for more than the mock can supply, so the call's whole output is
            // examined — with `count: 1` the run would (correctly) stop at its
            // target before ever looking at the second conversation.
            base_url, model: "mock".into(), count: 5, batch: 2, temperature: 1.0,
            max_tokens: 100, min_think_words: 0, think_frac: 0.0, system_frac: 0.0,
            timeout_secs: 10, max_calls: 1, id_model: None,
            api_key_env: "SYNTH_E2E_UNSET".into(),
            exemplar_file: "gold.jsonl".into(), out_file: "out.jsonl".into(), seed: Some(7),
        };

        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = calls.clone();
        let stats = run(
            &request,
            Box::new(|_, _| {}),
            Box::new(move |report| seen.lock().unwrap().push(report.clone())),
            Box::new(|_| {}),
            Arc::new(AtomicBool::new(false)),
        ).await.unwrap().unwrap();

        assert_eq!(stats.added, 1, "only the valid conversation is written");
        let written = fs::read_to_string(sft.join("out.jsonl")).unwrap();
        assert_eq!(written.lines().count(), 1);
        assert!(written.contains("where is the tokenizer?"));
        assert!(!written.contains("bad"));

        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].kept, 1);
        assert_eq!(calls[0].rejected, vec![("markup in a user turn".to_string(), 1)]);

        // The gold file is never written to.
        assert_eq!(fs::read_to_string(sft.join("gold.jsonl")).unwrap().lines().count(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    // Counts connections, so a test can prove the run stopped rather than retried.
    fn mock_rejecting_auth(hits: Arc<std::sync::atomic::AtomicUsize>) -> String {
        // OpenRouter's real wording for a key it does not recognise. It reads like
        // the header was missing, which is exactly why the run has to explain
        // itself rather than pass the message through.
        mock_status(hits, 401, r#"{"error":{"message":"Missing Authentication header","code":401}}"#)
    }

    // Google's OpenAI-compatible endpoint rejects a bad key with 400, not 401 — the
    // reason `fatal` cannot be decided from the status code alone.
    fn mock_google_bad_key(hits: Arc<std::sync::atomic::AtomicUsize>) -> String {
        mock_status(hits, 400, r#"{"error":{"code":400,"message":"Invalid Auth key.","status":"INVALID_ARGUMENT"}}"#)
    }

    fn mock_status(
        hits: Arc<std::sync::atomic::AtomicUsize>,
        code: u16,
        payload: &'static str,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(8) {
                hits.fetch_add(1, Ordering::SeqCst);
                let mut stream = stream.unwrap();
                // The request has to be read before the response is written, or the
                // client sees a reset mid-send and reports a connection failure —
                // which is retryable, and would test the wrong path entirely.
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut len = 0usize;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if let Some(v) = line.strip_prefix("content-length: ")
                        .or_else(|| line.strip_prefix("Content-Length: "))
                    { len = v.trim().parse().unwrap(); }
                    if line == "\r\n" || line.is_empty() { break; }
                }
                let mut buf = vec![0u8; len];
                std::io::Read::read_exact(&mut reader, &mut buf).unwrap();

                let response = format!(
                    "HTTP/1.1 {code} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                    payload.len(), payload);
                stream.write_all(response.as_bytes()).ok();
            }
        });
        format!("http://127.0.0.1:{port}/v1")
    }

    #[tokio::test]
    async fn a_rejected_key_ends_the_run_whatever_status_the_provider_used() {
        // OpenRouter's 401 and Google's 400 are the same fact in two shapes, and a
        // check written against either one alone would let the other spend the
        // whole budget.
        for make_mock in [mock_rejecting_auth, mock_google_bad_key] {
            rejected_key_ends_the_run(make_mock).await;
        }
    }

    async fn rejected_key_ends_the_run(
        make_mock: fn(Arc<std::sync::atomic::AtomicUsize>) -> String,
    ) {
        let _guard = exclusive_data_dir();
        let dir = std::env::temp_dir().join(format!("synth-auth-{}", Uuid::new_v4()));
        let sft = dir.join("raw/sft");
        fs::create_dir_all(&sft).unwrap();
        unsafe { std::env::set_var("DATA_DIR", &dir) };

        fs::write(sft.join("gold.jsonl"), serde_json::json!({
            "turns": [
                {"role":"user","text":"why is the run stalling?"},
                {"role":"assistant","text":"<|par|>The loader is single threaded, so the GPU waits on disk for most of every step and the device sits idle between batches. Raise the worker count to four and re-run; if the step time does not move, the bottleneck is the collate function rather than the read.<|/par|>"}
            ]
        }).to_string() + "\n").unwrap();

        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let base_url = make_mock(hits.clone());

        let request = SynthRequest {
            // Eight calls are allowed; a rejected key should still cost exactly one.
            base_url, model: "mock".into(), count: 20, batch: 2, temperature: 1.0,
            max_tokens: 100, min_think_words: 0, think_frac: 0.0, system_frac: 0.0,
            timeout_secs: 10, max_calls: 8, id_model: None,
            api_key_env: "SYNTH_AUTH_UNSET".into(),
            exemplar_file: "gold.jsonl".into(), out_file: "out.jsonl".into(), seed: Some(7),
        };

        let error = run(
            &request,
            Box::new(|_, _| {}),
            Box::new(|_| {}),
            Box::new(|_| {}),
            Arc::new(AtomicBool::new(false)),
        ).await.expect_err("a rejected key fails the job");

        assert_eq!(hits.load(Ordering::SeqCst), 1, "the run stops at the first 401");
        // The variable name is the actionable part — the provider's own message
        // points at the wrong thing.
        assert!(error.contains("SYNTH_AUTH_UNSET"), "unhelpful error: {error}");
        // Not asserted against a specific status: which one arrives is the
        // provider's choice, and pinning it here is what made this test wrong for
        // Google the first time.
        assert!(error.contains("HTTP"), "unhelpful error: {error}");
        fs::remove_dir_all(&dir).ok();
    }
}


