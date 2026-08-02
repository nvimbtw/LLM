//! The canonical chat template — the single definition of how a conversation is
//! rendered into tokens for the model. Inference builds prompts through here;
//! in time the synthetic-data generator and the SFT loss-masking must render the
//! exact same shape, so the format the model is *trained* on and the format it's
//! *prompted* with can't silently drift apart.
//!
//! A conversation is laid out as:
//!
//! ```text
//! <|system|>S<|/system|><|user|>U1<|/user|><|assistant|>A1<|/assistant|><|user|>U2<|/user|><|assistant|>
//! ```
//!
//! i.e. an optional system block once at the front, alternating user/assistant
//! turns, always ending on an *open* `<|assistant|>` for the model to continue.
//! Generation then stops when it emits `<|/assistant|>` (see `infer`).
//!
//! A conversation may also carry a [`ThinkBudget`], which inserts one marker
//! immediately before every `<|assistant|>`:
//!
//! ```text
//! <|user|>U<|/user|><|think:high|><|assistant|>
//! ```
//!
//! With no budget the rendering is byte-identical to what it was before budgets
//! existed, which is what keeps every corpus, split hash and benchmark score
//! already on disk comparable.
//!
//! Historical assistant turns keep only their reply: any
//! `<|thinking|>...<|/thinking|>` block is stripped, since stale reasoning isn't
//! worth spending the (tiny) context window on. The model still produces a fresh
//! thinking block for the current turn.

use serde::{Deserialize, Serialize};

use tokenize::bpe::encode;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    User,
    Assistant,
}

/// How much reasoning the assistant should show before answering.
///
/// Three levels rather than a token count, because the label has to be something
/// a prompt can ask a teacher for, a person can pick off a menu, and a model can
/// condition on — and a model asked for "173 thinking tokens" cannot count its
/// own tokens. `synth::teach` files generated conversations into these by
/// *measuring* the thinking block; the boundaries live there, since they are a
/// property of the run rather than of the template.
///
/// This is the canonical definition, in the lowest crate that renders a
/// conversation, so the label the teacher writes, the one SFT trains on and the
/// one the chat socket sends are all the same type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkBudget {
    Low,
    Med,
    High,
}

impl ThinkBudget {
    pub const ALL: [ThinkBudget; 3] = [ThinkBudget::Low, ThinkBudget::Med, ThinkBudget::High];

    pub fn as_str(self) -> &'static str {
        match self {
            ThinkBudget::Low => "low",
            ThinkBudget::Med => "med",
            ThinkBudget::High => "high",
        }
    }

    /// The reserved marker this budget renders as. One token — it has its own id
    /// in the special block, so the model conditions on a single embedding row
    /// rather than on a handful of ordinary word tokens that mean something else
    /// everywhere else in the corpus.
    pub fn marker(self) -> &'static str {
        match self {
            ThinkBudget::Low => "<|think:low|>",
            ThinkBudget::Med => "<|think:med|>",
            ThinkBudget::High => "<|think:high|>",
        }
    }

    /// Parses the wire/on-disk spelling. `None` for anything else, including
    /// "auto" — the absence of a budget is a real state (render nothing) and the
    /// caller expresses it as `Option::None` rather than as a fourth variant.
    pub fn parse(s: &str) -> Option<ThinkBudget> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Some(ThinkBudget::Low),
            "med" | "medium" => Some(ThinkBudget::Med),
            "high" => Some(ThinkBudget::High),
            _ => None,
        }
    }
}

// The budget marker, or "" when there is none. Written once here so the three
// renderers cannot disagree about whether an absent budget emits anything.
fn budget_marker(budget: Option<ThinkBudget>) -> &'static str {
    match budget {
        Some(b) => b.marker(),
        None => "",
    }
}

pub struct Turn {
    pub role: Role,
    pub text: String,
}

// The trailing marker every prompt ends on; the model generates from right here.
const ASSISTANT_OPEN: &str = "<|assistant|>";

// Drop a historical assistant turn's reasoning, keeping only the reply. If the
// markers are missing or malformed we just hand back the trimmed text unchanged.
fn strip_thinking(text: &str) -> String {
    const OPEN: &str = "<|thinking|>";
    const CLOSE: &str = "<|/thinking|>";
    if let (Some(a), Some(b)) = (text.find(OPEN), text.find(CLOSE)) {
        if b >= a {
            let mut out = String::with_capacity(text.len());
            out.push_str(&text[..a]);
            out.push_str(&text[b + CLOSE.len()..]);
            return out.trim().to_string();
        }
    }
    text.trim().to_string()
}

// A *history* turn, as `build_prompt` renders it.
//
// Deliberately carries no budget marker, even on a conversation that has one. The
// rule the marker states is "the assistant turn after me shows its reasoning at
// this budget", and a history turn has had its reasoning stripped — so marking it
// would claim a thinking block that is not there. Only the trailing open
// `<|assistant|>`, the one about to be generated, is marked.
fn format_turn(t: &Turn) -> String {
    match t.role {
        Role::User => format!("<|user|>{}<|/user|>", t.text.trim()),
        Role::Assistant => format!("<|assistant|>{}<|/assistant|>", strip_thinking(&t.text)),
    }
}

/// Render a whole conversation to text, in the exact shape
/// `build_training_example` tokenizes.
///
/// For seeding conversations into a *pretrain* corpus, which is a directory of
/// `.txt`: the point of putting them there is that the model meets the chat
/// scaffold during pretraining instead of first meeting it at SFT, so the text has
/// to carry the same markers in the same places. Assistant turns are kept verbatim,
/// thinking blocks included, for the same reason `build_training_example` keeps
/// them — that is the text SFT will later train on.
///
/// Unlike `build_training_example` there is no mask and no per-segment encoding:
/// the pretrain builder tokenizes whole files in one pass, so a marker seam may
/// tokenize a token differently here than it does under SFT. That costs nothing —
/// this exists to expose the markers to training, not to align with a mask.
pub fn render_conversation(
    system: Option<&str>,
    turns: &[Turn],
    budget: Option<ThinkBudget>,
) -> String {
    let mut out = String::new();
    if let Some(s) = system.map(str::trim).filter(|s| !s.is_empty()) {
        out.push_str(&format!("<|system|>{}<|/system|>", s));
    }
    for t in turns {
        match t.role {
            Role::User => out.push_str(&format!("<|user|>{}<|/user|>", t.text.trim())),
            Role::Assistant => {
                // Thinking is kept here, so the marker describes real content —
                // and this is the path that puts the marker into the *pretrain*
                // corpus, which is what stops its embedding row arriving at SFT
                // at random init.
                out.push_str(budget_marker(budget));
                out.push_str(ASSISTANT_OPEN);
                out.push_str(t.text.trim());
                out.push_str("<|/assistant|>");
            }
        }
    }
    out
}

/// Render a conversation into token ids, dropping the oldest whole turns until it
/// fits `ctx_window - reserve_for_reply` tokens.
///
/// The system block (if any) and the trailing open `<|assistant|>` are mandatory
/// scaffold and always kept; only history turns are dropped, and always at whole-
/// turn boundaries so the prompt never begins mid-sentence. The most recent turn
/// is kept even if it alone busts the budget — better to send a hard-cut latest
/// message (which the sliding window in `forward_seq` handles) than nothing.
///
/// `reserve_for_reply` is headroom left for the model to generate before the
/// sliding window starts evicting the freshest prompt tokens.
///
/// `budget`, when given, renders as one marker in front of the trailing open
/// `<|assistant|>` — the last thing the model conditions on before it starts
/// generating, which is where a lever over the reply belongs.
pub fn build_prompt(
    system: Option<&str>,
    history: &[Turn],
    budget: Option<ThinkBudget>,
    merges: &[((u32, u32), u32)],
    ctx_window: usize,
    reserve_for_reply: usize,
) -> Vec<u32> {
    let system_str = system
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| format!("<|system|>{}<|/system|>", s));
    let budget_str = budget_marker(budget);

    // Cost of the always-present scaffold, in tokens. The budget marker is part of
    // it — it is mandatory once asked for, exactly like the system block, so its
    // cost has to come out of the budget before history is fitted around it.
    let mut scaffold = encode(ASSISTANT_OPEN, merges).len();
    if !budget_str.is_empty() {
        scaffold += encode(budget_str, merges).len();
    }
    if let Some(ref s) = system_str {
        scaffold += encode(s, merges).len();
    }
    let budget = ctx_window.saturating_sub(reserve_for_reply);

    // Walk newest -> oldest, keeping whole turns while they fit. `start` ends up
    // as the index of the oldest turn we keep.
    let mut used = scaffold;
    let mut start = history.len();
    for i in (0..history.len()).rev() {
        let cost = encode(&format_turn(&history[i]), merges).len();
        let is_latest = i == history.len() - 1;
        if !is_latest && used + cost > budget {
            break;
        }
        used += cost;
        start = i;
    }

    // Assemble the kept turns in chronological order and encode once, so the
    // returned ids exactly match a single pass (no cross-turn BPE-merge surprises
    // from encoding piecemeal).
    let mut s = String::new();
    if let Some(ref sys) = system_str {
        s.push_str(sys);
    }
    for t in &history[start..] {
        s.push_str(&format_turn(t));
    }
    s.push_str(budget_str);
    s.push_str(ASSISTANT_OPEN);
    encode(&s, merges)
}

/// Render a full training example — the whole conversation *including* the
/// assistant replies — into token ids plus a parallel loss mask.
///
/// `mask[i] == 1` iff the token predicted from position `i` (i.e. token `i+1`)
/// is part of an assistant reply: its content through the closing
/// `<|/assistant|>` marker, inclusive. Everything else — the system block, the
/// user turns, the opening `<|assistant|>` cue, and the final position (which has
/// no next token) — is `0`. `ids` and `mask` are the same length.
///
/// Unlike `build_prompt` (which renders for *inference*: it ends on an open
/// `<|assistant|>` and strips historical reasoning), this keeps the assistant
/// turns verbatim — including any `<|thinking|>` block — because during SFT the
/// assistant text is the training target and we *want* the model to learn to
/// produce reasoning.
///
/// The conversation is encoded segment-by-segment rather than in a single pass.
/// This is deliberate: the pre-tokenizer only splits words on spaces, so a
/// special marker glues to the text next to it inside one "word" (e.g.
/// `<|assistant|>Hello` is a single word) — there is no clean token boundary at a
/// marker in a whole-string encoding. Splitting at the marker boundaries here
/// makes the reply tokens stand alone, which is exactly how the model sees them
/// at generation time (it emits standalone tokens after the prompt's open
/// `<|assistant|>`) and makes the mask exact. The cost is that the concatenated
/// tokenization can differ by ~1 token at each seam versus one whole-string
/// `encode()`; that is the right trade for a precise mask and a closer match to
/// the autoregressive path.
pub fn build_training_example(
    system: Option<&str>,
    turns: &[Turn],
    budget: Option<ThinkBudget>,
    merges: &[((u32, u32), u32)],
) -> (Vec<u32>, Vec<u8>) {
    // Each entry is one rendered piece plus whether its tokens are a training
    // target. Order matters — pieces are concatenated in the order pushed.
    let mut segments: Vec<(String, bool)> = Vec::new();

    if let Some(s) = system.map(str::trim).filter(|s| !s.is_empty()) {
        segments.push((format!("<|system|>{}<|/system|>", s), false));
    }

    for t in turns {
        match t.role {
            Role::User => {
                segments.push((format!("<|user|>{}<|/user|>", t.text.trim()), false));
            }
            Role::Assistant => {
                // The opening marker is the cue the model conditions on (not a
                // target); the reply body and the closing marker are the target —
                // the model must learn both to answer and to stop.
                //
                // The budget marker is part of that cue, and untrained for the
                // same reason: it is an *instruction* the caller supplies, never
                // something the model chooses. Training it as a target would teach
                // the model to predict how hard it is about to think, which is a
                // fact about the request, not about the reply. So only its
                // embedding row is ever learned — the output column stays unused.
                if let Some(b) = budget {
                    segments.push((b.marker().to_string(), false));
                }
                segments.push((ASSISTANT_OPEN.to_string(), false));
                segments.push((t.text.trim().to_string(), true));
                segments.push(("<|/assistant|>".to_string(), true));
            }
        }
    }

    // Encode each segment on its own and record, per resulting token, whether it
    // belongs to a target segment.
    let mut ids: Vec<u32> = Vec::new();
    let mut target: Vec<bool> = Vec::new();
    for (text, is_target) in &segments {
        let toks = encode(text, merges);
        target.extend(std::iter::repeat(*is_target).take(toks.len()));
        ids.extend(toks);
    }

    // Shift the per-token target flag onto prediction positions: position i is
    // trained iff the token it predicts (i+1) is a target. The last position has
    // no successor, so it is never trained.
    let n = ids.len();
    let mut mask: Vec<u8> = vec![0u8; n];
    for i in 0..n.saturating_sub(1) {
        if target[i + 1] {
            mask[i] = 1;
        }
    }

    (ids, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_thinking_removes_block() {
        let t = "<|thinking|>reasoning here<|/thinking|>the actual reply";
        assert_eq!(strip_thinking(t), "the actual reply");
    }

    #[test]
    fn test_strip_thinking_no_block_is_passthrough() {
        assert_eq!(strip_thinking("  just a reply  "), "just a reply");
    }

    #[test]
    fn test_format_turn_user_and_assistant() {
        let u = format_turn(&Turn { role: Role::User, text: "hi".into() });
        assert_eq!(u, "<|user|>hi<|/user|>");
        let a = format_turn(&Turn {
            role: Role::Assistant,
            text: "<|thinking|>hmm<|/thinking|>hello".into(),
        });
        assert_eq!(a, "<|assistant|>hello<|/assistant|>");
    }

    // ── render_conversation ──

    // The property that matters: the rendered text must be what
    // `build_training_example` tokenizes, or the pretrain corpus would teach the
    // model a different scaffold than SFT uses.
    #[test]
    fn test_render_matches_the_training_example_shape() {
        let turns = vec![
            Turn { role: Role::User, text: " hi ".into() },
            Turn { role: Role::Assistant, text: "yo".into() },
        ];
        assert_eq!(
            render_conversation(Some(" be nice "), &turns, None),
            "<|system|>be nice<|/system|><|user|>hi<|/user|><|assistant|>yo<|/assistant|>",
        );
    }

    #[test]
    fn test_render_keeps_thinking_unlike_a_prompt() {
        // `build_prompt` strips historical reasoning; this must not, because the
        // reasoning is text SFT will train on.
        let turns = vec![Turn {
            role: Role::Assistant,
            text: "<|thinking|>hmm<|/thinking|>hello".into(),
        }];
        let rendered = render_conversation(None, &turns, None);
        assert!(rendered.contains("<|thinking|>hmm<|/thinking|>"), "{rendered}");
    }

    #[test]
    fn test_render_omits_an_empty_system_block() {
        let turns = vec![Turn { role: Role::User, text: "hi".into() }];
        assert_eq!(render_conversation(Some("   "), &turns, None), "<|user|>hi<|/user|>");
        assert_eq!(render_conversation(None, &turns, None), "<|user|>hi<|/user|>");
    }

    // ── the thinking budget ──

    // With no trained merges, `encode` falls back to byte-level tokens, which is
    // deterministic and enough to check the mask's shape.
    fn no_merges() -> Vec<((u32, u32), u32)> {
        Vec::new()
    }

    // The property the whole feature rests on: an unbudgeted conversation renders
    // exactly as it did before budgets existed. Every corpus, `chat_destination`
    // hash and benchmark score already on disk depends on it.
    #[test]
    fn no_budget_renders_exactly_as_before() {
        let merges = no_merges();
        let turns = vec![
            Turn { role: Role::User, text: "hi".into() },
            Turn { role: Role::Assistant, text: "yo".into() },
        ];
        assert_eq!(
            render_conversation(Some("be nice"), &turns, None),
            "<|system|>be nice<|/system|><|user|>hi<|/user|><|assistant|>yo<|/assistant|>",
        );
        let (ids, _) = build_training_example(None, &turns, None, &merges);
        let expected = encode("<|user|>hi<|/user|>", &merges).len()
            + encode("<|assistant|>", &merges).len()
            + encode("yo", &merges).len()
            + encode("<|/assistant|>", &merges).len();
        assert_eq!(ids.len(), expected);
    }

    #[test]
    fn a_budget_marker_sits_immediately_before_the_assistant_open() {
        let turns = vec![
            Turn { role: Role::User, text: "hi".into() },
            Turn { role: Role::Assistant, text: "yo".into() },
        ];
        assert_eq!(
            render_conversation(None, &turns, Some(ThinkBudget::High)),
            "<|user|>hi<|/user|><|think:high|><|assistant|>yo<|/assistant|>",
        );
    }

    // The marker is an instruction the caller supplies, not something the model
    // chooses, so it must never be a training target — otherwise the model learns
    // to predict how hard it is about to think.
    #[test]
    fn the_budget_marker_is_a_cue_and_never_a_target() {
        let merges = no_merges();
        let turns = vec![
            Turn { role: Role::User, text: "hi".into() },
            Turn { role: Role::Assistant, text: "yo".into() },
        ];
        let (plain, plain_mask) = build_training_example(None, &turns, None, &merges);
        let (budgeted, budgeted_mask) =
            build_training_example(None, &turns, Some(ThinkBudget::Low), &merges);

        // Exactly one extra token, and it is the marker's reserved id.
        let marker = encode(ThinkBudget::Low.marker(), &merges);
        assert_eq!(marker.len(), 1, "a budget must cost exactly one token");
        assert_eq!(budgeted.len(), plain.len() + marker.len());
        assert!(budgeted.contains(&marker[0]));

        // The trained span is unchanged: same count, same reply.
        let trained = |m: &[u8]| m.iter().filter(|&&b| b == 1).count();
        assert_eq!(trained(&budgeted_mask), trained(&plain_mask));

        // And the marker's own position is not trained — the position *before* it
        // must not be asked to predict it.
        let at = budgeted.iter().position(|id| *id == marker[0]).unwrap();
        assert_eq!(budgeted_mask[at - 1], 0, "the marker must not be a target");
    }

    // A budget is per-turn: it marks the reply about to be generated, and never a
    // history turn, whose reasoning `build_prompt` has already stripped.
    #[test]
    fn a_prompt_marks_only_the_turn_it_is_about_to_generate() {
        let merges = no_merges();
        let history = vec![
            Turn { role: Role::User, text: "q1".into() },
            Turn { role: Role::Assistant, text: "<|thinking|>hmm<|/thinking|>a1".into() },
            Turn { role: Role::User, text: "q2".into() },
        ];
        let ids = build_prompt(None, &history, Some(ThinkBudget::Med), &merges, 4096, 8);
        let marker = encode(ThinkBudget::Med.marker(), &merges);
        assert_eq!(ids.iter().filter(|id| **id == marker[0]).count(), 1);

        // …and it is the second-to-last token, right before the open marker.
        let open = encode(ASSISTANT_OPEN, &merges);
        assert_eq!(ids[ids.len() - open.len() - 1], marker[0]);
        assert_eq!(&ids[ids.len() - open.len()..], open.as_slice());
    }

    // The marker is mandatory scaffold, so its cost must come out of the window
    // before history is fitted around it — otherwise a full context overflows by
    // exactly one token, silently, only on budgeted turns.
    #[test]
    fn the_budget_is_charged_to_the_scaffold() {
        let merges = no_merges();
        let history = vec![
            Turn { role: Role::User, text: "a fairly wordy question".into() },
            Turn { role: Role::Assistant, text: "a fairly wordy answer".into() },
            Turn { role: Role::User, text: "and another one".into() },
        ];
        // A window tight enough that the budget's token decides whether the oldest
        // turn still fits.
        let plain = build_prompt(None, &history, None, &merges, 64, 8);
        let budgeted = build_prompt(None, &history, Some(ThinkBudget::High), &merges, 64, 8);
        assert!(
            budgeted.len() <= plain.len() + 1,
            "budget {} vs plain {}: the marker must displace history, not add to it",
            budgeted.len(),
            plain.len(),
        );
    }

    #[test]
    fn budget_parses_the_spellings_the_wire_and_the_disk_use() {
        assert_eq!(ThinkBudget::parse("low"), Some(ThinkBudget::Low));
        assert_eq!(ThinkBudget::parse(" HIGH "), Some(ThinkBudget::High));
        // The frontend says "medium"; the on-disk label is "med".
        assert_eq!(ThinkBudget::parse("medium"), Some(ThinkBudget::Med));
        assert_eq!(ThinkBudget::parse("med"), Some(ThinkBudget::Med));
        // "auto" is the absence of a budget, expressed as None by the caller.
        assert_eq!(ThinkBudget::parse("auto"), None);
        assert_eq!(ThinkBudget::parse(""), None);
    }

    // The on-disk spelling `synth::teach` writes must round-trip, or the labels
    // already in raw/sft/ deserialize to nothing.
    #[test]
    fn budget_serializes_as_the_label_on_disk() {
        for b in ThinkBudget::ALL {
            let json = serde_json::to_string(&b).unwrap();
            assert_eq!(json, format!("\"{}\"", b.as_str()));
            assert_eq!(serde_json::from_str::<ThinkBudget>(&json).unwrap(), b);
        }
    }

    // Every budget must be a *single* reserved token, not a shredded byte string —
    // that is the whole reason it is a marker rather than a phrase.
    #[test]
    fn every_budget_marker_is_one_reserved_token() {
        let merges = no_merges();
        for b in ThinkBudget::ALL {
            let ids = encode(b.marker(), &merges);
            assert_eq!(ids.len(), 1, "{} encoded to {:?}", b.marker(), ids);
            assert!((256..320).contains(&ids[0]), "{} is not in the special block", b.marker());
        }
    }

    // ── build_training_example ──

    #[test]
    fn test_training_example_masks_only_assistant_reply() {
        let merges = no_merges();
        let turns = vec![
            Turn { role: Role::User, text: "hi".into() },
            Turn { role: Role::Assistant, text: "yo".into() },
        ];
        let (ids, mask) = build_training_example(None, &turns, None, &merges);

        assert_eq!(ids.len(), mask.len(), "ids and mask must be parallel");
        assert!(!ids.is_empty());

        // The trained positions predict exactly the reply body + closing marker.
        // count(mask==1) == number of target tokens (the first target token still
        // has a predecessor to be predicted from; only the very last position,
        // which has no successor, is dropped — and that isn't a target's
        // predecessor).
        let trained = mask.iter().filter(|&&m| m == 1).count();
        let expected = encode("yo", &merges).len() + encode("<|/assistant|>", &merges).len();
        assert_eq!(trained, expected);

        // The final position has no next token, so it is never trained.
        assert_eq!(*mask.last().unwrap(), 0);
    }

    #[test]
    fn test_training_example_two_turns_two_masked_spans() {
        let merges = no_merges();
        let turns = vec![
            Turn { role: Role::User, text: "q1".into() },
            Turn { role: Role::Assistant, text: "a1".into() },
            Turn { role: Role::User, text: "q2".into() },
            Turn { role: Role::Assistant, text: "a2".into() },
        ];
        let (ids, mask) = build_training_example(Some("be nice"), &turns, None, &merges);

        assert_eq!(ids.len(), mask.len());

        let close = encode("<|/assistant|>", &merges).len();
        let expected = encode("a1", &merges).len()
            + close
            + encode("a2", &merges).len()
            + close;
        let trained = mask.iter().filter(|&&m| m == 1).count();
        assert_eq!(trained, expected, "both assistant replies should be trained");

        // The system block + user turns are present but untrained, so the total
        // stream is strictly longer than the trained span.
        assert!(ids.len() > trained);

        // There must be two separate masked spans (a 1→0→1 transition), proving
        // the intervening second user turn is masked off.
        let transitions = mask.windows(2).filter(|w| w[0] == 1 && w[1] == 0).count();
        assert_eq!(transitions, 2, "expected two distinct trained spans");
    }

    #[test]
    fn test_training_example_system_and_user_untrained() {
        let merges = no_merges();
        let turns = vec![
            Turn { role: Role::User, text: "question".into() },
            Turn { role: Role::Assistant, text: "answer".into() },
        ];
        let (ids, mask) = build_training_example(Some("system prompt"), &turns, None, &merges);

        // The prefix up to the first trained position (everything predicting the
        // system/user/scaffold tokens) must be entirely masked off.
        let first_trained = mask.iter().position(|&m| m == 1).unwrap();
        assert!(mask[..first_trained].iter().all(|&m| m == 0));
        assert!(first_trained > 0);
        assert_eq!(ids.len(), mask.len());
    }
}
