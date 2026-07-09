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
//! Historical assistant turns keep only their reply: any
//! `<|thinking|>...<|/thinking|>` block is stripped, since stale reasoning isn't
//! worth spending the (tiny) context window on. The model still produces a fresh
//! thinking block for the current turn.

use tokenize::bpe::encode;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    User,
    Assistant,
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

fn format_turn(t: &Turn) -> String {
    match t.role {
        Role::User => format!("<|user|>{}<|/user|>", t.text.trim()),
        Role::Assistant => format!("<|assistant|>{}<|/assistant|>", strip_thinking(&t.text)),
    }
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
pub fn build_prompt(
    system: Option<&str>,
    history: &[Turn],
    merges: &[((u32, u32), u32)],
    ctx_window: usize,
    reserve_for_reply: usize,
) -> Vec<u32> {
    let system_str = system
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| format!("<|system|>{}<|/system|>", s));

    // Cost of the always-present scaffold, in tokens.
    let mut scaffold = encode(ASSISTANT_OPEN, merges).len();
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
    s.push_str(ASSISTANT_OPEN);
    encode(&s, merges)
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
}
