//! Follow-up request policy and buffered text steering.

use serde_json::{json, Value};
use tokio::sync::mpsc;

const DEFAULT_TOOL_ROUNDS: usize = 6;

/// Read a positive tool-round ceiling from the environment, falling back to six rounds.
pub(super) fn max_tool_rounds() -> usize {
    std::env::var("MAX_TOOL_ROUNDS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|rounds| *rounds > 0)
        .unwrap_or(DEFAULT_TOOL_ROUNDS)
}

/// Collect currently buffered interjections without waiting; preserve their order and blank-line separation.
pub(super) fn drain_interjection(receiver: &mut mpsc::UnboundedReceiver<String>) -> Option<String> {
    let mut collected = Vec::new();
    while let Ok(text) = receiver.try_recv() {
        collected.push(text);
    }
    (!collected.is_empty()).then(|| collected.join("\n\n"))
}

/// Insert queued text into an object request and return the text to acknowledge to the client.
/// A missing interjection leaves the next request untouched.
pub(super) fn fold_interjection(next: &mut Value, text: Option<String>) -> Option<String> {
    let text = text?;
    if let Some(object) = next.as_object_mut() {
        object.insert("interject".into(), json!(text));
    }
    Some(text)
}

/// Copy standing model, sampling and context settings into a follow-up request.
/// The original question, attachments and request ID are deliberately not repeated.
pub fn carry_settings(ask: &Value, next: &mut Value) {
    const CARRIED: [&str; 9] = [
        "model_name",
        "system",
        "budget",
        "max_tokens",
        "temperature",
        "top_k",
        "top_p",
        "context_multiplier",
        "history_floor",
    ];
    let (Some(from), Some(into)) = (ask.as_object(), next.as_object_mut()) else {
        return;
    };
    for key in CARRIED {
        if let Some(value) = from.get(key) {
            into.insert(key.into(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Follow-up requests retain every standing setting while excluding initial-message-only fields.
    #[test]
    fn follow_up_keeps_settings_without_repeating_the_question() {
        let ask = json!({"model_name":"tiny", "system":"help", "budget":3, "max_tokens":5,
            "temperature":0.4, "top_k":20, "top_p":0.8, "context_multiplier":4.0,
            "history_floor":"floor", "text":"question", "attachment_ids":["a"], "request_id":"r"});
        let mut next = json!({"tool_result":{"tool":"web-search"}});
        carry_settings(&ask, &mut next);
        for key in [
            "model_name",
            "system",
            "budget",
            "max_tokens",
            "temperature",
            "top_k",
            "top_p",
            "context_multiplier",
            "history_floor",
        ] {
            assert_eq!(next[key], ask[key], "{key}");
        }
        for key in ["text", "attachment_ids", "request_id"] {
            assert!(next.get(key).is_none(), "{key}");
        }
    }

    /// No queued text leaves the payload unchanged; queued text is returned for acknowledgement.
    #[test]
    fn interjection_is_folded_only_when_present() {
        let mut next = json!({"tool_result":{}});
        assert_eq!(fold_interjection(&mut next, None), None);
        assert!(next.get("interject").is_none());
        assert_eq!(
            fold_interjection(&mut next, Some("wait".into())),
            Some("wait".into())
        );
        assert_eq!(next["interject"], "wait");
    }

    /// A drain joins buffered messages once and does not block when the queue is empty or closed.
    #[tokio::test]
    async fn drain_preserves_order_without_waiting() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        sender.send("first".into()).unwrap();
        sender.send("second".into()).unwrap();
        drop(sender);
        assert_eq!(
            drain_interjection(&mut receiver),
            Some("first\n\nsecond".into())
        );
        assert_eq!(drain_interjection(&mut receiver), None);
    }
}
