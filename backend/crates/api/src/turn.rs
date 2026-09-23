//! One assistant turn, including the tool rounds it takes.
//!
//! The models here are too small to stop at a tool call. A large model emits one
//! and halts, waiting to be handed the answer; ours emits one and carries on
//! saying whatever it was going to say. So a turn is not generate-pause-resume.
//! It is: let the reply finish, run whatever it asked for while it does, then
//! ask again with the answers in the conversation.
//!
//! That is why a tool is spawned the moment its call closes and only joined at
//! the end of the reply — the search overlaps the tokens that follow it instead
//! of stalling them, which for a ten-second search is the whole latency budget.

use futures_util::{Stream, StreamExt};
use llm_engine::{Frame, Proxy};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tools::Registry;

use crate::ws;

/// How many times a reply may be handed tool output and asked to continue.
///
/// Default six. It was two — the reply that calls and the one that answers with
/// what came back — which assumed the second round is prose. A model doing real
/// work does not oblige: it looks at the output, learns the path was wrong, and
/// calls again. At two rounds that second call is parsed, spawned, and then
/// dropped on the floor when the loop runs out, leaving the reply ending in a
/// call nothing ever answered. Six lets a model actually converge on something.
///
/// Still bounded, and the reason is unchanged: without one, a model that has
/// learned to open a search in every reply never terminates, and each round is a
/// fresh generation the person waits on. Raise `MAX_TOOL_ROUNDS` to let a model
/// work for longer, at that cost.
const DEFAULT_TOOL_ROUNDS: usize = 6;

fn max_tool_rounds() -> usize {
    // At least one, or a turn would generate nothing at all.
    std::env::var("MAX_TOOL_ROUNDS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_TOOL_ROUNDS)
}

/// One `tool_call` the LLM engine reported, and the work started for it.
struct Pending {
    call: Value,
    work: JoinHandle<(String, Option<tools::ToolError>)>,
}

/// Drive a whole turn to its end, relaying everything to `session` as it happens.
///
/// `interjections` carries a person's mid-turn messages, sent over the same
/// socket while this round generates — see `routes::stream::read_interjections`,
/// which owns the client half of the socket for the turn's duration and is the
/// only writer. Buffered rather than applied as they arrive: the only safe
/// splice point is between a round's tool result and the next round's request,
/// which is also the only window in which a person's message can still change
/// what the model is about to be asked — see `drain_interjection`.
pub async fn run(
    session: &mut actix_ws::Session,
    engine: &Proxy,
    registry: &Registry,
    caller: &auth::Caller,
    path: &str,
    // The chat this turn belongs to, handed to every tool as its session key.
    // The sandboxed shell keeps a filesystem under it, so a file written in one
    // round is still there in the next; `tools` never learns what the key means.
    chat: &str,
    mut ask: Value,
    mut interjections: mpsc::UnboundedReceiver<String>,
) {
    let settings = ask.clone();

    let rounds = max_tool_rounds();
    for round in 0..rounds {
        let frames = match engine.stream(caller, path, ask).await {
            Ok(frames) => frames,
            Err(e) => {
                let _ = session.text(ws::error_frame(&e.to_string())).await;
                return;
            }
        };

        let last_round = round + 1 == rounds;
        let Some(pending) = relay(session, registry, chat, frames, last_round).await else {
            return;
        };
        if pending.is_empty() {
            return;
        }

        let Some(mut next) = answer(session, pending).await else {
            return;
        };
        carry_settings(&settings, &mut next);
        if let Some(text) = fold_interjection(&mut next, drain_interjection(&mut interjections)) {
            let confirmed = json!({ "text": text });
            if session
                .text(envelope("interjected", &confirmed).to_string())
                .await
                .is_err()
            {
                return;
            }
        }
        ask = next;
    }
}

/// Every message buffered since the last time this was called, joined into one
/// string — or `None` if nothing arrived. `try_recv` rather than awaiting: this
/// is called once per round, right after the round's tool work has been
/// joined, and a round with nothing queued must not block waiting for one.
fn drain_interjection(interjections: &mut mpsc::UnboundedReceiver<String>) -> Option<String> {
    let mut collected = Vec::new();
    while let Ok(text) = interjections.try_recv() {
        collected.push(text);
    }
    (!collected.is_empty()).then(|| collected.join("\n\n"))
}

/// Splice a buffered interjection into the next round's request, as the
/// LLM engine's `interject` field (`llm-engine/api/routes/chats.py::Ask`). Returns the
/// text when it folded one in, so the caller knows to confirm it to the
/// client — `None` leaves `next` untouched, which is a round with nothing
/// queued behaving exactly as it did before this existed.
fn fold_interjection(next: &mut Value, text: Option<String>) -> Option<String> {
    let text = text?;
    if let Some(object) = next.as_object_mut() {
        object.insert("interject".to_string(), json!(text));
    }
    Some(text)
}

/// Forward this round's frames, starting a tool the moment its call closes.
///
/// None when the browser hung up. An empty list means the reply asked for
/// nothing, which is the end of the turn.
///
/// The LLM engine ends every round with `done`, but only the last one ends the
/// *turn* — so each is stamped with whether another round follows. Without it
/// the client cannot tell the two apart, and closing on the first would cut the
/// turn off before the answer it went to fetch.
async fn relay(
    session: &mut actix_ws::Session,
    registry: &Registry,
    chat: &str,
    frames: impl Stream<Item = llm_engine::Result<Frame>>,
    last_round: bool,
) -> Option<Vec<Pending>> {
    futures_util::pin_mut!(frames);

    let mut pending = Vec::new();
    while let Some(frame) = frames.next().await {
        let message = match frame {
            Ok(frame) => {
                if frame.kind == "tool_call" {
                    pending.push(start(registry, chat, frame.data.clone()));
                }
                // Safe to decide here: a `tool_call` always precedes the `done`
                // of the round that made it.
                match frame.kind.as_str() {
                    "done" => stamped(&frame, last_round || pending.is_empty()),
                    _ => flatten(&frame),
                }
            }
            Err(e) => ws::error_frame(&e.to_string()),
        };
        if session.text(message).await.is_err() {
            // The browser hung up. Nothing downstream has anywhere to go, and
            // the searches already running will be dropped with their handles.
            return None;
        }
    }
    Some(pending)
}

fn start(registry: &Registry, chat: &str, call: Value) -> Pending {
    let registry = registry.clone();
    let name = string_at(&call, "tool");
    let argument = string_at(&call, "argument");
    let chat = chat.to_string();
    Pending {
        call,
        work: actix_web::rt::spawn(async move { registry.answer(&name, &argument, &chat).await }),
    }
}

/// Await every tool this round started and build the request that continues the
/// turn. None when the browser hung up.
///
/// Only the first result continues the turn: the LLM engine takes one `tool_result`
/// per request because each is a message of its own in the conversation, and
/// answering two calls at once would mean deciding which the next reply follows.
/// The rest are still reported, so nothing a person watched start disappears.
async fn answer(session: &mut actix_ws::Session, pending: Vec<Pending>) -> Option<Value> {
    let mut first: Option<Value> = None;

    for Pending { call, work } in pending {
        // A panicked task is reported as a failed tool rather than taken down
        // with the turn: the reply it belongs to is already generated.
        let (body, failure) = work.await.unwrap_or_else(|e| {
            (
                tools::markup::warning("The tool did not finish."),
                Some(panicked(e)),
            )
        });
        if let Some(e) = failure {
            eprintln!("tool {} failed: {e}", string_at(&call, "tool"));
        }

        let result = json!({
            "tool": string_at(&call, "tool"),
            "argument": string_at(&call, "argument"),
            "body": body,
        });
        if session
            .text(envelope("tool_result", &result).to_string())
            .await
            .is_err()
        {
            return None;
        }
        first.get_or_insert(result);
    }

    Some(json!({ "tool_result": first? }))
}

fn panicked(e: tokio::task::JoinError) -> tools::ToolError {
    tools::ToolError::Transport(e.to_string())
}

/// A payload flattened onto the `type` field the client switches on.
pub fn envelope(kind: &str, data: &Value) -> Value {
    let mut payload = json!({ "type": kind });
    if let (Some(target), Some(source)) = (payload.as_object_mut(), data.as_object()) {
        for (key, value) in source {
            target.insert(key.clone(), value.clone());
        }
    }
    payload
}

fn flatten(frame: &Frame) -> String {
    envelope(&frame.kind, &frame.data).to_string()
}

/// A `done` frame, saying whether it ends the turn or only this round.
fn stamped(frame: &Frame, is_final: bool) -> String {
    let mut payload = envelope(&frame.kind, &frame.data);
    // Set after the merge, so a field the LLM engine happens to add cannot shadow it.
    payload["final"] = json!(is_final);
    payload.to_string()
}

pub fn string_at(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// The model and sampling settings carry across a round; the text does not.
///
/// The LLM engine takes exactly one of `text` and `tool_result`, so the question
/// that opened the turn must be dropped from the follow-up — it is already
/// stored as a message and comes back as history.
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
            into.insert(key.to_string(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_follow_up_keeps_the_model_and_the_sampling_but_drops_the_question() {
        let ask = json!({
            "model_name": "tiny", "text": "how tall?", "temperature": 0.4, "top_k": 20,
        });
        let mut next = json!({ "tool_result": { "tool": "web-search" } });
        carry_settings(&ask, &mut next);

        assert_eq!(next["model_name"], "tiny");
        assert_eq!(next["temperature"], 0.4);
        assert_eq!(next["top_k"], 20);
        assert!(next.get("text").is_none());
    }

    #[test]
    fn a_follow_up_keeps_the_context_extension_settings_too() {
        // Both are properties of the standing conversation, exactly like
        // temperature — a turn that happens to call a tool should not silently
        // drop back to an un-extended window for its follow-up round.
        let ask = json!({
            "model_name": "tiny", "text": "how tall?",
            "context_multiplier": 4.0, "history_floor": "11111111-1111-1111-1111-111111111111",
        });
        let mut next = json!({ "tool_result": { "tool": "web-search" } });
        carry_settings(&ask, &mut next);

        assert_eq!(next["context_multiplier"], 4.0);
        assert_eq!(
            next["history_floor"],
            "11111111-1111-1111-1111-111111111111"
        );
    }

    #[test]
    fn a_follow_up_carries_no_attachments() {
        // They belong to the question, and the LLM engine stored them with it. A
        // second round that repeated them would try to bind rows already sent
        // and fail the turn — see `_bind` in the LLM engine's chats route.
        let ask = json!({ "model_name": "tiny", "text": "what is this", "attachment_ids": ["a"] });
        let mut next = json!({ "tool_result": { "tool": "web-search" } });
        carry_settings(&ask, &mut next);

        assert!(next.get("attachment_ids").is_none());
    }

    #[test]
    fn a_frame_the_client_has_never_seen_still_arrives_with_its_fields() {
        let frame = Frame {
            kind: "tool_call".to_string(),
            data: json!({ "tool": "web-search", "argument": "elbrus" }),
        };
        let flattened: Value = serde_json::from_str(&flatten(&frame)).expect("valid json");

        assert_eq!(flattened["type"], "tool_call");
        assert_eq!(flattened["argument"], "elbrus");
    }

    #[test]
    fn a_call_missing_its_fields_is_read_as_empty_rather_than_panicking() {
        assert_eq!(string_at(&json!({}), "tool"), "");
        assert_eq!(string_at(&json!({ "tool": 7 }), "tool"), "");
    }

    #[test]
    fn a_round_with_nothing_queued_folds_in_nothing() {
        let mut next = json!({ "tool_result": { "tool": "web-search" } });
        assert_eq!(fold_interjection(&mut next, None), None);
        assert!(next.get("interject").is_none());
    }

    #[test]
    fn a_queued_message_is_spliced_in_as_the_interject_field() {
        let mut next = json!({ "tool_result": { "tool": "web-search" } });
        let confirmed = fold_interjection(&mut next, Some("wait, also check the date".into()));
        assert_eq!(confirmed.as_deref(), Some("wait, also check the date"));
        assert_eq!(next["interject"], "wait, also check the date");
    }

    #[tokio::test]
    async fn draining_joins_everything_sent_before_the_round_finished() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        tx.send("first".to_string()).unwrap();
        tx.send("second".to_string()).unwrap();

        assert_eq!(
            drain_interjection(&mut rx),
            Some("first\n\nsecond".to_string())
        );
        // Drained once, so a round with nothing new sent since sees nothing.
        assert_eq!(drain_interjection(&mut rx), None);
    }
}
