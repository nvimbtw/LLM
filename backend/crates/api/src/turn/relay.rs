//! Persist completed rounds before forwarding their final acknowledgement.

use futures_util::{Stream, StreamExt};
use llm_engine::Frame;
use serde_json::json;

use super::{tools::Pending, ChatTurn};
use crate::{
    conversation,
    streaming::{envelope, ToolCall},
    ws,
};

/// Relay one inference round, committing its assistant message before `done` is delivered.
/// Tool calls start during generation. Errors, truncation or disconnect drop their owned tasks.
pub(super) async fn relay(
    session: &mut actix_ws::Session,
    turn: &ChatTurn<'_>,
    frames: impl Stream<Item = llm_engine::Result<Frame>>,
    last_round: bool,
    prepared: &conversation::Round,
) -> Option<Vec<Pending>> {
    futures_util::pin_mut!(frames);
    let mut pending = Vec::new();
    while let Some(frame) = frames.next().await {
        let mut frame = accepted_frame(session, frame).await?;
        let completed = frame.kind == "done";
        if completed {
            frame.data = match conversation::finish(
                turn.pool,
                turn.owner,
                turn.chat_id,
                prepared,
                &frame.data,
            )
            .await
            {
                Ok(reply) => reply,
                Err(error) => {
                    let _ = session.text(ws::error_frame(&error.to_string())).await;
                    return None;
                }
            };
        }
        if frame.kind == "tool_call" {
            pending.push(Pending::start(
                turn.registry,
                &turn.tool_session,
                ToolCall::from_value(&frame.data),
            ));
        }
        session
            .text(client_frame(&frame, last_round || pending.is_empty()))
            .await
            .ok()?;
        if completed {
            return Some(pending);
        }
    }
    let _ = session
        .text(ws::error_frame("inference ended before completion"))
        .await;
    None
}

/// Accept a non-error service frame or report a terminal service/transport error before stopping the round.
async fn accepted_frame(
    session: &mut actix_ws::Session,
    frame: llm_engine::Result<Frame>,
) -> Option<Frame> {
    match frame {
        Ok(frame) if frame.kind != "error" => Some(frame),
        Ok(frame) => {
            let _ = session
                .text(envelope(&frame.kind, &frame.data).to_string())
                .await;
            None
        }
        Err(error) => {
            let _ = session.text(ws::error_frame(&error.to_string())).await;
            None
        }
    }
}

/// Flatten a frame and stamp `done` after merging so service fields cannot override turn finality.
fn client_frame(frame: &Frame, is_final: bool) -> String {
    let mut payload = envelope(&frame.kind, &frame.data);
    if frame.kind == "done" {
        payload["final"] = json!(is_final);
    }
    payload.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finality belongs to Rust while unknown service fields remain opaque and available to clients.
    #[test]
    fn done_finality_overrides_the_upstream_field() {
        let frame = Frame {
            kind: "done".into(),
            data: json!({"final":true, "extension":42}),
        };
        let payload: serde_json::Value =
            serde_json::from_str(&client_frame(&frame, false)).unwrap();
        assert_eq!(payload["final"], false);
        assert_eq!(payload["extension"], 42);
    }
}
