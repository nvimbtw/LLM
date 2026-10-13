//! Generation and sequential shell effects for one gym episode.

use super::{protocol::Episode, steps};
use crate::{
    streaming::{envelope, string_at, ToolCall},
    ws,
};
use futures_util::{Stream, StreamExt};
use llm_engine::Frame;
use serde_json::{json, Value};

pub(super) struct Driver<'a> {
    pub engine: &'a llm_engine::Proxy,
    pub gyms: &'a gyms_engine::Proxy,
    pub caller: &'a auth::Caller,
    pub model: &'a str,
    pub ask: &'a Value,
    pub observation_budget_chars: usize,
}

struct Reply {
    text: String,
    calls: Vec<ToolCall>,
}

impl Driver<'_> {
    /// Generate bounded rounds and grade the terminal reply; return `None` when client delivery fails.
    /// Legacy generation-start failures report an error and leave the episode ungraded.
    pub async fn episode(&self, session: &mut actix_ws::Session, episode: &Episode) -> Option<()> {
        let mut turns = vec![json!({"role":"user","text":episode.prompt})];
        let rounds = episode.rounds();
        for round in 0..rounds {
            let terminal = round + 1 == rounds;
            let body = episode.generation(self.model, &turns, self.ask, terminal);
            let frames = match self.engine.stream(self.caller, "/chat/stream", body).await {
                Ok(frames) => frames,
                Err(error) => {
                    session
                        .text(ws::error_frame(&error.to_string()))
                        .await
                        .ok()?;
                    return Some(());
                }
            };
            let reply = relay(session, frames, &episode.id).await?;
            if reply.calls.is_empty() || terminal {
                let graded = self.finish(&episode.id, &reply.text).await;
                session
                    .text(envelope("episode_done", &graded).to_string())
                    .await
                    .ok()?;
                return Some(());
            }
            turns.push(json!({"role":"assistant","text":reply.text}));
            turns.push(self.step_through(session, &episode.id, &reply).await?);
        }
        Some(())
    }

    /// Execute calls in filesystem order, report each result, and retain only the first answer in model history.
    async fn step_through(
        &self,
        session: &mut actix_ws::Session,
        episode_id: &str,
        reply: &Reply,
    ) -> Option<Value> {
        let mut first = None;
        for call in &reply.calls {
            let body = steps::execute(
                self.gyms,
                self.caller,
                episode_id,
                &call.tool,
                &call.argument,
                &reply.text,
                self.observation_budget_chars,
            )
            .await;
            let mut result = call.result(body);
            result["episode_id"] = json!(episode_id);
            session
                .text(envelope("tool_result", &result).to_string())
                .await
                .ok()?;
            first.get_or_insert(json!({"role":"tool","tool":call.tool,"argument":call.argument,"text":result["body"]}));
        }
        first
    }

    /// Store the final reply's grade, retaining the existing error-shaped completion payload on service failure.
    async fn finish(&self, episode_id: &str, reply: &str) -> Value {
        match self
            .gyms
            .post(
                self.caller,
                &format!("/gym/episodes/{episode_id}/finish"),
                json!({"reply":reply}),
            )
            .await
        {
            Ok(graded) => graded,
            Err(error) => json!({"episode_id":episode_id,"error":error.to_string()}),
        }
    }
}

/// Forward a round through stream exhaustion, rename `done` to `round_done`, and tag it with its episode.
/// Preserve the legacy handling of upstream error frames; a socket delivery failure abandons the episode.
async fn relay(
    session: &mut actix_ws::Session,
    frames: impl Stream<Item = llm_engine::Result<Frame>>,
    episode_id: &str,
) -> Option<Reply> {
    futures_util::pin_mut!(frames);
    let mut reply = Reply {
        text: String::new(),
        calls: Vec::new(),
    };
    while let Some(frame) = frames.next().await {
        let message = match frame {
            Ok(frame) => {
                if frame.kind == "tool_call" {
                    reply.calls.push(ToolCall::from_value(&frame.data))
                }
                if frame.kind == "done" {
                    reply.text = string_at(&frame.data, "text")
                }
                episode_frame(&frame, episode_id)
            }
            Err(error) => ws::error_frame(&error.to_string()),
        };
        session.text(message).await.ok()?;
    }
    Some(reply)
}

/// Add authoritative episode attribution after merging opaque service fields and rename round completion.
fn episode_frame(frame: &Frame, episode_id: &str) -> String {
    let kind = if frame.kind == "done" {
        "round_done"
    } else {
        &frame.kind
    };
    let mut payload = envelope(kind, &frame.data);
    payload["episode_id"] = json!(episode_id);
    payload.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gym attribution cannot be replaced by upstream data and completion uses the client round vocabulary.
    #[test]
    fn round_frames_keep_authoritative_episode_identity() {
        let frame = Frame {
            kind: "done".into(),
            data: json!({"episode_id":"wrong","extension":42}),
        };
        let payload: Value = serde_json::from_str(&episode_frame(&frame, "right")).unwrap();
        assert_eq!(payload["episode_id"], "right");
        assert_eq!(payload["type"], "round_done");
        assert_eq!(payload["extension"], 42);
    }
}
