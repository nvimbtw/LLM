//! Tool execution owned by one generation round.

use serde_json::{json, Value};
use tokio::task::JoinHandle;
use tools::Registry;

use crate::streaming::{envelope, ToolCall};

pub(super) struct Pending {
    call: ToolCall,
    work: JoinHandle<(String, Option<tools::ToolError>)>,
}

impl Pending {
    /// Start this call immediately in the chat's tool session; dropping the owner aborts unfinished work.
    pub fn start(registry: &Registry, chat: &str, call: ToolCall) -> Self {
        let registry = registry.clone();
        let task_call = call.clone();
        let chat = chat.to_owned();
        Self {
            call,
            work: actix_web::rt::spawn(async move {
                registry
                    .answer(&task_call.tool, &task_call.argument, &chat)
                    .await
            }),
        }
    }

    /// Await the owned tool and return its rendered result, converting panics/cancellation into a caveat.
    async fn result(&mut self) -> Value {
        let (body, failure) = (&mut self.work).await.unwrap_or_else(|error| {
            (
                tools::markup::warning("The tool did not finish."),
                Some(tools::ToolError::Transport(error.to_string())),
            )
        });
        if let Some(error) = failure {
            eprintln!("tool {} failed: {error}", self.call.tool);
        }
        self.call.result(body)
    }
}

impl Drop for Pending {
    /// Abort the tool task when its round is cancelled, truncated or disconnected.
    fn drop(&mut self) {
        self.work.abort();
    }
}

/// Report all completed calls in order and continue with the first result only.
/// A failed socket write stops the turn and drops any remaining owned tasks.
pub(super) async fn answer(
    session: &mut actix_ws::Session,
    pending: Vec<Pending>,
) -> Option<Value> {
    let mut first = None;
    for mut pending in pending {
        let result = pending.result().await;
        session
            .text(envelope("tool_result", &result).to_string())
            .await
            .ok()?;
        first.get_or_insert(result);
    }
    Some(json!({"tool_result": first?}))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dropping a round's pending call cancels its task instead of detaching background effects.
    #[tokio::test]
    async fn pending_work_is_aborted_on_drop() {
        let (started, entered) = tokio::sync::oneshot::channel();
        let (released, exited) = tokio::sync::oneshot::channel::<()>();
        let work = tokio::spawn(async move {
            let _release = released;
            started.send(()).unwrap();
            std::future::pending::<()>().await;
            unreachable!()
        });
        let pending = Pending {
            call: ToolCall::from_value(&json!({})),
            work,
        };
        entered.await.unwrap();
        drop(pending);
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), exited)
                .await
                .unwrap()
                .is_err()
        );
    }

    /// Joining a finished call preserves its tool/argument/body and leaves no active task to detach.
    #[tokio::test]
    async fn completed_work_is_joined_normally() {
        let mut pending = Pending {
            call: ToolCall::from_value(&json!({"tool":"bash", "argument":"pwd"})),
            work: tokio::spawn(async { ("output".to_owned(), None) }),
        };
        assert_eq!(
            pending.result().await,
            json!({"tool":"bash", "argument":"pwd", "body":"output"})
        );
        assert!(pending.work.is_finished());
    }
}
