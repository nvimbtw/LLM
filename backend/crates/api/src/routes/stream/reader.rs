//! Socket readers with an explicit owner for their spawned task.

use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use tokio::{sync::mpsc, task::JoinHandle};
use uuid::Uuid;

pub(super) struct SocketReader(JoinHandle<()>);

impl SocketReader {
    /// Own a spawned browser reader; dropping this guard aborts reading rather than detaching it.
    pub fn new(task: JoinHandle<()>) -> Self {
        Self(task)
    }
}

impl Drop for SocketReader {
    /// Cancel the reader task on normal completion, session cancellation or unwinding.
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Control {
    #[serde(rename = "cancel")]
    Cancel,
    #[serde(rename = "interject")]
    Interject { text: String },
}

/// Read only valid nonempty text steering; attachments and unknown frame types remain ignored.
/// A dropped receiver means the turn has ended and the client retains responsibility for resending queued text.
pub(super) async fn interjections(
    mut stream: actix_ws::MessageStream,
    sender: mpsc::UnboundedSender<String>,
) {
    while let Some(Ok(message)) = stream.next().await {
        let actix_ws::Message::Text(text) = message else {
            continue;
        };
        if let Some(text) = interjection(&text) {
            let _ = sender.send(text);
        }
    }
}

/// Return text steering from a control frame, preserving whitespace and rejecting empty or wrongly typed text.
fn interjection(text: &str) -> Option<String> {
    match serde_json::from_str::<Control>(text).ok()? {
        Control::Interject { text } if !text.is_empty() => Some(text),
        _ => None,
    }
}

/// Forward best-effort explicit cancellation with the verified administrator actor/scope; lifecycle acknowledgement stays in job events.
/// Stopping this reader alone never cancels a job, and an ordinary observer has no control identity.
pub(super) async fn cancellation(
    mut stream: actix_ws::MessageStream,
    engine: llm_engine::Proxy,
    run_id: Uuid,
    administrator: Option<Uuid>,
) {
    while let Some(Ok(message)) = stream.next().await {
        let actix_ws::Message::Text(text) = message else {
            continue;
        };
        if let Some(actor) = administrator
            .filter(|_| matches!(serde_json::from_str::<Control>(&text), Ok(Control::Cancel)))
        {
            let _ = engine
                .scoped_json(
                    &auth::Caller::User(actor),
                    "administration",
                    llm_engine::Method::POST,
                    &format!("/runs/{run_id}/cancel"),
                    Some(Value::Null),
                    8 * 1024,
                )
                .await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Control parsing keeps the text-only interjection contract without treating whitespace as empty.
    #[test]
    fn controls_preserve_existing_acceptance_rules() {
        assert_eq!(
            interjection(r#"{"type":"interject","text":" "}"#),
            Some(" ".into())
        );
        for text in [
            "garbage",
            r#"{"type":"other","text":"wait"}"#,
            r#"{"type":"interject","text":7}"#,
            r#"{"type":"interject","text":""}"#,
        ] {
            assert_eq!(interjection(text), None);
        }
    }

    /// Dropping a reader owner cancels the blocked task and releases its resources.
    #[tokio::test]
    async fn dropping_reader_stops_its_task() {
        let (entered, started) = tokio::sync::oneshot::channel();
        let (release, released) = tokio::sync::oneshot::channel::<()>();
        let reader = SocketReader::new(tokio::spawn(async move {
            let _release = release;
            entered.send(()).unwrap();
            std::future::pending::<()>().await;
        }));
        started.await.unwrap();
        drop(reader);
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), released)
                .await
                .unwrap()
                .is_err()
        );
    }

    /// A normally finished reader can be dropped after its effects complete without affecting another task.
    #[tokio::test]
    async fn finished_reader_cleans_up_normally() {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let reader = SocketReader::new(tokio::spawn(async move {
            sender.send(()).unwrap();
        }));
        receiver.await.unwrap();
        drop(reader);
    }
}
