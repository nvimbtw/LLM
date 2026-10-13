//! Authentication and opaque event relay shared by browser socket sessions.

use crate::{streaming::envelope, ws};
use auth::token::Claims;
use futures_util::{Stream, StreamExt};
use llm_engine::Frame;
use serde_json::{json, Value};

/// Authenticate the opening text frame and acknowledge readiness; pings are answered before authentication.
pub(super) async fn handshake(
    session: &mut actix_ws::Session,
    stream: &mut actix_ws::MessageStream,
    secret: &str,
) -> Option<Claims> {
    use actix_ws::Message;

    loop {
        match stream.next().await {
            Some(Ok(Message::Text(text))) => match ws::authenticate(&text, secret) {
                Some(claims) => {
                    let _ = session.text(json!({ "type": "ready" }).to_string()).await;
                    return Some(claims);
                }
                None => {
                    let _ = session.text(ws::error_frame("authentication failed")).await;
                    let _ = session.clone().close(None).await;
                    return None;
                }
            },
            Some(Ok(Message::Ping(bytes))) => {
                let _ = session.pong(&bytes).await;
            }
            Some(Ok(Message::Close(_))) | None => return None,
            _ => {}
        }
    }
}

/// Read the next text payload as JSON, returning no request after invalid JSON, close or end of stream.
pub(super) async fn next_json(stream: &mut actix_ws::MessageStream) -> Option<Value> {
    use actix_ws::Message;

    loop {
        match stream.next().await {
            Some(Ok(Message::Text(text))) => return serde_json::from_str(&text).ok(),
            Some(Ok(Message::Close(_))) | None => return None,
            _ => {}
        }
    }
}

/// Forward opaque service frames until exhaustion or delivery failure; disconnect leaves durable jobs running.
pub(super) async fn relay(
    session: &mut actix_ws::Session,
    frames: impl Stream<Item = llm_engine::Result<Frame>>,
) {
    futures_util::pin_mut!(frames);
    while let Some(frame) = frames.next().await {
        let message = match frame {
            Ok(frame) => envelope(&frame.kind, &frame.data).to_string(),
            Err(e) => ws::error_frame(&e.to_string()),
        };
        if session.text(message).await.is_err() {
            return;
        }
    }
}
