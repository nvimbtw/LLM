//! Sockets: the browser's half of a stream the engine sends as SSE.
//!
//! The engine reports over plain HTTP because it has one consumer and no
//! handshake to get wrong. A browser wants a socket — so this is where the two
//! meet: frames are relayed as they arrive, and the one thing a client sends
//! back, cancelling a run, goes out as an ordinary request.

use actix_web::{HttpRequest, HttpResponse, get, web};
use engine::{Frame, Proxy};
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::ws;
use auth::AccessLevel;
use auth::token::Claims;

#[get("/runs/{run_id}")]
async fn run_socket(
    req: HttpRequest,
    body: web::Payload,
    run_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
    secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, stream) = actix_ws::handle(&req, body)?;
    actix_web::rt::spawn(run_session(
        session,
        stream,
        engine.get_ref().clone(),
        run_id.into_inner(),
        secret.get_ref().clone(),
    ));
    Ok(response)
}

#[get("/chats/{chat_id}")]
async fn chat_socket(
    req: HttpRequest,
    body: web::Payload,
    chat_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
    secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, stream) = actix_ws::handle(&req, body)?;
    actix_web::rt::spawn(chat_session(
        session,
        stream,
        engine.get_ref().clone(),
        chat_id.into_inner(),
        secret.get_ref().clone(),
    ));
    Ok(response)
}

async fn run_session(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    engine: Proxy,
    run_id: Uuid,
    secret: String,
) {
    let Some(claims) = handshake(&mut session, &mut stream, &secret).await else { return };
    let caller = auth::Caller::User(claims.sub);

    let control = actix_web::rt::spawn(watch_for_cancel(
        stream,
        engine.clone(),
        run_id,
        matches!(claims.access_level, AccessLevel::Admin),
    ));

    match engine.events(&caller, &format!("/runs/{run_id}/events")).await {
        Ok(frames) => relay(&mut session, frames).await,
        Err(e) => {
            let _ = session.text(ws::error_frame(&e.to_string())).await;
        }
    }

    control.abort();
    let _ = session.close(None).await;
}

async fn chat_session(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    engine: Proxy,
    chat_id: Uuid,
    secret: String,
) {
    let Some(claims) = handshake(&mut session, &mut stream, &secret).await else { return };
    let caller = auth::Caller::User(claims.sub);

    // The turn to answer arrives after the token, on the same socket — one
    // connection for the question and the reply it streams back.
    let Some(ask) = next_json(&mut stream).await else { return };

    let path = format!("/chats/{chat_id}/messages/stream");
    match engine.stream(&caller, &path, ask).await {
        Ok(frames) => relay(&mut session, frames).await,
        Err(e) => {
            let _ = session.text(ws::error_frame(&e.to_string())).await;
        }
    }
    let _ = session.close(None).await;
}

/// Read the token frame every socket opens with.
///
/// Browsers cannot set a header on a WebSocket handshake, so the token comes as
/// the first message instead — the same value the HTTP routes carry as
/// `Authorization: Bearer`.
async fn handshake(
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

async fn next_json(stream: &mut actix_ws::MessageStream) -> Option<Value> {
    use actix_ws::Message;

    loop {
        match stream.next().await {
            Some(Ok(Message::Text(text))) => return serde_json::from_str(&text).ok(),
            Some(Ok(Message::Close(_))) | None => return None,
            _ => {}
        }
    }
}

/// Engine frames, flattened onto a `type` field the client switches on.
async fn relay(
    session: &mut actix_ws::Session,
    frames: impl Stream<Item = engine::Result<Frame>>,
) {
    futures_util::pin_mut!(frames);
    while let Some(frame) = frames.next().await {
        let message = match frame {
            Ok(frame) => {
                let mut payload = json!({ "type": frame.kind });
                merge(&mut payload, frame.data);
                payload.to_string()
            }
            Err(e) => ws::error_frame(&e.to_string()),
        };
        if session.text(message).await.is_err() {
            return; // The browser hung up. The run carries on without it.
        }
    }
}

/// Cancelling is the only thing a client says back, and only an admin may.
async fn watch_for_cancel(
    mut stream: actix_ws::MessageStream,
    engine: Proxy,
    run_id: Uuid,
    is_admin: bool,
) {
    use actix_ws::Message;

    while let Some(Ok(message)) = stream.next().await {
        let Message::Text(text) = message else { continue };
        let asked = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| v.get("type").and_then(Value::as_str).map(str::to_owned));
        if asked.as_deref() == Some("cancel") && is_admin {
            let path = format!("/runs/{run_id}/cancel");
            let _ = engine.post(&auth::Caller::System, &path, Value::Null).await;
        }
    }
}

fn merge(target: &mut Value, source: Value) {
    let (Some(target), Value::Object(source)) = (target.as_object_mut(), source) else { return };
    for (key, value) in source {
        target.insert(key, value);
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(run_socket);
    cfg.service(chat_socket);
}
