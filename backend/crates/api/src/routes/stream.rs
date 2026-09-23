//! The browser's half of a stream the LLM engine sends as SSE.
//!
//! The LLM engine reports over plain HTTP because it has one consumer and no
//! handshake to get wrong; a browser wants a socket. This is where the two meet.

use actix_web::{get, web, HttpRequest, HttpResponse};
use futures_util::{Stream, StreamExt};
use llm_engine::{Frame, Proxy};
use serde_json::{json, Value};
use tools::Registry;
use uuid::Uuid;

use crate::{gym, turn, ws};
use auth::token::Claims;
use auth::AccessLevel;

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

#[get("/data/runs/{run_id}")]
async fn data_run_socket(
    req: HttpRequest,
    body: web::Payload,
    run_id: web::Path<Uuid>,
    data: web::Data<data_engine::Proxy>,
    secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, stream) = actix_ws::handle(&req, body)?;
    actix_web::rt::spawn(data_run_session(
        session,
        stream,
        data.get_ref().clone(),
        run_id.into_inner(),
        secret.get_ref().clone(),
    ));
    Ok(response)
}

/// A pull or a processing pass, frame by frame.
///
/// Deliberately not `run_session` with a second proxy type: that one also wires
/// up cancellation, and a data run is cancelled through a plain POST that any
/// admin already has. Sharing the body would mean threading a "can this be
/// cancelled over the socket" flag through it to save six lines.
async fn data_run_session(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    data: data_engine::Proxy,
    run_id: Uuid,
    secret: String,
) {
    let Some(claims) = handshake(&mut session, &mut stream, &secret).await else {
        return;
    };
    let caller = auth::Caller::User(claims.sub);

    match data
        .events(&caller, &format!("/runs/{run_id}/events"))
        .await
    {
        Ok(frames) => relay(&mut session, frames).await,
        Err(e) => {
            let _ = session.text(ws::error_frame(&e.to_string())).await;
        }
    }

    let _ = session.close(None).await;
}

#[get("/gyms/runs/{run_id}")]
async fn gym_socket(
    req: HttpRequest,
    body: web::Payload,
    run_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
    gyms: web::Data<gyms_engine::Proxy>,
    secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, stream) = actix_ws::handle(&req, body)?;
    actix_web::rt::spawn(gym_session(
        session,
        stream,
        engine.get_ref().clone(),
        gyms.get_ref().clone(),
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
    registry: web::Data<Registry>,
    secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, stream) = actix_ws::handle(&req, body)?;
    actix_web::rt::spawn(chat_session(
        session,
        stream,
        engine.get_ref().clone(),
        registry.get_ref().clone(),
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
    let Some(claims) = handshake(&mut session, &mut stream, &secret).await else {
        return;
    };
    let caller = auth::Caller::User(claims.sub);

    let control = actix_web::rt::spawn(watch_for_cancel(
        stream,
        engine.clone(),
        run_id,
        matches!(claims.access_level, AccessLevel::Admin),
    ));

    match engine
        .events(&caller, &format!("/runs/{run_id}/events"))
        .await
    {
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
    registry: Registry,
    chat_id: Uuid,
    secret: String,
) {
    let Some(claims) = handshake(&mut session, &mut stream, &secret).await else {
        return;
    };
    let caller = auth::Caller::User(claims.sub);

    // One connection for the question and the reply it streams back — which may
    // take several rounds through the LLM engine if the reply calls a tool.
    let Some(ask) = next_json(&mut stream).await else {
        return;
    };

    // From here on the socket is read concurrently with the turn: a person may
    // type and send again before this reply is done, and the only way to learn
    // that is to keep reading. `stream` moves into this task for the rest of
    // the turn's lifetime — `turn::run` never touches it directly, only the
    // channel it feeds.
    let (sender, interjections) = tokio::sync::mpsc::unbounded_channel();
    let reader = actix_web::rt::spawn(read_interjections(stream, sender));

    let path = format!("/chats/{chat_id}/messages/stream");
    // The chat id is the tools' session key: it is stable for the whole
    // conversation, which is exactly the lifetime a sandbox filesystem wants.
    turn::run(
        &mut session,
        &engine,
        &registry,
        &caller,
        &path,
        &chat_id.to_string(),
        ask,
        interjections,
    )
    .await;
    reader.abort();
    let _ = session.close(None).await;
}

/// The client half of a chat socket once its turn has started: watches for
/// `{ "type": "interject", "text": ... }` and forwards the text to `turn::run`.
///
/// No `attachment_ids` here, unlike the opening `ask` — see `Ask.interject` in
/// `llm-engine/api/routes/chats.py` for why an interjection is text-only. Anything
/// else on the socket (a stray `ping`, garbage, another frame type) is ignored
/// rather than closing the connection: this task has no way to answer back,
/// and the turn it is feeding is the thing actually driving the socket.
async fn read_interjections(
    mut stream: actix_ws::MessageStream,
    sender: tokio::sync::mpsc::UnboundedSender<String>,
) {
    use actix_ws::Message;

    while let Some(Ok(message)) = stream.next().await {
        let Message::Text(text) = message else {
            continue;
        };
        let Some(value) = serde_json::from_str::<Value>(&text).ok() else {
            continue;
        };
        if value.get("type").and_then(Value::as_str) != Some("interject") {
            continue;
        }
        let Some(text) = value.get("text").and_then(Value::as_str) else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        // The receiver is dropped once the turn ends, at which point sending
        // fails silently — a message sent right as the last round closes has
        // nowhere left to go, and the client resends it as a fresh turn (see
        // `docs/frontend/README.md`'s "Stream protocol (chat)").
        let _ = sender.send(text.to_string());
    }
}

/// A gym run, episode by episode. Admin only, like every other gym route: a run
/// names a model and spends a GPU for as long as it takes.
async fn gym_session(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    engine: Proxy,
    gyms: gyms_engine::Proxy,
    run_id: Uuid,
    secret: String,
) {
    let Some(claims) = handshake(&mut session, &mut stream, &secret).await else {
        return;
    };
    if !matches!(claims.access_level, AccessLevel::Admin) {
        let _ = session.text(ws::error_frame("admin only")).await;
        let _ = session.close(None).await;
        return;
    }
    let caller = auth::Caller::User(claims.sub);

    // The sampling the client chose, if any. Unlike a chat there is no question
    // in it — the run already holds the model and the tasks.
    let Some(ask) = next_json(&mut stream).await else {
        return;
    };

    gym::run(&mut session, &engine, &gyms, &caller, run_id, &ask).await;
    let _ = session.close(None).await;
}

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

/// Frames flattened onto a `type` field the client switches on.
async fn relay(
    session: &mut actix_ws::Session,
    frames: impl Stream<Item = llm_engine::Result<Frame>>,
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
        let Message::Text(text) = message else {
            continue;
        };
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
    let (Some(target), Value::Object(source)) = (target.as_object_mut(), source) else {
        return;
    };
    for (key, value) in source {
        target.insert(key, value);
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    // Before `/runs/{run_id}`, which would otherwise try to read "data" as a
    // run id and fail the path extractor.
    cfg.service(data_run_socket);
    cfg.service(gym_socket);
    cfg.service(run_socket);
    cfg.service(chat_socket);
}
