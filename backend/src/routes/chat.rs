use actix_web::{HttpRequest, HttpResponse, Responder, delete, get, post, put, web};
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::sync::Arc;
use uuid::Uuid;

use crate::queue::{ChatRequest, Queue};
use crate::routes::validate;
use crate::{extractors::AuthUser, ws};
use db::user::AccessLevel;

#[derive(Serialize)]
struct ChatResponse {
    id_chat: Uuid,
    title: String,
    created_at: DateTime<Utc>,
    // Cumulative tokens across the chat's messages. A freshly created chat (the
    // From<Chat> path) has none yet, so it reports 0.
    tokens: i64,
}

impl From<db::chat::Chat> for ChatResponse {
    fn from(chat: db::chat::Chat) -> Self {
        ChatResponse {
            id_chat: chat.id_chat,
            title: chat.title,
            created_at: chat.created_at,
            tokens: 0,
        }
    }
}

impl From<db::chat::ChatListItem> for ChatResponse {
    fn from(chat: db::chat::ChatListItem) -> Self {
        ChatResponse {
            id_chat: chat.id_chat,
            title: chat.title,
            created_at: chat.created_at,
            tokens: chat.tokens,
        }
    }
}

#[derive(Serialize)]
struct MessageResponse {
    id_message: Uuid,
    role: &'static str,
    content: String,
    // Display name of the model that wrote this reply, for the byline under it.
    // Null on a user's own message.
    model_name: Option<String>,
    // The thumbs the user already gave this reply, so a reload renders the
    // rating rather than an empty pair of buttons. Null when unrated.
    feedback: Option<db::message::Feedback>,
    created_at: DateTime<Utc>,
}

impl From<db::message::MessageWithModel> for MessageResponse {
    fn from(message: db::message::MessageWithModel) -> Self {
        // The schema encodes authorship by which FK is set; we collapse that
        // into a simple role string the frontend can switch on. id_user set =>
        // a person sent it; otherwise it's an assistant reply.
        let role = if message.id_user.is_some() {
            "user"
        } else {
            "assistant"
        };

        MessageResponse {
            id_message: message.id_message,
            role,
            content: message.content,
            model_name: message.model_name,
            feedback: message.feedback,
            created_at: message.created_at,
        }
    }
}

// Loads a chat and confirms the caller owns it, returning either the chat or a
// ready-to-send error response. Chats are private, so this is owner-only — no
// admin override. On a Err the handler just returns the HttpResponse directly.
async fn owned_chat(
    pool: &PgPool,
    id_chat: Uuid,
    user_id: Uuid,
) -> Result<db::chat::Chat, HttpResponse> {
    match db::chat::find_by_id(pool, id_chat).await {
        Ok(Some(chat)) if chat.id_user == user_id => Ok(chat),
        Ok(Some(_)) => Err(HttpResponse::Forbidden().body("forbidden")),
        Ok(_) => Err(HttpResponse::NotFound().body("chat not found")),
        Err(e) => {
            eprintln!("chat lookup db error: {}", e);
            Err(HttpResponse::InternalServerError().body("failed to load chat"))
        }
    }
}

// New chats get a provisional title from the first message; the client can
// rename later via PUT. Takes the first line, capped to a sane length.
fn derive_title(content: &str) -> String {
    const MAX_CHARS: usize = 50;
    let first_line = content.lines().next().unwrap_or("").trim();
    let title: String = first_line.chars().take(MAX_CHARS).collect();

    if title.is_empty() {
        "New chat".to_string()
    } else {
        title
    }
}

// Every subsequent inbound frame is a chat turn.
#[derive(Deserialize)]
struct SendMessageFrame {
    // Absent => start a new chat; present => append to an existing one.
    chat_id: Option<Uuid>,
    content: String,
    // Which trained model should generate the reply. Required for a real turn;
    // resolves to the model's on-disk weights.
    model_id: Option<Uuid>,
    // Optional system prompt to steer the reply. Absent/empty => no system turn;
    // the client omits it when the user hasn't set one.
    #[serde(default)]
    system: Option<String>,
}

// An inbound `{ "action": "cancel" }` while a turn is in flight. The turn is a
// queued job, so this is the same cancel the /jobs page offers, reachable from
// where the user actually is.
#[derive(Deserialize)]
struct ActionFrame {
    action: String,
}

// WebSocket entrypoint. The handshake is a GET upgrade; we hand the socket off
// to a spawned task and return the upgrade response immediately. Auth happens
// inside the session (first frame), not here, since the browser can't send the
// Authorization header on the handshake.
#[get("/ws")]
async fn chat_ws(
    req: HttpRequest,
    body: web::Payload,
    pool: web::Data<PgPool>,
    queue: web::Data<Arc<Queue>>,
    jwt_secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, msg_stream) = actix_ws::handle(&req, body)?;

    // Owned clones for the 'static task: PgPool is an Arc internally (cheap to
    // clone), and we copy the secret string.
    let pool = pool.get_ref().clone();
    let queue = queue.get_ref().clone();
    let secret = jwt_secret.get_ref().clone();

    actix_web::rt::spawn(chat_session(session, msg_stream, pool, queue, secret));

    Ok(response)
}

// Drives one socket for its lifetime: authenticate on the first text frame,
// then treat every later text frame as a chat turn.
async fn chat_session(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    pool: PgPool,
    queue: Arc<Queue>,
    secret: String,
) {
    // (user id, is_admin) once authenticated. is_admin gates use of private models.
    let mut auth: Option<(Uuid, bool)> = None;

    while let Some(Ok(msg)) = stream.next().await {
        match msg {
            actix_ws::Message::Text(text) => {
                // Message::Text carries a ByteString; bind it as &str once so the
                // helpers below get the type they expect.
                let text: &str = &text;
                match auth {
                    // Not yet authenticated: this frame must be the auth token.
                    None => match ws::authenticate(text, &secret) {
                        Some(claims) => {
                            let is_admin = matches!(claims.access_level, AccessLevel::Admin);
                            auth = Some((claims.sub, is_admin));
                            // Best-effort: if the client vanished between frames,
                            // the next stream read ends the loop anyway.
                            let _ = session
                                .text(serde_json::json!({ "type": "ready" }).to_string())
                                .await;
                        }
                        None => {
                            // Best-effort notify + close; we return regardless of
                            // whether the peer is still there to receive them.
                            let _ = session.text(ws::error_frame("authentication failed")).await;
                            let _ = session.close(None).await;
                            return;
                        }
                    },
                    // Authenticated: handle a chat turn, streaming tokens back.
                    Some((uid, is_admin)) => {
                        if handle_message(
                            &mut session,
                            &mut stream,
                            &pool,
                            &queue,
                            uid,
                            is_admin,
                            text,
                        )
                        .await
                        .is_err()
                        {
                            // Client went away mid-stream; tear the session down.
                            return;
                        }
                    }
                }
            }
            actix_ws::Message::Ping(bytes) => {
                // Best-effort pong; a dead peer just ends the loop on next read.
                let _ = session.pong(&bytes).await;
            }
            actix_ws::Message::Close(_) => break,
            _ => {}
        }
    }

    // Best-effort close on the way out; the socket may already be gone.
    let _ = session.close(None).await;
}

// Processes one chat turn: resolve/create the chat, persist the user message, then
// queue the generation and forward its tokens as they arrive.
//
// The generation itself is a `chat` job on the GPU queue rather than a thread
// started here. That costs a wait when something long is already running — the
// turn sits in line behind it — and buys two things: the turn can no longer
// collide with a training run on the one device, and the reply survives the tab
// being closed, because the runner persists it whether or not anyone is listening.
//
// Each stage below sends its own error frame and signals "stop this turn" when it
// fails, so this stays a readable sequence. Returns Err only when sending to the
// socket fails (client disconnected), so the caller can stop the session.
async fn handle_message(
    session: &mut actix_ws::Session,
    stream: &mut actix_ws::MessageStream,
    pool: &PgPool,
    queue: &Arc<Queue>,
    user_id: Uuid,
    is_admin: bool,
    text: &str,
) -> Result<(), actix_ws::Closed> {
    let Some(frame) = parse_frame(session, text).await? else {
        return Ok(());
    };
    let content = frame.content.trim();
    if content.is_empty() {
        return session.text(ws::error_frame("message content is empty")).await;
    }
    if validate::max_len("message", content, validate::MAX_MESSAGE_LEN).is_err() {
        return session.text(ws::error_frame("message content too long")).await;
    }

    // Normalize the optional system prompt: trim, drop if blank, length-check.
    let system = frame
        .system
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    if let Some(ref s) = system {
        if validate::max_len("system prompt", s, validate::MAX_MESSAGE_LEN).is_err() {
            return session.text(ws::error_frame("system prompt too long")).await;
        }
    }

    let Some(chat_id) = resolve_chat(session, pool, user_id, frame.chat_id, content).await? else {
        return Ok(());
    };
    // Resolve the model before persisting, so we can tokenize the user message
    // with its tokenizer and store the count alongside the row.
    let Some(model) = resolve_model(session, pool, frame.model_id, is_admin).await? else {
        return Ok(());
    };
    let model_dir = storage::models_dir().join(model.id_model.to_string());

    // Non-fatal: if the tokenizer can't be read the message is still saved, just
    // with a 0 token count rather than blocking the turn.
    let user_tokens = llm::infer::encode_count(&model_dir, content).unwrap_or(0) as i32;
    if !persist_user_message(session, pool, chat_id, user_id, content, user_tokens).await? {
        return Ok(());
    }

    // Subscribed *before* the job exists. The runner announces a job going live on
    // this channel, and a short turn can reach the GPU before the enqueue call even
    // returns here — subscribing afterwards would miss the announcement and leave
    // this socket waiting for a stream that had already started.
    let events = queue.subscribe();

    let request = ChatRequest {
        id_chat: chat_id,
        id_model: model.id_model,
        id_user: Some(user_id),
        system,
    };
    let config = match serde_json::to_value(&request) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("serialize chat request error: {}", e);
            return session.text(ws::error_frame("failed to queue reply")).await;
        }
    };

    let job = match db::job::create(
        pool,
        db::job::JobKind::Chat,
        &derive_title(content),
        &config,
        Some(user_id),
    )
    .await
    {
        Ok(job) => job,
        Err(e) => {
            eprintln!("queue chat job db error: {}", e);
            return session.text(ws::error_frame("failed to queue reply")).await;
        }
    };
    queue.wake();
    queue.notify_changed();

    // Tell the client where it is in line before anything else. A turn that starts
    // immediately still gets this frame with `ahead: 0`, so the client has one code
    // path rather than two.
    let ahead = jobs_ahead(pool, job.queue_rank).await;
    session
        .text(
            serde_json::json!({
                "type": "queued",
                "job_id": job.id_job,
                "chat_id": chat_id,
                "ahead": ahead,
            })
            .to_string(),
        )
        .await?;

    follow_job(session, stream, pool, queue, events, job.id_job, chat_id).await
}

// How many live jobs sit in front of this one. Best-effort: a failed count reports
// zero rather than sinking the turn over a number that is only ever a hint.
async fn jobs_ahead(pool: &PgPool, rank: f64) -> usize {
    match db::job::list_active(pool).await {
        Ok(jobs) => jobs.iter().filter(|j| j.queue_rank < rank).count(),
        Err(e) => {
            eprintln!("count queued jobs db error: {}", e);
            0
        }
    }
}

// Waits for the turn to reach the GPU, then forwards its frames to the client.
//
// Two things can happen while waiting, and both are handled here: the job goes
// live (attach to its broadcast), or it reaches a terminal state without this
// socket ever seeing it run (cancelled in the queue, or finished so fast the
// attach lost the race). The second case falls back to the durable row, which is
// the point of persisting the reply in the first place.
async fn follow_job(
    session: &mut actix_ws::Session,
    stream: &mut actix_ws::MessageStream,
    pool: &PgPool,
    queue: &Arc<Queue>,
    mut events: tokio::sync::broadcast::Receiver<String>,
    id_job: Uuid,
    chat_id: Uuid,
) -> Result<(), actix_ws::Closed> {
    use actix_ws::Message;

    // ---- wait for the job to go live ----
    let mut frames = loop {
        if let Some(live) = queue.live_for(id_job) {
            break Some(live.tx.subscribe());
        }
        // Not running. Either it hasn't started, or it is already over.
        match db::job::find_by_id(pool, id_job).await {
            Ok(Some(job)) if job.status.is_terminal() => break None,
            Ok(_) => {}
            Err(e) => {
                eprintln!("poll chat job db error: {}", e);
                return session.text(ws::error_frame("failed to track the reply")).await;
            }
        }

        tokio::select! {
            biased;
            // A cancel while queued has to be honoured here: nothing is running, so
            // there is no worker flag to set.
            incoming = stream.next() => match incoming {
                Some(Ok(Message::Text(text))) => {
                    if is_cancel(&text) {
                        let _ = db::job::cancel_queued(pool, id_job).await;
                        queue.request_cancel(id_job);
                        queue.notify_changed();
                    }
                }
                Some(Ok(Message::Ping(bytes))) => {
                    let _ = session.pong(&bytes).await;
                }
                Some(Ok(Message::Close(_))) | None => return Err(actix_ws::Closed),
                _ => {}
            },
            event = events.recv() => match event {
                // Any board change is a reason to re-check; the loop head does the
                // actual deciding, so the payload doesn't need inspecting.
                Ok(_) => {}
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break None,
            },
        }
    };

    // ---- forward frames until the turn ends ----
    if let Some(rx) = frames.as_mut() {
        loop {
            tokio::select! {
                biased;
                incoming = stream.next() => match incoming {
                    Some(Ok(Message::Text(text))) => {
                        if is_cancel(&text) {
                            queue.request_cancel(id_job);
                        }
                    }
                    Some(Ok(Message::Ping(bytes))) => {
                        let _ = session.pong(&bytes).await;
                    }
                    Some(Ok(Message::Close(_))) | None => return Err(actix_ws::Closed),
                    _ => {}
                },
                msg = rx.recv() => match msg {
                    Ok(text) => {
                        // The runner's frames are already in this socket's wire
                        // shape — `token`, `done`, `error` — so they forward
                        // verbatim rather than being re-rendered here.
                        let terminal = text.contains("\"type\":\"done\"")
                            || text.contains("\"type\":\"error\"");
                        session.text(text).await?;
                        if terminal {
                            return Ok(());
                        }
                    }
                    // The job ended without a terminal frame reaching us (or this
                    // socket fell behind). The row is authoritative.
                    Err(_) => break,
                },
            }
        }
    }

    // ---- fell through: report the outcome from the durable row ----
    match db::job::find_by_id(pool, id_job).await {
        Ok(Some(job)) => {
            let done = serde_json::json!({
                "type": "done",
                "chat_id": chat_id,
                // Present once the reply was persisted. The client reloads the
                // conversation when it gets a `done` it has no streamed text for.
                "id_message": job.ref_id,
            });
            session.text(done.to_string()).await
        }
        Ok(None) => session.text(ws::error_frame("the reply was cancelled")).await,
        Err(e) => {
            eprintln!("resolve chat job db error: {}", e);
            session.text(ws::error_frame("failed to track the reply")).await
        }
    }
}

fn is_cancel(text: &str) -> bool {
    serde_json::from_str::<ActionFrame>(text)
        .map(|f| f.action == "cancel")
        .unwrap_or(false)
}

// Parse the incoming frame; sends an error and returns None if it's malformed.
async fn parse_frame(
    session: &mut actix_ws::Session,
    text: &str,
) -> Result<Option<SendMessageFrame>, actix_ws::Closed> {
    match serde_json::from_str(text) {
        Ok(frame) => Ok(Some(frame)),
        Err(_) => {
            session.text(ws::error_frame("invalid message frame")).await?;
            Ok(None)
        }
    }
}

// Resolve the target chat: an existing one the caller owns, or a new chat seeded
// with a title derived from this first message. None => an error was sent.
async fn resolve_chat(
    session: &mut actix_ws::Session,
    pool: &PgPool,
    user_id: Uuid,
    chat_id: Option<Uuid>,
    content: &str,
) -> Result<Option<Uuid>, actix_ws::Closed> {
    match chat_id {
        Some(id) => match db::chat::find_by_id(pool, id).await {
            Ok(Some(chat)) if chat.id_user == user_id => Ok(Some(chat.id_chat)),
            Ok(Some(_)) => {
                session.text(ws::error_frame("forbidden")).await?;
                Ok(None)
            }
            Ok(None) => {
                session.text(ws::error_frame("chat not found")).await?;
                Ok(None)
            }
            Err(e) => {
                eprintln!("chat lookup db error: {}", e);
                session.text(ws::error_frame("failed to load chat")).await?;
                Ok(None)
            }
        },
        None => match db::chat::create(pool, user_id, &derive_title(content)).await {
            Ok(chat) => Ok(Some(chat.id_chat)),
            Err(e) => {
                eprintln!("create chat db error: {}", e);
                session.text(ws::error_frame("failed to create chat")).await?;
                Ok(None)
            }
        },
    }
}

// Persist the user's message. Returns false (after sending an error) on failure.
async fn persist_user_message(
    session: &mut actix_ws::Session,
    pool: &PgPool,
    chat_id: Uuid,
    user_id: Uuid,
    content: &str,
    tokens: i32,
) -> Result<bool, actix_ws::Closed> {
    if let Err(e) = db::message::create_user_message(pool, chat_id, user_id, content, tokens).await
    {
        eprintln!("create user message db error: {}", e);
        session.text(ws::error_frame("failed to save message")).await?;
        return Ok(false);
    }
    Ok(true)
}

// Resolve the model to run, enforcing that a private model is only visible to an
// admin (reported as "not found" to others). None => an error was sent.
async fn resolve_model(
    session: &mut actix_ws::Session,
    pool: &PgPool,
    model_id: Option<Uuid>,
    is_admin: bool,
) -> Result<Option<db::model::Model>, actix_ws::Closed> {
    let model = match model_id {
        Some(id) => match db::model::find_by_id(pool, id).await {
            Ok(Some(model)) => model,
            Ok(None) => {
                session.text(ws::error_frame("model not found")).await?;
                return Ok(None);
            }
            Err(e) => {
                eprintln!("model lookup db error: {}", e);
                session.text(ws::error_frame("failed to load model")).await?;
                return Ok(None);
            }
        },
        None => {
            session.text(ws::error_frame("no model selected")).await?;
            return Ok(None);
        }
    };
    if !model.published && !is_admin {
        session.text(ws::error_frame("model not found")).await?;
        return Ok(None);
    }
    Ok(Some(model))
}


#[derive(Deserialize)]
struct CreateChatPayload {
    // The first message — used only to derive a provisional title here. The
    // message itself is NOT persisted by this endpoint; it's saved by the WS
    // turn that follows. Splitting chat creation from generation lets the client
    // navigate to the new chat (and start streaming the reply there) immediately
    // instead of waiting for the whole reply to finish.
    content: String,
}

#[post("")]
async fn create_chat(
    auth: AuthUser,
    payload: web::Json<CreateChatPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let content = payload.content.trim();
    if content.is_empty() {
        return HttpResponse::BadRequest().body("message content is empty");
    }
    if let Err(msg) = validate::max_len("message", content, validate::MAX_MESSAGE_LEN) {
        return HttpResponse::BadRequest().body(msg);
    }

    match db::chat::create(pool.get_ref(), auth.claims.sub, &derive_title(content)).await {
        Ok(chat) => HttpResponse::Created().json(ChatResponse::from(chat)),
        Err(e) => {
            eprintln!("create chat db error: {}", e);
            HttpResponse::InternalServerError().body("failed to create chat")
        }
    }
}

#[get("")]
async fn list_chats(auth: AuthUser, pool: web::Data<PgPool>) -> impl Responder {
    match db::chat::list_for_user(pool.get_ref(), auth.claims.sub).await {
        Ok(chats) => {
            let body: Vec<ChatResponse> = chats.into_iter().map(ChatResponse::from).collect();
            HttpResponse::Ok().json(body)
        }
        Err(e) => {
            eprintln!("list chats db error: {}", e);
            HttpResponse::InternalServerError().body("failed to list chats")
        }
    }
}

#[get("/{id}")]
async fn get_messages(
    auth: AuthUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id_chat = path.into_inner();

    if let Err(resp) = owned_chat(pool.get_ref(), id_chat, auth.claims.sub).await {
        return resp;
    }

    match db::message::list_for_chat(pool.get_ref(), id_chat).await {
        Ok(messages) => {
            let body: Vec<MessageResponse> =
                messages.into_iter().map(MessageResponse::from).collect();
            HttpResponse::Ok().json(body)
        }
        Err(e) => {
            eprintln!("list messages db error: {}", e);
            HttpResponse::InternalServerError().body("failed to load messages")
        }
    }
}

// `{"feedback": "up" | "down" | null}` — null clears the rating. The field is
// `Option` *and* `#[serde(default)]` so an omitted key and an explicit null both
// mean "un-rate", rather than the omission being a 400 the client has to avoid.
#[derive(Deserialize)]
struct FeedbackPayload {
    #[serde(default)]
    feedback: Option<db::message::Feedback>,
}

// Thumbs up/down on one assistant reply.
//
// Idempotent, hence PUT: the client sends the state it wants rather than a toggle,
// so a double-click or a retry can't leave the rating flipped from what the user
// sees. Mounted under /chat/message/… rather than /chat/{id}/… because a message id
// identifies its chat on its own — the client would otherwise have to carry a chat
// id it doesn't need.
#[put("/message/{id}/feedback")]
async fn set_message_feedback(
    auth: AuthUser,
    path: web::Path<Uuid>,
    payload: web::Json<FeedbackPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id_message = path.into_inner();

    let origin = match db::message::find_origin(pool.get_ref(), id_message).await {
        Ok(Some(origin)) => origin,
        Ok(None) => return HttpResponse::NotFound().body("message not found"),
        Err(e) => {
            eprintln!("message origin db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to load message");
        }
    };

    // Ownership is a property of the chat, not the message, so this is the same
    // owner-only check every other handler in this file runs.
    if let Err(resp) = owned_chat(pool.get_ref(), origin.id_chat, auth.claims.sub).await {
        return resp;
    }

    // Rating your own message would put a user turn into the preference export
    // labelled as if a model wrote it.
    if origin.id_user.is_some() {
        return HttpResponse::BadRequest().body("only an assistant reply can be rated");
    }

    match db::message::set_feedback(pool.get_ref(), id_message, payload.feedback).await {
        Ok(true) => HttpResponse::NoContent().finish(),
        Ok(false) => HttpResponse::NotFound().body("message not found"),
        Err(e) => {
            eprintln!("set message feedback db error: {}", e);
            HttpResponse::InternalServerError().body("failed to save feedback")
        }
    }
}

#[derive(Deserialize)]
struct UpdateChatPayload {
    title: String,
}

#[put("/{id}")]
async fn update_chat(
    auth: AuthUser,
    path: web::Path<Uuid>,
    payload: web::Json<UpdateChatPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id_chat = path.into_inner();

    let title = payload.title.trim();
    if title.is_empty() {
        return HttpResponse::BadRequest().body("title is empty");
    }

    if let Err(resp) = owned_chat(pool.get_ref(), id_chat, auth.claims.sub).await {
        return resp;
    }

    match db::chat::update_title(pool.get_ref(), id_chat, title).await {
        Ok(Some(chat)) => HttpResponse::Ok().json(ChatResponse::from(chat)),
        Ok(None) => HttpResponse::NotFound().body("chat not found"),
        Err(e) => {
            eprintln!("update chat db error: {}", e);
            HttpResponse::InternalServerError().body("failed to update chat")
        }
    }
}

#[delete("/{id}")]
async fn delete_chat(
    auth: AuthUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id_chat = path.into_inner();

    if let Err(resp) = owned_chat(pool.get_ref(), id_chat, auth.claims.sub).await {
        return resp;
    }

    match db::chat::delete_with_messages(pool.get_ref(), id_chat).await {
        Ok(true) => HttpResponse::NoContent().finish(),
        Ok(false) => HttpResponse::NotFound().body("chat not found"),
        Err(e) => {
            eprintln!("delete chat db error: {}", e);
            HttpResponse::InternalServerError().body("failed to delete chat")
        }
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    // chat_ws (GET /chat/ws) must be registered before get_messages (GET
    // /chat/{id}), otherwise "ws" would be captured by the "{id}" pattern.
    cfg.service(chat_ws);
    cfg.service(create_chat);
    cfg.service(list_chats);
    // Same reason as chat_ws: PUT /chat/message/{id}/feedback is two segments
    // deeper than PUT /chat/{id}, so it must be registered first to avoid "message"
    // being captured as a chat id.
    cfg.service(set_message_feedback);
    cfg.service(get_messages);
    cfg.service(update_chat);
    cfg.service(delete_chat);
}
