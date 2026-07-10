use actix_web::{HttpRequest, HttpResponse, Responder, delete, get, post, put, web};
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::routes::validate;
use crate::{db, db::user::AccessLevel, extractors::AuthUser, ws};

#[derive(Serialize)]
struct ChatResponse {
    id_chat: Uuid,
    title: String,
    created_at: DateTime<Utc>,
}

impl From<db::chat::Chat> for ChatResponse {
    fn from(chat: db::chat::Chat) -> Self {
        ChatResponse {
            id_chat: chat.id_chat,
            title: chat.title,
            created_at: chat.created_at,
        }
    }
}

#[derive(Serialize)]
struct MessageResponse {
    id_message: Uuid,
    role: &'static str,
    content: String,
    created_at: DateTime<Utc>,
}

impl From<db::message::Message> for MessageResponse {
    fn from(message: db::message::Message) -> Self {
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
}

// Sampling knobs for a chat turn. A reply runs until the model emits
// <|endoftext|>; MAX_TOKENS is only a safety ceiling so a model that never emits
// it can't generate forever (the sliding context window means it otherwise
// wouldn't stop on its own).
const MAX_TOKENS: usize = 4096;
const TEMPERATURE: f32 = 0.8;
const TOP_K: u32 = 40;

// WebSocket entrypoint. The handshake is a GET upgrade; we hand the socket off
// to a spawned task and return the upgrade response immediately. Auth happens
// inside the session (first frame), not here, since the browser can't send the
// Authorization header on the handshake.
#[get("/ws")]
async fn chat_ws(
    req: HttpRequest,
    body: web::Payload,
    pool: web::Data<PgPool>,
    jwt_secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, msg_stream) = actix_ws::handle(&req, body)?;

    // Owned clones for the 'static task: PgPool is an Arc internally (cheap to
    // clone), and we copy the secret string.
    let pool = pool.get_ref().clone();
    let secret = jwt_secret.get_ref().clone();

    actix_web::rt::spawn(chat_session(session, msg_stream, pool, secret));

    Ok(response)
}

// Drives one socket for its lifetime: authenticate on the first text frame,
// then treat every later text frame as a chat turn.
async fn chat_session(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    pool: PgPool,
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
                        if handle_message(&mut session, &pool, uid, is_admin, text)
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

// Processes one chat turn: resolve/create the chat, persist the user message,
// stream the assistant reply word-by-word, then persist the full reply. Each
// stage below sends its own error frame and signals "stop this turn" when it
// fails, so this stays a readable sequence. Returns Err only when sending to the
// socket fails (client disconnected), so the caller can stop the session.
async fn handle_message(
    session: &mut actix_ws::Session,
    pool: &PgPool,
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

    let Some(chat_id) = resolve_chat(session, pool, user_id, frame.chat_id, content).await? else {
        return Ok(());
    };
    if !persist_user_message(session, pool, chat_id, user_id, content).await? {
        return Ok(());
    }
    let Some(model) = resolve_model(session, pool, frame.model_id, is_admin).await? else {
        return Ok(());
    };
    let Some(history) = load_history(session, pool, chat_id).await? else {
        return Ok(());
    };

    let model_dir = crate::storage::models_dir().join(model.id_model.to_string());
    let Some(reply) = stream_reply(session, &model_dir, history).await? else {
        return Ok(());
    };

    finalize_turn(session, pool, user_id, chat_id, model.id_model, reply).await
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
) -> Result<bool, actix_ws::Closed> {
    if let Err(e) = db::message::create_user_message(pool, chat_id, user_id, content).await {
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

// Load the whole conversation (oldest-first, including the user message just
// persisted) as template turns. The inference layer renders and truncates them
// to the model's context window. None => an error was sent.
async fn load_history(
    session: &mut actix_ws::Session,
    pool: &PgPool,
    chat_id: Uuid,
) -> Result<Option<Vec<llm::chat_template::Turn>>, actix_ws::Closed> {
    let messages = match db::message::list_for_chat(pool, chat_id).await {
        Ok(messages) => messages,
        Err(e) => {
            eprintln!("list messages db error: {}", e);
            session.text(ws::error_frame("failed to load history")).await?;
            return Ok(None);
        }
    };
    let history = messages
        .into_iter()
        .map(|m| llm::chat_template::Turn {
            // id_user set => a person sent it; otherwise it's an assistant reply.
            role: if m.id_user.is_some() {
                llm::chat_template::Role::User
            } else {
                llm::chat_template::Role::Assistant
            },
            text: m.content,
        })
        .collect();
    Ok(Some(history))
}

// The streamed reply plus its token counts, produced by one generation.
struct StreamedReply {
    text: String,
    input_tokens: usize,
    output_tokens: usize,
}

// Run inference on a dedicated thread (CUDA is blocking and must not run on the
// async executor), forwarding decoded tokens to the socket as they arrive. None
// => generation failed and an error was sent.
async fn stream_reply(
    session: &mut actix_ws::Session,
    model_dir: &std::path::Path,
    history: Vec<llm::chat_template::Turn>,
) -> Result<Option<StreamedReply>, actix_ws::Closed> {
    let model_dir = model_dir.to_path_buf();
    let (tok_tx, mut tok_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (done_tx, done_rx) =
        tokio::sync::oneshot::channel::<Result<llm::infer::GenerationOutput, String>>();
    std::thread::spawn(move || {
        // System prompt slot is supported by the template; passing None until the
        // SFT data includes system turns, so we don't spend the tiny context
        // window on text the model was never trained to act on.
        let result = llm::infer::generate_chat_response_streaming(
            &model_dir,
            None,
            &history,
            MAX_TOKENS,
            TEMPERATURE,
            TOP_K,
            |token| {
                // Best-effort: if the receiver was dropped (client disconnected
                // or the turn aborted) the token is discarded and generation
                // winds down on its own — nothing here can act on the failure.
                let _ = tok_tx.send(token.to_string());
            },
        );
        // Same: the oneshot receiver may be gone if the turn was abandoned.
        let _ = done_tx.send(result);
    });

    while let Some(token) = tok_rx.recv().await {
        let out = serde_json::json!({ "type": "token", "value": token }).to_string();
        session.text(out).await?;
    }

    // Prefer the authoritative full text over the streamed concatenation, and
    // surface a failure instead of persisting an empty reply. done_rx only errs
    // if the worker thread panicked.
    match done_rx.await {
        Ok(Ok(out)) => Ok(Some(StreamedReply {
            text: out.text,
            input_tokens: out.input_tokens,
            output_tokens: out.output_tokens,
        })),
        Ok(Err(e)) => {
            eprintln!("inference error: {}", e);
            session.text(ws::error_frame("inference failed")).await?;
            Ok(None)
        }
        Err(_) => {
            eprintln!("inference thread panicked");
            session.text(ws::error_frame("inference failed")).await?;
            Ok(None)
        }
    }
}

// Persist the assistant reply, record usage (non-fatal), and send the done frame.
async fn finalize_turn(
    session: &mut actix_ws::Session,
    pool: &PgPool,
    user_id: Uuid,
    chat_id: Uuid,
    model_id: Uuid,
    reply: StreamedReply,
) -> Result<(), actix_ws::Closed> {
    let assistant =
        match db::message::create_assistant_message(pool, chat_id, Some(model_id), &reply.text)
            .await
        {
            Ok(message) => message,
            Err(e) => {
                eprintln!("create assistant message db error: {}", e);
                return session.text(ws::error_frame("failed to save response")).await;
            }
        };

    // Non-fatal: a tracking failure shouldn't sink a reply the user already
    // received and we already persisted.
    if let Err(e) = db::usage::record(
        pool,
        user_id,
        Some(model_id),
        reply.input_tokens as i32,
        reply.output_tokens as i32,
    )
    .await
    {
        eprintln!("record usage db error: {}", e);
    }

    let done = serde_json::json!({
        "type": "done",
        "chat_id": chat_id,
        "id_message": assistant.id_message,
    })
    .to_string();
    session.text(done).await
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
    cfg.service(get_messages);
    cfg.service(update_chat);
    cfg.service(delete_chat);
}
