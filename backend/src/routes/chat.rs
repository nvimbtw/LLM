use actix_web::{HttpRequest, HttpResponse, Responder, delete, get, post, put, web};
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::{db, extractors::AuthUser, ws};

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
    let mut user_id: Option<Uuid> = None;

    while let Some(Ok(msg)) = stream.next().await {
        match msg {
            actix_ws::Message::Text(text) => {
                // Message::Text carries a ByteString; bind it as &str once so the
                // helpers below get the type they expect.
                let text: &str = &text;
                match user_id {
                    // Not yet authenticated: this frame must be the auth token.
                    None => match ws::authenticate(text, &secret) {
                        Some(uid) => {
                            user_id = Some(uid);
                            let _ = session
                                .text(serde_json::json!({ "type": "ready" }).to_string())
                                .await;
                        }
                        None => {
                            let _ = session.text(ws::error_frame("authentication failed")).await;
                            let _ = session.close(None).await;
                            return;
                        }
                    },
                    // Authenticated: handle a chat turn, streaming tokens back.
                    Some(uid) => {
                        if handle_message(&mut session, &pool, uid, text)
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
                let _ = session.pong(&bytes).await;
            }
            actix_ws::Message::Close(_) => break,
            _ => {}
        }
    }

    let _ = session.close(None).await;
}

// Processes one chat turn: resolve/create the chat, persist the user message,
// stream the assistant reply word-by-word, then persist the full reply.
// Returns Err only when sending to the socket fails (client disconnected), so
// the caller can stop the session.
async fn handle_message(
    session: &mut actix_ws::Session,
    pool: &PgPool,
    user_id: Uuid,
    text: &str,
) -> Result<(), actix_ws::Closed> {
    let frame: SendMessageFrame = match serde_json::from_str(text) {
        Ok(frame) => frame,
        Err(_) => return session.text(ws::error_frame("invalid message frame")).await,
    };

    let content = frame.content.trim();
    if content.is_empty() {
        return session.text(ws::error_frame("message content is empty")).await;
    }

    // Resolve the target chat: an existing one the caller owns, or a new chat
    // seeded with a title derived from this first message.
    let chat_id = match frame.chat_id {
        Some(id) => match db::chat::find_by_id(pool, id).await {
            Ok(Some(chat)) if chat.id_user == user_id => chat.id_chat,
            Ok(Some(_)) => return session.text(ws::error_frame("forbidden")).await,
            Ok(None) => return session.text(ws::error_frame("chat not found")).await,
            Err(e) => {
                eprintln!("chat lookup db error: {}", e);
                return session.text(ws::error_frame("failed to load chat")).await;
            }
        },
        None => match db::chat::create(pool, user_id, &derive_title(content)).await {
            Ok(chat) => chat.id_chat,
            Err(e) => {
                eprintln!("create chat db error: {}", e);
                return session.text(ws::error_frame("failed to create chat")).await;
            }
        },
    };

    // Persist the user's message.
    if let Err(e) = db::message::create_user_message(pool, chat_id, user_id, content).await {
        eprintln!("create user message db error: {}", e);
        return session.text(ws::error_frame("failed to save message")).await;
    }

    // Resolve the model to run. Its weights live on disk under its name; the
    // inference engine loads (and caches) them on first use.
    let model = match frame.model_id {
        Some(id) => match db::model::find_by_id(pool, id).await {
            Ok(Some(model)) => model,
            Ok(None) => return session.text(ws::error_frame("model not found")).await,
            Err(e) => {
                eprintln!("model lookup db error: {}", e);
                return session.text(ws::error_frame("failed to load model")).await;
            }
        },
        None => return session.text(ws::error_frame("no model selected")).await,
    };

    // Inference is blocking GPU work (CUDA), so it must not run on the async
    // executor — it runs on a dedicated thread. Decoded token text comes back
    // over `tok_rx` and is forwarded to the socket as it arrives; `done_rx`
    // carries the final outcome (full text, or an error) once generation ends.
    let model_dir = crate::storage::models_dir().join(&model.name);
    let prompt = content.to_string();
    let (tok_tx, mut tok_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Result<String, String>>();
    std::thread::spawn(move || {
        let result = llm::infer::generate_response_streaming(
            &model_dir,
            &prompt,
            MAX_TOKENS,
            TEMPERATURE,
            TOP_K,
            |token| {
                let _ = tok_tx.send(token.to_string());
            },
        );
        let _ = done_tx.send(result);
    });

    let mut reply = String::new();
    while let Some(token) = tok_rx.recv().await {
        reply.push_str(&token);
        let out = serde_json::json!({ "type": "token", "value": token }).to_string();
        session.text(out).await?;
    }

    // Generation finished: prefer the authoritative full text, and surface a
    // failure instead of persisting an empty reply. done_rx only errs if the
    // worker thread panicked.
    match done_rx.await {
        Ok(Ok(full)) => reply = full,
        Ok(Err(e)) => {
            eprintln!("inference error: {}", e);
            return session.text(ws::error_frame("inference failed")).await;
        }
        Err(_) => {
            eprintln!("inference thread panicked");
            return session.text(ws::error_frame("inference failed")).await;
        }
    }

    // Persist the full assistant reply, attributed to the model that produced it.
    let assistant =
        match db::message::create_assistant_message(pool, chat_id, Some(model.id_model), &reply)
            .await
        {
            Ok(message) => message,
            Err(e) => {
                eprintln!("create assistant message db error: {}", e);
                return session.text(ws::error_frame("failed to save response")).await;
            }
        };

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
