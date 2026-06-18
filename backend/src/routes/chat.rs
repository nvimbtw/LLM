use actix_web::{HttpRequest, HttpResponse, Responder, delete, get, put, web};
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::{crypto, db, extractors::AuthUser};

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

// The very first frame a client sends over the socket: the encrypted token,
// same value the HTTP routes pass as `Authorization: Bearer`. We auth here
// because browsers can't set headers on a WebSocket handshake.
#[derive(Deserialize)]
struct AuthFrame {
    token: String,
}

// Every subsequent inbound frame is a chat turn.
#[derive(Deserialize)]
struct SendMessageFrame {
    // Absent => start a new chat; present => append to an existing one.
    chat_id: Option<Uuid>,
    content: String,
}

// Placeholder that mimics a real model's structured-token output. The <|...|>
// markers are the generated control tokens the frontend parses into rich
// content. Real model text replaces this once inference integration lands.
const PLACEHOLDER_REPLY: &str = r#"<|thinking | 'Thinking about the scope of the question'|>
<|thinking | 'Recalling the relevant standard library file APIs'|>
<|thinking | 'Weighing read_to_string against a buffered reader'|>
<|thinking | 'Considering error handling with Result and the question mark operator'|>
<|thinking | 'Deciding whether to also show the raw bytes variant fs::read'|>
<|thinking | 'Sketching a minimal runnable example that compiles'|>
<|thinking | 'Noting the memory cost of reading a whole file at once'|>
<|thinking | 'Deciding which caveats are worth flagging to the user'|>
<|thinking | 'Picking a documentation link to point at'|>
<|thinking | 'Settling on the final structure of the answer'|>
<|heading 1|>Reading a File in Rust<|/heading 1|>
<|paragraph|>Here is a quick overview of how to read a file into a <|code inline | lang='rs'|>String<|/code inline|> using only the standard library. It is one of the most common <|bold|>I/O<|/bold|> tasks you will run into.<|/paragraph|>
<|heading 2|>Steps<|/heading 2|>
<|ordered list|>
<|item|>Bring <|code inline | lang='rs'|>std::fs<|/code inline|> into scope.<|/item|>
<|item|>Call <|code inline | lang='rs'|>fs::read_to_string<|/code inline|> with your path.<|/item|>
<|item|>Handle the <|code inline | lang='rs'|>Result<|/code inline|> it returns.<|/item|>
<|/ordered list|>
<|heading 2|>Example<|/heading 2|>
<|code block | lang='rs'|>use std::fs;

fn main() -> std::io::Result<()> {
    let contents = fs::read_to_string("hello.txt")?;
    println!("{contents}");
    Ok(())
}<|/code block|>
<|divider|>
<|paragraph|>A few things worth remembering: this reads the <|italic|>entire<|/italic|> file into memory, so for large inputs you should <|inline|>prefer a streaming approach<|/inline|> instead.<|/paragraph|>
<|list|>
<|item|>Use <|code inline | lang='rs'|>BufReader<|/code inline|> for line-by-line reading.<|/item|>
<|item|>Use <|code inline | lang='rs'|>fs::read<|/code inline|> when you need the raw bytes.<|/item|>
<|/list|>
<|paragraph|>For the full API, see the <|link | href='https://doc.rust-lang.org/std/fs/fn.read_to_string.html'|>official documentation<|/link|>.<|/paragraph|>
<|quote|>Tip: handle errors explicitly rather than unwrapping in production code.<|/quote|>"#;

// Delay between streamed tokens, to fake generation latency.
const TOKEN_DELAY_MS: u64 = 150;

fn error_frame(message: &str) -> String {
    serde_json::json!({ "type": "error", "message": message }).to_string()
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
                    None => match authenticate(text, &secret) {
                        Some(uid) => {
                            user_id = Some(uid);
                            let _ = session
                                .text(serde_json::json!({ "type": "ready" }).to_string())
                                .await;
                        }
                        None => {
                            let _ = session.text(error_frame("authentication failed")).await;
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

// Decrypts + verifies the token frame, yielding the caller's user id. None on
// any failure (bad JSON, bad/expired token) — same all-or-nothing stance as
// the AuthUser extractor.
fn authenticate(text: &str, secret: &str) -> Option<Uuid> {
    let frame: AuthFrame = serde_json::from_str(text).ok()?;
    let jwt = crypto::token_cipher::decrypt(&frame.token, secret)?;
    let claims = crypto::jwt::verify_token(&jwt, secret).ok()?;
    Some(claims.sub)
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
        Err(_) => return session.text(error_frame("invalid message frame")).await,
    };

    let content = frame.content.trim();
    if content.is_empty() {
        return session.text(error_frame("message content is empty")).await;
    }

    // Resolve the target chat: an existing one the caller owns, or a new chat
    // seeded with a title derived from this first message.
    let chat_id = match frame.chat_id {
        Some(id) => match db::chat::find_by_id(pool, id).await {
            Ok(Some(chat)) if chat.id_user == user_id => chat.id_chat,
            Ok(Some(_)) => return session.text(error_frame("forbidden")).await,
            Ok(None) => return session.text(error_frame("chat not found")).await,
            Err(e) => {
                eprintln!("chat lookup db error: {}", e);
                return session.text(error_frame("failed to load chat")).await;
            }
        },
        None => match db::chat::create(pool, user_id, &derive_title(content)).await {
            Ok(chat) => chat.id_chat,
            Err(e) => {
                eprintln!("create chat db error: {}", e);
                return session.text(error_frame("failed to create chat")).await;
            }
        },
    };

    // Persist the user's message.
    if let Err(e) = db::message::create_user_message(pool, chat_id, user_id, content).await {
        eprintln!("create user message db error: {}", e);
        return session.text(error_frame("failed to save message")).await;
    }

    // TODO: create model response — stream real tokens from the model here
    // instead of chopping a fixed placeholder on whitespace. "token = one word"
    // is a stand-in; a real tokenizer won't split the <|...|> control tokens.
    let reply = PLACEHOLDER_REPLY;
    for word in reply.split_whitespace() {
        actix_web::rt::time::sleep(std::time::Duration::from_millis(TOKEN_DELAY_MS)).await;
        let token = serde_json::json!({ "type": "token", "value": word }).to_string();
        session.text(token).await?;
    }

    // Persist the full assistant reply (not the chunked stream).
    let assistant = match db::message::create_assistant_message(pool, chat_id, reply).await {
        Ok(message) => message,
        Err(e) => {
            eprintln!("create assistant message db error: {}", e);
            return session.text(error_frame("failed to save response")).await;
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
    cfg.service(list_chats);
    cfg.service(get_messages);
    cfg.service(update_chat);
    cfg.service(delete_chat);
}
