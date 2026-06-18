use actix_web::{HttpResponse, Responder, delete, get, post, put, web};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::{db, extractors::AuthUser};

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

#[derive(Deserialize)]
struct SendMessagePayload {
    // Absent => start a new chat; present => append to an existing one.
    chat_id: Option<Uuid>,
    content: String,
}

#[derive(Serialize)]
struct SendMessageResponse {
    // Always echoed back so the client learns the id of a freshly created chat.
    chat_id: Uuid,
    message: MessageResponse,
}

#[post("")]
async fn send_message(
    auth: AuthUser,
    payload: web::Json<SendMessagePayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let content = payload.content.trim();
    if content.is_empty() {
        return HttpResponse::BadRequest().body("message content is empty");
    }

    let user_id = auth.claims.sub;

    // Resolve the target chat: an existing one the caller owns, or a brand-new
    // chat seeded with a title derived from this first message.
    let chat_id = match payload.chat_id {
        Some(id) => match owned_chat(pool.get_ref(), id, user_id).await {
            Ok(chat) => chat.id_chat,
            Err(resp) => return resp,
        },
        None => match db::chat::create(pool.get_ref(), user_id, &derive_title(content)).await {
            Ok(chat) => chat.id_chat,
            Err(e) => {
                eprintln!("create chat db error: {}", e);
                return HttpResponse::InternalServerError().body("failed to create chat");
            }
        },
    };

    // Persist the user's message.
    if let Err(e) =
        db::message::create_user_message(pool.get_ref(), chat_id, user_id, content).await
    {
        eprintln!("create user message db error: {}", e);
        return HttpResponse::InternalServerError().body("failed to save message");
    }

    // TODO: create model response — feed the chat history to the model and store
    // its generated text (and the model's id) instead of this placeholder.
    // The sleep fakes generation latency; it's an async (non-blocking) sleep so
    // the worker thread stays free for other requests while we wait.
    actix_web::rt::time::sleep(std::time::Duration::from_secs(5)).await;
    // Placeholder that mimics a real model's structured-token output. The
    // <|...|> markers are the generated control tokens the frontend parses into
    // rich content — see the token reference accompanying this code. Real model
    // text replaces this string once integration lands.
    let reply = r#"<|heading 1|>Reading a File in Rust<|/heading 1|>
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

    let assistant =
        match db::message::create_assistant_message(pool.get_ref(), chat_id, reply).await {
            Ok(message) => message,
            Err(e) => {
                eprintln!("create assistant message db error: {}", e);
                return HttpResponse::InternalServerError().body("failed to save response");
            }
        };

    HttpResponse::Ok().json(SendMessageResponse {
        chat_id,
        message: MessageResponse::from(assistant),
    })
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
    cfg.service(send_message);
    cfg.service(list_chats);
    cfg.service(get_messages);
    cfg.service(update_chat);
    cfg.service(delete_chat);
}
