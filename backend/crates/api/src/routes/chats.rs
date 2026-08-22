//! Conversations. The engine owns them; this hop proves who is asking.

use actix_web::{HttpRequest, HttpResponse, delete, get, patch, post, put, web};
use engine::Proxy;
use serde_json::Value;
use uuid::Uuid;

use crate::extractors::AuthUser;
use crate::upstream::{Answer, answer, caller, with_query};

#[get("")]
async fn list(user: AuthUser, engine: web::Data<Proxy>) -> Answer {
    answer(engine.get(&caller(&user), "/chats").await?)
}

#[post("")]
async fn create(user: AuthUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    let created = engine.post(&caller(&user), "/chats", body.into_inner()).await?;
    Ok(HttpResponse::Created().json(created))
}

#[get("/{chat_id}")]
async fn get_one(user: AuthUser, chat_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(engine.get(&caller(&user), &format!("/chats/{chat_id}")).await?)
}

/// Where `/prune` (the frontend command) would cut this chat's history — read
/// only, on both sides of this hop. See `context_floor` in the engine's
/// `api/routes/chats.py` for what the answer means and why nothing is written
/// down anywhere to produce it.
#[get("/{chat_id}/context-floor")]
async fn context_floor(
    user: AuthUser,
    chat_id: web::Path<Uuid>,
    req: HttpRequest,
    engine: web::Data<Proxy>,
) -> Answer {
    let path = with_query(&format!("/chats/{chat_id}/context-floor"), &req);
    answer(engine.get(&caller(&user), &path).await?)
}

#[patch("/{chat_id}")]
async fn rename(
    user: AuthUser,
    chat_id: web::Path<Uuid>,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    answer(engine.patch(&caller(&user), &format!("/chats/{chat_id}"), body.into_inner()).await?)
}

#[delete("/{chat_id}")]
async fn remove(user: AuthUser, chat_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    engine.delete(&caller(&user), &format!("/chats/{chat_id}")).await?;
    Ok(HttpResponse::NoContent().finish())
}

#[post("/{chat_id}/messages")]
async fn ask(
    user: AuthUser,
    chat_id: web::Path<Uuid>,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    let path = format!("/chats/{chat_id}/messages");
    let reply = engine.post(&caller(&user), &path, body.into_inner()).await?;
    Ok(HttpResponse::Created().json(reply))
}

#[put("/{message_id}/feedback")]
async fn rate(
    user: AuthUser,
    message_id: web::Path<Uuid>,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    let path = format!("/messages/{message_id}/feedback");
    answer(engine.put(&caller(&user), &path, body.into_inner()).await?)
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list);
    cfg.service(create);
    cfg.service(get_one);
    cfg.service(context_floor);
    cfg.service(rename);
    cfg.service(remove);
    cfg.service(ask);
}

/// Mounted at `/messages`: a rating is aimed at a turn, not at its conversation.
pub fn configure_messages(cfg: &mut web::ServiceConfig) {
    cfg.service(rate);
}
