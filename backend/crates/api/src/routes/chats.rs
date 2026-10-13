//! Conversation persistence belongs to Rust; model preparation stays in Python.
use crate::upstream::{answer, caller, Answer};
use crate::{conversation as c, extractors::AuthUser};
use actix_web::{delete, get, patch, post, put, web, HttpResponse};
use db::conversation as queries;
use llm_engine::Proxy;
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::PgPool;
use uuid::Uuid;

fn title(body: &Value, default: bool) -> Result<&str, service::Error> {
    let value = body["title"]
        .as_str()
        .or(default.then_some("New chat"))
        .ok_or_else(|| c::failure(400, "title is required"))?;
    if value.chars().count() > 200 {
        return Err(c::failure(400, "title is too long"));
    }
    Ok(value)
}

#[get("")]
async fn list(user: AuthUser, pool: web::Data<PgPool>) -> Answer {
    let rows = queries::list(&pool, user.claims.sub)
        .await
        .map_err(c::database)?;
    answer(json!({"chats":rows.iter().map(c::chat).collect::<Vec<_>>()}))
}
#[post("")]
async fn create(user: AuthUser, body: web::Json<Value>, pool: web::Data<PgPool>) -> Answer {
    let row = queries::create(&pool, user.claims.sub, title(&body, true)?)
        .await
        .map_err(c::database)?;
    Ok(HttpResponse::Created().json(c::chat(&row)))
}
#[get("/{chat_id}")]
async fn get_one(user: AuthUser, id: web::Path<Uuid>, pool: web::Data<PgPool>) -> Answer {
    let row = c::owned(&pool, user.claims.sub, *id).await?;
    let mut result = c::chat(&row);
    result["messages"] = json!(c::history(&pool, user.claims.sub, *id).await?);
    answer(result)
}
#[derive(Deserialize)]
struct Floor {
    model_name: String,
}
#[get("/{chat_id}/context-floor")]
async fn context_floor(
    user: AuthUser,
    id: web::Path<Uuid>,
    query: web::Query<Floor>,
    pool: web::Data<PgPool>,
    engine: web::Data<Proxy>,
) -> Answer {
    c::owned(&pool, user.claims.sub, *id).await?;
    let messages = queries::messages(&pool, user.claims.sub, *id)
        .await
        .map_err(c::database)?;
    let turns: Vec<_> = messages
        .iter()
        .map(|m| json!({"role":m.role,"text":m.content}))
        .collect();
    let floor = engine
        .post(
            &caller(&user),
            "/chat/context-floor",
            json!({"model_name":query.model_name,"turns":turns}),
        )
        .await?;
    let start = floor["start"]
        .as_u64()
        .ok_or_else(|| c::failure(502, "invalid context floor"))? as usize;
    let start = start.min(messages.len());
    answer(
        json!({"floor_message_id":messages.get(start).map(|m| m.id),"pruned_turns":start,
        "pruned_tokens":messages[..start].iter().map(|m| i64::from(m.tokens)).sum::<i64>()}),
    )
}
#[patch("/{chat_id}")]
async fn rename(
    user: AuthUser,
    id: web::Path<Uuid>,
    body: web::Json<Value>,
    pool: web::Data<PgPool>,
) -> Answer {
    if !queries::rename(&pool, user.claims.sub, *id, title(&body, false)?)
        .await
        .map_err(c::database)?
    {
        return Err(c::failure(404, "conversation not found").into());
    }
    answer(c::chat(&c::owned(&pool, user.claims.sub, *id).await?))
}
#[delete("/{chat_id}")]
async fn remove(user: AuthUser, id: web::Path<Uuid>, pool: web::Data<PgPool>) -> Answer {
    if !queries::remove(&pool, user.claims.sub, *id)
        .await
        .map_err(c::database)?
    {
        return Err(c::failure(404, "conversation not found").into());
    }
    Ok(HttpResponse::NoContent().finish())
}
#[post("/{chat_id}/messages")]
async fn ask(
    user: AuthUser,
    id: web::Path<Uuid>,
    body: web::Json<Value>,
    pool: web::Data<PgPool>,
    engine: web::Data<Proxy>,
) -> Answer {
    let _guard = c::lock(*id).await;
    let round = c::prepare(&pool, &engine, user.claims.sub, *id, body.into_inner()).await?;
    let reply = engine
        .post(&caller(&user), "/chat", round.config.clone())
        .await?;
    Ok(HttpResponse::Created().json(c::finish(&pool, user.claims.sub, *id, &round, &reply).await?))
}
#[put("/{message_id}/feedback")]
async fn rate(
    user: AuthUser,
    id: web::Path<Uuid>,
    body: web::Json<Value>,
    pool: web::Data<PgPool>,
) -> Answer {
    let feedback = body["feedback"].as_str();
    if !body["feedback"].is_null() && !matches!(feedback, Some("up" | "down")) {
        return Err(c::failure(400, "invalid feedback").into());
    }
    if !queries::feedback(&pool, user.claims.sub, *id, feedback)
        .await
        .map_err(c::database)?
    {
        return Err(c::failure(404, "message not found").into());
    }
    answer(json!({"message_id":*id,"feedback":feedback}))
}
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list)
        .service(create)
        .service(get_one)
        .service(context_floor)
        .service(rename)
        .service(remove)
        .service(ask);
}
pub fn configure_messages(cfg: &mut web::ServiceConfig) {
    cfg.service(rate);
}
