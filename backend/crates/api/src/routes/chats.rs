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

/// Validate a chat title's 200-character limit, supplying the legacy default only on creation.
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

/// Parse optional membership on creation or require an explicit UUID/null on assignment.
fn project_id(body: &Value, required: bool) -> Result<Option<Uuid>, service::Error> {
    let value = body.get("project_id");
    if required && value.is_none() {
        return Err(c::failure(400, "project_id is required"));
    }
    serde_json::from_value(value.cloned().unwrap_or(Value::Null))
        .map_err(|_| c::failure(400, "invalid project_id"))
}

#[derive(Deserialize)]
struct ListQuery {
    limit: Option<i64>,
}

/// List owned chats, with an optional bounded recent slice for the sidebar.
#[get("")]
async fn list(user: AuthUser, query: web::Query<ListQuery>, pool: web::Data<PgPool>) -> Answer {
    let rows = match query.limit {
        Some(limit) if (1..=100).contains(&limit) => {
            queries::list_recent(&pool, user.claims.sub, limit).await
        }
        Some(_) => return Err(c::failure(400, "invalid conversation limit").into()),
        None => queries::list(&pool, user.claims.sub).await,
    }
    .map_err(c::database)?;
    answer(json!({"chats":rows.iter().map(c::chat).collect::<Vec<_>>()}))
}
/// Create one owned chat with optional owned project membership in the same database insert.
#[post("")]
async fn create(user: AuthUser, body: web::Json<Value>, pool: web::Data<PgPool>) -> Answer {
    let row = queries::create_in_project(
        &pool,
        user.claims.sub,
        title(&body, true)?,
        project_id(&body, false)?,
    )
    .await
    .map_err(c::database)?;
    Ok(HttpResponse::Created().json(c::chat(&row)))
}
/// Return owned chat metadata and persisted messages without requiring an inference service.
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
/// Ask the engine to fit authorized history with the same stored project context used for generation.
#[get("/{chat_id}/context-floor")]
async fn context_floor(
    user: AuthUser,
    id: web::Path<Uuid>,
    query: web::Query<Floor>,
    pool: web::Data<PgPool>,
    engine: web::Data<Proxy>,
) -> Answer {
    let _guard = c::lock(*id).await;
    c::owned(&pool, user.claims.sub, *id).await?;
    let messages = queries::messages(&pool, user.claims.sub, *id)
        .await
        .map_err(c::database)?;
    let turns: Vec<_> = messages
        .iter()
        .map(|m| json!({"role":m.role,"text":m.content}))
        .collect();
    let config = c::context_request(
        &pool,
        user.claims.sub,
        *id,
        &messages,
        json!({"model_name":query.model_name,"turns":turns}),
    )
    .await?;
    let floor = engine
        .post(&caller(&user), "/chat/context-floor", config)
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
/// Rename an owned chat and return its public metadata.
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
/// Move or detach an owned chat under its turn lease, excluding membership changes during generation.
#[put("/{chat_id}/project")]
async fn assign_project(
    user: AuthUser,
    id: web::Path<Uuid>,
    body: web::Json<Value>,
    pool: web::Data<PgPool>,
) -> Answer {
    let project = project_id(&body, true)?;
    let _guard = c::lock(*id).await;
    let row = queries::assign_project(&pool, user.claims.sub, *id, project)
        .await
        .map_err(c::database)?
        .ok_or_else(|| c::failure(404, "conversation or project not found"))?;
    answer(c::chat(&row))
}

/// Delete an owned chat together with its cascading local messages and files.
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
/// Prepare and store an authorized turn, then commit its completed assistant response before returning 201.
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
/// Set or clear valid feedback only on a message owned by the authenticated user.
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
/// Register owned conversation CRUD, membership, context sizing and generation routes.
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list)
        .service(create)
        .service(get_one)
        .service(context_floor)
        .service(rename)
        .service(assign_project)
        .service(remove)
        .service(ask);
}
/// Register feedback on the independent message-resource scope.
pub fn configure_messages(cfg: &mut web::ServiceConfig) {
    cfg.service(rate);
}
