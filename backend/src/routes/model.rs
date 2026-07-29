use actix_web::{HttpResponse, Responder, delete, get, put, web};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::PgPool;
use std::fs;
use uuid::Uuid;

use db;
use db::user::AccessLevel;
use crate::routes::validate;
use crate::extractors::{AdminUser, AuthUser};
use storage;

// Response shape sent to clients. Kept separate from the db `Model` struct per
// the layering rules, even though they currently carry the same fields.
#[derive(Serialize)]
struct ModelResponse {
    id_model: Uuid,
    name: String,
    description: Option<String>,
    parameters: Option<Value>,
    context: Option<i32>,
    family: Option<String>,
    published: bool,
    created_at: DateTime<Utc>,
}

impl From<db::model::Model> for ModelResponse {
    fn from(m: db::model::Model) -> Self {
        ModelResponse {
            id_model: m.id_model,
            name: m.name,
            description: m.description,
            parameters: m.parameters,
            context: m.context,
            family: m.family,
            published: m.published,
            created_at: m.created_at,
        }
    }
}

// Any authenticated user may browse the registry — they need it to pick a model
// in chat. Non-admins only see published models; admins see all. Mutating routes
// below require admin.
#[get("")]
async fn list_models(auth: AuthUser, pool: web::Data<PgPool>) -> impl Responder {
    let only_published = !matches!(auth.claims.access_level, AccessLevel::Admin);
    match db::model::list(pool.get_ref(), only_published).await {
        Ok(models) => {
            let body: Vec<ModelResponse> = models.into_iter().map(ModelResponse::from).collect();
            HttpResponse::Ok().json(body)
        }
        Err(e) => {
            eprintln!("list models db error: {}", e);
            HttpResponse::InternalServerError().body("failed to list models")
        }
    }
}

#[get("/{id}")]
async fn get_model(auth: AuthUser, path: web::Path<Uuid>, pool: web::Data<PgPool>) -> impl Responder {
    let is_admin = matches!(auth.claims.access_level, AccessLevel::Admin);
    match db::model::find_by_id(pool.get_ref(), path.into_inner()).await {
        // A private model is invisible to non-admins — report it as not found
        // rather than forbidden so its existence isn't leaked.
        Ok(Some(model)) if model.published || is_admin => {
            HttpResponse::Ok().json(ModelResponse::from(model))
        }
        Ok(Some(_)) => HttpResponse::NotFound().body("model not found"),
        Ok(None) => HttpResponse::NotFound().body("model not found"),
        Err(e) => {
            eprintln!("get model db error: {}", e);
            HttpResponse::InternalServerError().body("failed to get model")
        }
    }
}

#[derive(Deserialize)]
struct UpdateModelPayload {
    name: Option<String>,
    description: Option<String>,
    // Free-text family the model belongs to; assigning a previously-unused name
    // is how a "new family" gets created (families aren't a separate table).
    family: Option<String>,
    // Publish (true) / unpublish (false) the model.
    published: Option<bool>,
}

#[put("/{id}")]
async fn update_model(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    payload: web::Json<UpdateModelPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    if let Some(msg) = payload
        .name
        .as_deref()
        .and_then(|n| validate::field_len("name", n, 1, validate::MAX_MODEL_NAME_LEN).err())
    {
        return HttpResponse::BadRequest().body(msg);
    }
    if let Some(msg) = payload
        .description
        .as_deref()
        .and_then(|d| validate::max_len("description", d, validate::MAX_MODEL_DESC_LEN).err())
    {
        return HttpResponse::BadRequest().body(msg);
    }

    match db::model::update(
        pool.get_ref(),
        path.into_inner(),
        payload.name.as_deref(),
        payload.description.as_deref(),
        payload.family.as_deref(),
        payload.published,
    )
    .await
    {
        Ok(Some(model)) => HttpResponse::Ok().json(ModelResponse::from(model)),
        Ok(None) => HttpResponse::NotFound().body("model not found"),
        // name is UNIQUE; renaming onto an existing name is a client conflict.
        Err(sqlx::Error::Database(db_err)) if db_err.is_unique_violation() => {
            HttpResponse::Conflict().body("model name already exists")
        }
        Err(e) => {
            eprintln!("update model db error: {}", e);
            HttpResponse::InternalServerError().body("failed to update model")
        }
    }
}

#[delete("/{id}")]
async fn delete_model(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    // The db delete detaches referencing messages and removes the row in one
    // transaction, returning the model's name to confirm it existed. Weights are
    // stored under the model's id, so we clean up that directory.
    let id = path.into_inner();
    match db::model::delete(pool.get_ref(), id).await {
        Ok(Some(_name)) => {
            let dir = storage::models_dir().join(id.to_string());
            if let Err(e) = fs::remove_dir_all(&dir) {
                // The row is already gone; a missing/half-removed directory isn't
                // worth failing the request over — log and report success.
                if e.kind() != std::io::ErrorKind::NotFound {
                    eprintln!("failed to remove model dir {}: {}", dir.display(), e);
                }
            }
            HttpResponse::NoContent().finish()
        }
        Ok(None) => HttpResponse::NotFound().body("model not found"),
        Err(e) => {
            eprintln!("delete model db error: {}", e);
            HttpResponse::InternalServerError().body("failed to delete model")
        }
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list_models);
    cfg.service(get_model);
    cfg.service(update_model);
    cfg.service(delete_model);
}
