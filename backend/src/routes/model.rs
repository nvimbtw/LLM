use actix_web::{HttpResponse, Responder, delete, get, put, web};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::PgPool;
use std::fs;
use uuid::Uuid;

use crate::db;
use crate::extractors::{AdminUser, AuthUser};
use crate::storage;

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
            created_at: m.created_at,
        }
    }
}

// Any authenticated user may browse the registry — they need it to pick a model
// in chat. Mutating routes below require admin.
#[get("")]
async fn list_models(_auth: AuthUser, pool: web::Data<PgPool>) -> impl Responder {
    match db::model::list(pool.get_ref()).await {
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
async fn get_model(
    _auth: AuthUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    match db::model::find_by_id(pool.get_ref(), path.into_inner()).await {
        Ok(Some(model)) => HttpResponse::Ok().json(ModelResponse::from(model)),
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
}

#[put("/{id}")]
async fn update_model(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    payload: web::Json<UpdateModelPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    match db::model::update(
        pool.get_ref(),
        path.into_inner(),
        payload.name.as_deref(),
        payload.description.as_deref(),
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
    // transaction, returning the model's name so we can clean up its weights on
    // disk too.
    match db::model::delete(pool.get_ref(), path.into_inner()).await {
        Ok(Some(name)) => {
            let dir = storage::models_dir().join(&name);
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
