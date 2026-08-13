//! Named training configs. `config` is stored opaquely — a run validates its
//! own shape when it starts, so there is nothing to check here but the name.

use actix_web::{HttpResponse, Responder, delete, get, post, put, web};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

use crate::extractors::AuthUser;

/// `id_user` is deliberately absent: a client only ever sees its own presets.
#[derive(Serialize)]
struct PresetResponse {
    id_preset: Uuid,
    name: String,
    config: Value,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl From<db::preset::Preset> for PresetResponse {
    fn from(p: db::preset::Preset) -> Self {
        PresetResponse {
            id_preset: p.id_preset,
            name: p.name,
            config: p.config,
            created_at: p.created_at,
            updated_at: p.updated_at,
        }
    }
}

#[derive(Deserialize)]
struct CreatePresetPayload {
    name: String,
    config: Value,
}

#[derive(Deserialize)]
struct UpdatePresetPayload {
    name: Option<String>,
    config: Option<Value>,
}

fn blank_name() -> HttpResponse {
    HttpResponse::BadRequest().body("preset name is required")
}

/// A preset owned by someone else reads as absent, never as forbidden.
fn not_found() -> HttpResponse {
    HttpResponse::NotFound().body("preset not found")
}

/// `(id_user, name)` is UNIQUE, so a clash is the client's to resolve.
fn duplicate_name() -> HttpResponse {
    HttpResponse::Conflict().body("preset name already exists")
}

fn failed(action: &str, e: sqlx::Error) -> HttpResponse {
    eprintln!("{action} preset db error: {e}");
    HttpResponse::InternalServerError().body(format!("failed to {action} preset"))
}

#[get("")]
async fn list_presets(auth: AuthUser, pool: web::Data<PgPool>) -> impl Responder {
    match db::preset::list_for_user(pool.get_ref(), auth.claims.sub).await {
        Ok(presets) => HttpResponse::Ok()
            .json(presets.into_iter().map(PresetResponse::from).collect::<Vec<_>>()),
        Err(e) => failed("list", e),
    }
}

#[post("")]
async fn create_preset(
    auth: AuthUser,
    payload: web::Json<CreatePresetPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let name = payload.name.trim();
    if name.is_empty() {
        return blank_name();
    }

    match db::preset::create(pool.get_ref(), auth.claims.sub, name, &payload.config).await {
        Ok(preset) => HttpResponse::Created().json(PresetResponse::from(preset)),
        Err(sqlx::Error::Database(e)) if e.is_unique_violation() => duplicate_name(),
        Err(e) => failed("create", e),
    }
}

#[put("/{id}")]
async fn update_preset(
    auth: AuthUser,
    path: web::Path<Uuid>,
    payload: web::Json<UpdatePresetPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let name = payload.name.as_deref().map(str::trim);
    if name == Some("") {
        return blank_name();
    }

    let updated = db::preset::update(
        pool.get_ref(),
        path.into_inner(),
        auth.claims.sub,
        name,
        payload.config.as_ref(),
    )
    .await;

    match updated {
        Ok(Some(preset)) => HttpResponse::Ok().json(PresetResponse::from(preset)),
        Ok(None) => not_found(),
        Err(sqlx::Error::Database(e)) if e.is_unique_violation() => duplicate_name(),
        Err(e) => failed("update", e),
    }
}

#[delete("/{id}")]
async fn delete_preset(
    auth: AuthUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    match db::preset::delete(pool.get_ref(), path.into_inner(), auth.claims.sub).await {
        Ok(true) => HttpResponse::NoContent().finish(),
        Ok(false) => not_found(),
        Err(e) => failed("delete", e),
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list_presets);
    cfg.service(create_preset);
    cfg.service(update_preset);
    cfg.service(delete_preset);
}
