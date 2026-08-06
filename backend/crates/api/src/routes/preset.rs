use actix_web::{HttpResponse, Responder, delete, get, post, put, web};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

use db;
use crate::extractors::AuthUser;

// Response shape sent to clients. Kept separate from the db `Preset` struct per
// the layering rules, even though they currently carry the same fields. `id_user`
// is deliberately not echoed — the client only ever sees its own presets.
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

// The preset store treats `config` as an opaque JSON blob — the `/training` POST
// validates the config's shape when a run actually starts, so there's nothing to
// check here beyond a non-empty, trimmed name.
fn validate_name(name: &str) -> Result<(), HttpResponse> {
    if name.trim().is_empty() {
        return Err(HttpResponse::BadRequest().body("preset name is required"));
    }
    Ok(())
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

#[get("")]
async fn list_presets(auth: AuthUser, pool: web::Data<PgPool>) -> impl Responder {
    match db::preset::list_for_user(pool.get_ref(), auth.claims.sub).await {
        Ok(presets) => {
            let body: Vec<PresetResponse> =
                presets.into_iter().map(PresetResponse::from).collect();
            HttpResponse::Ok().json(body)
        }
        Err(e) => {
            eprintln!("list presets db error: {}", e);
            HttpResponse::InternalServerError().body("failed to list presets")
        }
    }
}

#[post("")]
async fn create_preset(
    auth: AuthUser,
    payload: web::Json<CreatePresetPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let name = payload.name.trim();
    if let Err(resp) = validate_name(name) {
        return resp;
    }
    match db::preset::create(pool.get_ref(), auth.claims.sub, name, &payload.config).await {
        Ok(preset) => HttpResponse::Created().json(PresetResponse::from(preset)),
        // UNIQUE(id_user, name): the user already has a preset by this name.
        Err(sqlx::Error::Database(db_err)) if db_err.is_unique_violation() => {
            HttpResponse::Conflict().body("preset name already exists")
        }
        Err(e) => {
            eprintln!("create preset db error: {}", e);
            HttpResponse::InternalServerError().body("failed to create preset")
        }
    }
}

#[put("/{id}")]
async fn update_preset(
    auth: AuthUser,
    path: web::Path<Uuid>,
    payload: web::Json<UpdatePresetPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    // A supplied name must be non-empty; an omitted name leaves it unchanged.
    let name = match payload.name.as_deref().map(str::trim) {
        Some(n) => {
            if let Err(resp) = validate_name(n) {
                return resp;
            }
            Some(n)
        }
        None => None,
    };

    match db::preset::update(
        pool.get_ref(),
        path.into_inner(),
        auth.claims.sub,
        name,
        payload.config.as_ref(),
    )
    .await
    {
        Ok(Some(preset)) => HttpResponse::Ok().json(PresetResponse::from(preset)),
        // Not found *for this user* — a preset owned by someone else reads as 404,
        // never 403, so its existence isn't leaked (mirrors get_model).
        Ok(None) => HttpResponse::NotFound().body("preset not found"),
        // (id_user, name) is UNIQUE; renaming onto an existing name is a conflict.
        Err(sqlx::Error::Database(db_err)) if db_err.is_unique_violation() => {
            HttpResponse::Conflict().body("preset name already exists")
        }
        Err(e) => {
            eprintln!("update preset db error: {}", e);
            HttpResponse::InternalServerError().body("failed to update preset")
        }
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
        Ok(false) => HttpResponse::NotFound().body("preset not found"),
        Err(e) => {
            eprintln!("delete preset db error: {}", e);
            HttpResponse::InternalServerError().body("failed to delete preset")
        }
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list_presets);
    cfg.service(create_preset);
    cfg.service(update_preset);
    cfg.service(delete_preset);
}
