use actix_web::{delete, get, post, put, web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

use crate::extractors::AuthUser;
use crate::routes::validate;
use auth::{token::Claims, AccessLevel};

/// The db `User` carries the password hash, so it is never serialized directly.
#[derive(Serialize)]
struct UserResponse {
    id_user: Uuid,
    username: String,
    access_level: AccessLevel,
}

impl From<db::user::User> for UserResponse {
    fn from(user: db::user::User) -> Self {
        UserResponse {
            id_user: user.id_user,
            username: user.username,
            access_level: user.access_level,
        }
    }
}

fn is_self_or_admin(claims: &Claims, target: Uuid) -> bool {
    claims.sub == target || matches!(claims.access_level, AccessLevel::Admin)
}

fn forbidden() -> HttpResponse {
    HttpResponse::Forbidden().body("forbidden")
}

fn not_found() -> HttpResponse {
    HttpResponse::NotFound().body("user not found")
}

fn taken() -> HttpResponse {
    HttpResponse::Conflict().body("username already exists")
}

fn failed(action: &str, e: impl std::fmt::Display) -> HttpResponse {
    eprintln!("{action} user error: {e}");
    HttpResponse::InternalServerError().body(format!("failed to {action} user"))
}

#[derive(Deserialize)]
struct CreateUserPayload {
    username: String,
    password: String,
}

#[post("")]
async fn insert_user(
    payload: web::Json<CreateUserPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    if let Err(msg) = validate::username(&payload.username) {
        return HttpResponse::BadRequest().body(msg);
    }
    if let Err(msg) = validate::password(&payload.password) {
        return HttpResponse::BadRequest().body(msg);
    }

    let hash = match auth::password::hash_password(&payload.password) {
        Ok(hash) => hash,
        Err(e) => return failed("create", e),
    };

    match db::user::create(
        pool.get_ref(),
        &payload.username,
        &hash,
        Some(AccessLevel::User),
    )
    .await
    {
        Ok(_) => HttpResponse::Created().finish(),
        Err(sqlx::Error::Database(e)) if e.is_unique_violation() => taken(),
        Err(e) => failed("create", e),
    }
}

/// Resolved from the token alone, so a client never needs to know its own id.
#[get("/me")]
async fn get_me(auth: AuthUser, pool: web::Data<PgPool>) -> impl Responder {
    fetch(pool.get_ref(), auth.claims.sub).await
}

#[get("/{id}")]
async fn get_user(
    auth: AuthUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id = path.into_inner();
    if !is_self_or_admin(&auth.claims, id) {
        return forbidden();
    }
    fetch(pool.get_ref(), id).await
}

async fn fetch(pool: &PgPool, id: Uuid) -> HttpResponse {
    match db::user::find_by_id(pool, id).await {
        Ok(Some(user)) => HttpResponse::Ok().json(UserResponse::from(user)),
        Ok(None) => not_found(),
        Err(e) => failed("get", e),
    }
}

/// Not folded into `UserResponse`: `/{id}` is also how an admin looks up
/// someone else, and that response shouldn't carry a stranger's system prompt
/// and decoding knobs along with it.
#[derive(Serialize, Deserialize)]
struct SettingsPayload {
    settings: Value,
}

#[get("/me/settings")]
async fn get_my_settings(auth: AuthUser, pool: web::Data<PgPool>) -> impl Responder {
    match db::user::get_settings(pool.get_ref(), auth.claims.sub).await {
        Ok(Some(settings)) => HttpResponse::Ok().json(SettingsPayload { settings }),
        Ok(None) => not_found(),
        Err(e) => failed("get settings for", e),
    }
}

/// Replaces the stored blob outright — the client always holds and sends the
/// full object, so there is no partial-update case to reconcile here.
#[put("/me/settings")]
async fn update_my_settings(
    auth: AuthUser,
    payload: web::Json<SettingsPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    match db::user::update_settings(pool.get_ref(), auth.claims.sub, &payload.settings).await {
        Ok(Some(settings)) => HttpResponse::Ok().json(SettingsPayload { settings }),
        Ok(None) => not_found(),
        Err(e) => failed("update settings for", e),
    }
}

/// What the composer is set to send with. Its own resource rather than a field
/// on `SettingsPayload`: a model or budget pick writes immediately, a settings
/// change debounces, and a shared blob would mean the later of two independent
/// writes clobbering the other's field.
#[derive(Serialize, Deserialize)]
struct ComposerChoicePayload {
    composer_choice: Value,
}

#[get("/me/composer-choice")]
async fn get_my_composer_choice(auth: AuthUser, pool: web::Data<PgPool>) -> impl Responder {
    match db::user::get_composer_choice(pool.get_ref(), auth.claims.sub).await {
        Ok(Some(composer_choice)) => {
            HttpResponse::Ok().json(ComposerChoicePayload { composer_choice })
        }
        Ok(None) => not_found(),
        Err(e) => failed("get composer choice for", e),
    }
}

#[put("/me/composer-choice")]
async fn update_my_composer_choice(
    auth: AuthUser,
    payload: web::Json<ComposerChoicePayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    match db::user::update_composer_choice(
        pool.get_ref(),
        auth.claims.sub,
        &payload.composer_choice,
    )
    .await
    {
        Ok(Some(composer_choice)) => {
            HttpResponse::Ok().json(ComposerChoicePayload { composer_choice })
        }
        Ok(None) => not_found(),
        Err(e) => failed("update composer choice for", e),
    }
}

#[derive(Deserialize)]
struct UpdateUserPayload {
    username: Option<String>,
    password: Option<String>,
}

#[put("/{id}")]
async fn update_user(
    auth: AuthUser,
    path: web::Path<Uuid>,
    payload: web::Json<UpdateUserPayload>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id = path.into_inner();
    if !is_self_or_admin(&auth.claims, id) {
        return forbidden();
    }

    if let Some(Err(msg)) = payload.username.as_deref().map(validate::username) {
        return HttpResponse::BadRequest().body(msg);
    }
    if let Some(Err(msg)) = payload.password.as_deref().map(validate::password) {
        return HttpResponse::BadRequest().body(msg);
    }

    let hash = match payload
        .password
        .as_deref()
        .map(auth::password::hash_password)
    {
        Some(Ok(hash)) => Some(hash),
        Some(Err(e)) => return failed("update", e),
        None => None,
    };

    match db::user::update(
        pool.get_ref(),
        id,
        payload.username.as_deref(),
        hash.as_deref(),
    )
    .await
    {
        Ok(Some(user)) => HttpResponse::Ok().json(UserResponse::from(user)),
        Ok(None) => not_found(),
        Err(sqlx::Error::Database(e)) if e.is_unique_violation() => taken(),
        Err(e) => failed("update", e),
    }
}

#[delete("/{id}")]
async fn delete_user(
    auth: AuthUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
    engine: web::Data<llm_engine::Proxy>,
) -> impl Responder {
    let id = path.into_inner();
    if !is_self_or_admin(&auth.claims, id) {
        return forbidden();
    }

    // The LLM engine goes first, and its erase is idempotent: this order can be
    // retried, the other leaves chats owned by an id nothing can resolve.
    if let Err(e) = engine
        .delete(&::auth::Caller::System, &format!("/owners/{id}"))
        .await
    {
        eprintln!("engine erase error: {e}");
        return HttpResponse::BadGateway().body("failed to erase this user's conversations");
    }

    match db::user::delete(pool.get_ref(), id).await {
        Ok(true) => HttpResponse::NoContent().finish(),
        Ok(false) => not_found(),
        Err(e) => failed("delete", e),
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(insert_user);
    // Before get_user, or "/me" is captured by "/{id}" and fails to parse.
    cfg.service(get_me);
    cfg.service(get_my_settings);
    cfg.service(update_my_settings);
    cfg.service(get_my_composer_choice);
    cfg.service(update_my_composer_choice);
    cfg.service(get_user);
    cfg.service(update_user);
    cfg.service(delete_user);
}
