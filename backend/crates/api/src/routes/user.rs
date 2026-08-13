use actix_web::{HttpResponse, Responder, delete, get, post, put, web};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::extractors::AuthUser;
use crate::routes::validate;
use auth::{AccessLevel, token::Claims};

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

    match db::user::create(pool.get_ref(), &payload.username, &hash, Some(AccessLevel::User)).await
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
async fn get_user(auth: AuthUser, path: web::Path<Uuid>, pool: web::Data<PgPool>) -> impl Responder {
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

    let hash = match payload.password.as_deref().map(auth::password::hash_password) {
        Some(Ok(hash)) => Some(hash),
        Some(Err(e)) => return failed("update", e),
        None => None,
    };

    match db::user::update(pool.get_ref(), id, payload.username.as_deref(), hash.as_deref()).await {
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
    engine: web::Data<engine::Proxy>,
) -> impl Responder {
    let id = path.into_inner();
    if !is_self_or_admin(&auth.claims, id) {
        return forbidden();
    }

    // The engine goes first, and its erase is idempotent: this order can be
    // retried, the other leaves chats owned by an id nothing can resolve.
    if let Err(e) = engine.delete(&::auth::Caller::System, &format!("/owners/{id}")).await {
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
    cfg.service(get_user);
    cfg.service(update_user);
    cfg.service(delete_user);
}
