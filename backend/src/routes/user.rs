use actix_web::{HttpResponse, Responder, delete, get, post, put, web};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::routes::validate;
use crate::{db, extractors::AuthUser, jwt};

// Response shape sent to clients. The db `User` struct carries the password
// hash, so we never serialize it directly — this strips that field.
#[derive(Serialize)]
struct UserResponse {
    id_user: Uuid,
    username: String,
    access_level: db::user::AccessLevel,
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

// Self-or-admin policy: a caller may act on their own record, and admins may
// act on anyone. Used by the /{id} handlers; /me is implicitly always self.
fn authorized(claims: &jwt::Claims, target: Uuid) -> bool {
    claims.sub == target || matches!(claims.access_level, db::user::AccessLevel::Admin)
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
    let username = &payload.username;
    let password = &payload.password;

    if let Err(msg) = validate::field_len(
        "username",
        username,
        validate::MIN_USERNAME_LEN,
        validate::MAX_USERNAME_LEN,
    ) {
        return HttpResponse::BadRequest().body(msg);
    }
    if let Err(msg) = validate::field_len(
        "password",
        password,
        validate::MIN_PASSWORD_LEN,
        validate::MAX_PASSWORD_LEN,
    ) {
        return HttpResponse::BadRequest().body(msg);
    }

    let pwd_hash = match crypto::password::hash_password(password) {
        Ok(hash) => hash,
        Err(e) => {
            // Log the real cause server-side; never leak internals to the client.
            eprintln!("password hashing failed: {}", e);
            return HttpResponse::InternalServerError().body("failed to create user");
        }
    };

    match db::user::create(
        pool.get_ref(),
        username.as_str(),
        pwd_hash.as_str(),
        Some(db::user::AccessLevel::User),
    )
    .await
    {
        Ok(_) => HttpResponse::Ok().body("inserting user"),
        Err(sqlx::Error::Database(db_err)) if db_err.is_unique_violation() => {
            HttpResponse::Conflict().body("username already exists")
        }
        Err(e) => {
            eprintln!("failed to create user: {}", e);
            HttpResponse::InternalServerError().body("failed to create user")
        }
    }
}

// Resolves the authenticated caller from their token alone — no id in the URL,
// so the frontend never needs to know (or decrypt) its own user id.
#[get("/me")]
async fn get_me(auth: AuthUser, pool: web::Data<PgPool>) -> impl Responder {
    match db::user::find_by_id(pool.get_ref(), auth.claims.sub).await {
        Ok(Some(user)) => HttpResponse::Ok().json(UserResponse::from(user)),
        Ok(None) => HttpResponse::NotFound().body("user not found"),
        Err(e) => {
            eprintln!("get me db error: {}", e);
            HttpResponse::InternalServerError().body("failed to get user")
        }
    }
}

#[get("/{id}")]
async fn get_user(
    auth: AuthUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id = path.into_inner();

    if !authorized(&auth.claims, id) {
        return HttpResponse::Forbidden().body("forbidden");
    }

    match db::user::find_by_id(pool.get_ref(), id).await {
        Ok(Some(user)) => HttpResponse::Ok().json(UserResponse::from(user)),
        Ok(None) => HttpResponse::NotFound().body("user not found"),
        Err(e) => {
            eprintln!("get user db error: {}", e);
            HttpResponse::InternalServerError().body("failed to get user")
        }
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

    if !authorized(&auth.claims, id) {
        return HttpResponse::Forbidden().body("forbidden");
    }

    if let Some(msg) = payload.username.as_deref().and_then(|u| {
        validate::field_len(
            "username",
            u,
            validate::MIN_USERNAME_LEN,
            validate::MAX_USERNAME_LEN,
        )
        .err()
    }) {
        return HttpResponse::BadRequest().body(msg);
    }
    if let Some(msg) = payload.password.as_deref().and_then(|p| {
        validate::field_len(
            "password",
            p,
            validate::MIN_PASSWORD_LEN,
            validate::MAX_PASSWORD_LEN,
        )
        .err()
    }) {
        return HttpResponse::BadRequest().body(msg);
    }

    // Hash the new password (if one was supplied) before it touches the db.
    let pwd_hash = match &payload.password {
        Some(password) => match crypto::password::hash_password(password) {
            Ok(hash) => Some(hash),
            Err(e) => {
                eprintln!("password hash error: {}", e);
                return HttpResponse::InternalServerError().body("failed to update user");
            }
        },
        _ => None,
    };

    match db::user::update(
        pool.get_ref(),
        id,
        payload.username.as_deref(),
        pwd_hash.as_deref(),
    )
    .await
    {
        Ok(Some(user)) => HttpResponse::Ok().json(UserResponse::from(user)),
        Ok(_) => HttpResponse::NotFound().body("user not found"),
        Err(sqlx::Error::Database(db_err)) if db_err.is_unique_violation() => {
            HttpResponse::Conflict().body("username already exists")
        }
        Err(e) => {
            eprintln!("update user db error: {}", e);
            HttpResponse::InternalServerError().body("failed to update user")
        }
    }
}

#[delete("/{id}")]
async fn delete_user(
    auth: AuthUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id = path.into_inner();

    if !authorized(&auth.claims, id) {
        return HttpResponse::Forbidden().body("forbidden");
    }

    match db::user::delete(pool.get_ref(), id).await {
        Ok(true) => HttpResponse::NoContent().finish(),
        Ok(false) => HttpResponse::NotFound().body("user not found"),
        // chats/messages reference users(id_user); deleting a user that still
        // has rows pointing at it trips the FK constraint — that's a client
        // conflict, not a server fault.
        Err(sqlx::Error::Database(db_err)) if db_err.is_foreign_key_violation() => {
            HttpResponse::Conflict().body("user has associated data")
        }
        Err(e) => {
            eprintln!("delete user db error: {}", e);
            HttpResponse::InternalServerError().body("failed to delete user")
        }
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(insert_user);
    // get_me must be registered before get_user: otherwise "/me" would be
    // captured by the "/{id}" pattern and fail to parse as a UUID.
    cfg.service(get_me);
    cfg.service(get_user);
    cfg.service(update_user);
    cfg.service(delete_user);
}
