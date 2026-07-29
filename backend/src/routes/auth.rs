use actix_web::{HttpResponse, Responder, post, web};
use serde::Deserialize;
use sqlx::PgPool;

use crate::routes::validate;
use crate::{db, jwt};

#[derive(Deserialize)]
struct LoginPayload {
    username: String,
    password: String,
}

#[post("")]
async fn login(
    payload: web::Json<LoginPayload>,
    pool: web::Data<PgPool>,
    jwt_secret: web::Data<String>,
) -> impl Responder {
    // Bound the inputs before hashing/verifying so an oversized password can't
    // force expensive argon2 work. Only the upper bound is checked and the reason
    // is generic — the auth failure below must not reveal which field was wrong.
    if validate::max_len("username", &payload.username, validate::MAX_USERNAME_LEN).is_err()
        || validate::max_len("password", &payload.password, validate::MAX_PASSWORD_LEN).is_err()
    {
        return HttpResponse::BadRequest().body("invalid request");
    }

    let user = match db::user::find_by_username(pool.get_ref(), &payload.username).await {
        Ok(Some(user)) => user,
        Ok(_) => return HttpResponse::Unauthorized().body("invalid username or password"),
        Err(e) => {
            eprintln!("login db error: {}", e);
            return HttpResponse::InternalServerError().body("login failed");
        }
    };

    let is_valid = match crypto::password::verify_password(&payload.password, &user.password) {
        Ok(valid) => valid,
        Err(e) => {
            eprintln!("password verify error: {}", e);
            return HttpResponse::InternalServerError().body("login failed");
        }
    };

    if !is_valid {
        return HttpResponse::Unauthorized().body("invalid username or password");
    }

    let token = match jwt::generate_token(
        user.id_user,
        user.access_level.clone(),
        jwt_secret.get_ref(),
    ) {
        Ok(token) => token,
        Err(e) => {
            eprintln!("token generation error: {}", e);
            return HttpResponse::InternalServerError().body("login failed");
        }
    };

    // Hand the JWT to the client encrypted, so its claims aren't readable
    // client-side. The matching decrypt happens in the AuthUser extractor.
    // access_level is returned alongside so the frontend can branch admin vs.
    // normal UI without decrypting the token — it stays authoritative server-side.
    match crypto::token_cipher::encrypt(&token, jwt_secret.get_ref()) {
        Ok(encrypted) => HttpResponse::Ok().json(serde_json::json!({
            "token": encrypted,
            "access_level": user.access_level,
        })),
        Err(e) => {
            eprintln!("token encryption error: {}", e);
            HttpResponse::InternalServerError().body("login failed")
        }
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(login);
}
