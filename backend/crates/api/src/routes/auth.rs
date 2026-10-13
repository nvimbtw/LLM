use actix_web::{post, web, HttpResponse, Responder};
use serde::Deserialize;
use sqlx::PgPool;

use crate::routes::validate;
use auth::token as jwt;

const REFUSED: &str = "invalid username or password";

#[derive(Deserialize)]
struct LoginPayload {
    username: String,
    password: String,
}

/// Validate credentials and issue an encrypted user token; invalid credentials share one refusal response.
/// The accompanying access level is a UI hint; authorization still verifies server-side claims.
#[post("")]
async fn login(
    payload: web::Json<LoginPayload>,
    pool: web::Data<PgPool>,
    jwt_secret: web::Data<String>,
) -> impl Responder {
    if !validate::plausible_credentials(&payload.username, &payload.password) {
        return HttpResponse::BadRequest().body("invalid request");
    }

    let secret = jwt_secret.get_ref();

    let user = match db::user::find_by_username(pool.get_ref(), &payload.username).await {
        Ok(Some(user)) => user,
        Ok(None) => return HttpResponse::Unauthorized().body(REFUSED),
        Err(e) => return failed(e),
    };

    match auth::password::verify_password(&payload.password, &user.password) {
        Ok(true) => {}
        Ok(false) => return HttpResponse::Unauthorized().body(REFUSED),
        Err(e) => return failed(e),
    }

    let token = match jwt::issue_user_token(user.id_user, user.access_level, secret) {
        Ok(token) => token,
        Err(e) => return failed(e),
    };

    match auth::token_cipher::encrypt(&token, secret) {
        Ok(encrypted) => HttpResponse::Ok().json(serde_json::json!({
            "token": encrypted,
            "access_level": user.access_level,
        })),
        Err(e) => failed(e),
    }
}

/// Log a login failure internally and return a safe, stable 500 response.
fn failed(e: impl std::fmt::Display) -> HttpResponse {
    eprintln!("login error: {e}");
    HttpResponse::InternalServerError().body("login failed")
}

/// Register the login handler beneath the authentication route scope.
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(login);
}
