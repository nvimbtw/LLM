#![allow(non_snake_case)]

use std::future::{Ready, ready};

use actix_web::{
    FromRequest, HttpRequest,
    dev::Payload,
    error::{ErrorForbidden, ErrorInternalServerError, ErrorUnauthorized},
    http::header::AUTHORIZATION,
    web,
};

use crate::jwt::{self, Claims};
use crypto::token_cipher;
use crate::db::user::AccessLevel;

// Shared verification used by every auth extractor: pull the
// `Authorization: Bearer <token>` header, decrypt and verify the JWT, and hand
// back the decoded claims. Errors map to the right status (500 for a missing
// server secret, 401 for anything wrong with the client's token).
fn authenticate(req: &HttpRequest) -> Result<Claims, actix_web::Error> {
    // The secret is registered via `app_data` in main.rs. Its absence is a
    // server misconfiguration, not a client error — hence 500, not 401.
    let secret = match req.app_data::<web::Data<String>>() {
        Some(secret) => secret.get_ref().clone(),
        None => {
            eprintln!("jwt secret missing from app_data");
            return Err(ErrorInternalServerError("auth misconfigured"));
        }
    };

    let header = match req.headers().get(AUTHORIZATION) {
        Some(header) => header,
        None => return Err(ErrorUnauthorized("missing authorization header")),
    };

    // `to_str` fails on non-ASCII bytes; `strip_prefix` enforces the
    // `Bearer ` scheme. Either failure is a malformed header → 401.
    let encrypted = match header.to_str().ok().and_then(|h| h.strip_prefix("Bearer ")) {
        Some(token) => token,
        None => return Err(ErrorUnauthorized("invalid authorization header")),
    };

    // The client holds the token encrypted; decrypt it back into the signed
    // JWT before the signature can be checked. A failure here means the blob
    // was garbage or tampered with → 401.
    let token = match token_cipher::decrypt(encrypted, &secret) {
        Some(token) => token,
        None => return Err(ErrorUnauthorized("invalid token")),
    };

    jwt::verify_token(&token, &secret).map_err(|e| {
        // Covers bad signature, expiry, malformed token, etc. Log the
        // specifics server-side; tell the client nothing beyond 401.
        eprintln!("jwt verify error: {}", e);
        ErrorUnauthorized("invalid token")
    })
}

// Request guard for any endpoint that requires a valid JWT. Adding `AuthUser`
// as a handler argument makes actix run this extractor first: it pulls the
// `Authorization: Bearer <token>` header, verifies it, and short-circuits with
// 401 before the handler body ever runs. `claims` exposes the decoded payload
// (sub, access_level) for routes that need to make authorization decisions.
pub struct AuthUser {
    pub claims: Claims,
}

impl FromRequest for AuthUser {
    type Error = actix_web::Error;
    // Verification is synchronous (no DB/IO), so we hand back an already-resolved
    // future rather than dragging in async machinery.
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _payload: &mut Payload) -> Self::Future {
        ready(authenticate(req).map(|claims| AuthUser { claims }))
    }
}

// Request guard for admin-only endpoints. Same verification as `AuthUser`, but
// also requires the token's `access_level` to be `Admin` — a valid non-admin
// token is rejected with 403 (authenticated, but not permitted). Use this in
// place of `AuthUser` and the handler is guaranteed to run only for admins.
pub struct AdminUser {
    // Carried for handlers that need the authenticated admin's claims, mirroring
    // `AuthUser`; the guard's main job is proving admin access, so some admin
    // handlers never read this.
    #[allow(dead_code)]
    pub claims: Claims,
}

impl FromRequest for AdminUser {
    type Error = actix_web::Error;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _payload: &mut Payload) -> Self::Future {
        ready(authenticate(req).and_then(|claims| {
            if matches!(claims.access_level, AccessLevel::Admin) {
                Ok(AdminUser { claims })
            } else {
                Err(ErrorForbidden("admin access required"))
            }
        }))
    }
}
