use std::future::{Ready, ready};

use actix_web::{
    FromRequest, HttpRequest,
    dev::Payload,
    error::{ErrorInternalServerError, ErrorUnauthorized},
    http::header::AUTHORIZATION,
    web,
};

use crate::crypto::{
    jwt::{self, Claims},
    token_cipher,
};

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
        // The secret is registered via `app_data` in main.rs. Its absence is a
        // server misconfiguration, not a client error — hence 500, not 401.
        let secret = match req.app_data::<web::Data<String>>() {
            Some(secret) => secret.get_ref().clone(),
            None => {
                eprintln!("jwt secret missing from app_data");
                return ready(Err(ErrorInternalServerError("auth misconfigured")));
            }
        };

        let header = match req.headers().get(AUTHORIZATION) {
            Some(header) => header,
            None => return ready(Err(ErrorUnauthorized("missing authorization header"))),
        };

        // `to_str` fails on non-ASCII bytes; `strip_prefix` enforces the
        // `Bearer ` scheme. Either failure is a malformed header → 401.
        let encrypted = match header.to_str().ok().and_then(|h| h.strip_prefix("Bearer ")) {
            Some(token) => token,
            None => return ready(Err(ErrorUnauthorized("invalid authorization header"))),
        };

        // The client holds the token encrypted; decrypt it back into the signed
        // JWT before the signature can be checked. A failure here means the blob
        // was garbage or tampered with → 401.
        let token = match token_cipher::decrypt(encrypted, &secret) {
            Some(token) => token,
            None => return ready(Err(ErrorUnauthorized("invalid token"))),
        };

        match jwt::verify_token(&token, &secret) {
            Ok(claims) => ready(Ok(AuthUser { claims })),
            Err(e) => {
                // Covers bad signature, expiry, malformed token, etc. Log the
                // specifics server-side; tell the client nothing beyond 401.
                eprintln!("jwt verify error: {}", e);
                ready(Err(ErrorUnauthorized("invalid token")))
            }
        }
    }
}
