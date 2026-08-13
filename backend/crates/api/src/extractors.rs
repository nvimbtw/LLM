//! Request guards. Naming one as a handler argument runs it before the body,
//! so an unauthenticated request never reaches the handler at all.

use std::future::{Ready, ready};

use actix_web::{
    FromRequest, HttpRequest,
    dev::Payload,
    error::{ErrorForbidden, ErrorInternalServerError, ErrorUnauthorized},
    http::header::AUTHORIZATION,
    web,
};

use auth::AccessLevel;
use auth::token::{self as jwt, Claims};
use auth::token_cipher;

fn authenticate(req: &HttpRequest) -> Result<Claims, actix_web::Error> {
    // A missing secret is this server misconfigured, not the client at fault.
    let Some(secret) = req.app_data::<web::Data<String>>() else {
        eprintln!("jwt secret missing from app_data");
        return Err(ErrorInternalServerError("auth misconfigured"));
    };
    let secret = secret.get_ref();

    let Some(header) = req.headers().get(AUTHORIZATION) else {
        return Err(ErrorUnauthorized("missing authorization header"));
    };
    let Some(encrypted) = header.to_str().ok().and_then(|h| h.strip_prefix("Bearer ")) else {
        return Err(ErrorUnauthorized("invalid authorization header"));
    };

    // The client holds the JWT encrypted, so its claims are not readable there.
    // Decrypting is what turns the blob back into something with a signature.
    let Some(token) = token_cipher::decrypt(encrypted, secret) else {
        return Err(ErrorUnauthorized("invalid token"));
    };

    jwt::verify_user_token(&token, secret).map_err(|e| {
        eprintln!("jwt verify error: {e}");
        ErrorUnauthorized("invalid token")
    })
}

pub struct AuthUser {
    pub claims: Claims,
}

impl FromRequest for AuthUser {
    type Error = actix_web::Error;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _payload: &mut Payload) -> Self::Future {
        ready(authenticate(req).map(|claims| AuthUser { claims }))
    }
}

/// Authenticated *and* an admin. A valid non-admin token is 403, not 401.
pub struct AdminUser {
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
