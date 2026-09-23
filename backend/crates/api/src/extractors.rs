//! Request guards. Naming one as a handler argument runs it before the body,
//! so an unauthenticated request never reaches the handler at all.

use std::future::{ready, Ready};

use actix_web::{
    dev::Payload,
    error::{ErrorForbidden, ErrorInternalServerError, ErrorUnauthorized},
    http::header::AUTHORIZATION,
    web, FromRequest, HttpRequest,
};

use auth::token::{self as jwt, Claims};
use auth::token_cipher;
use auth::AccessLevel;
use service::InboundSecret;

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

/// A **service** asking, rather than a person: the LLM engine fetching the documents
/// a queued run has reached the point of needing.
///
/// Only the LLM engine may ask — naming the acceptable issuer here is what keeps the
/// data engine, which has no business reading through the backend, from doing
/// it. The caller it carries is the person the run was queued for, and every
/// route below re-mints toward the other service with exactly that, so a run
/// reads what whoever queued it could have read and nothing more.
pub struct ServiceCaller {
    pub caller: auth::Caller,
}

impl FromRequest for ServiceCaller {
    type Error = actix_web::Error;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _payload: &mut Payload) -> Self::Future {
        // A missing secret is not a missing *configuration*: unset is the
        // development pair, announced at startup. What would be a fault is the
        // policy object never being registered at all.
        let Some(secret) = req.app_data::<web::Data<InboundSecret>>() else {
            eprintln!("inbound service secret missing from app_data");
            return ready(Err(ErrorInternalServerError("auth misconfigured")));
        };
        let header = req
            .headers()
            .get(AUTHORIZATION)
            .and_then(|h| h.to_str().ok());

        ready(
            secret
                .caller(header, &[llm_engine::AUDIENCE])
                .map(|caller| ServiceCaller { caller })
                .map_err(|refusal| {
                    eprintln!("inbound service call refused: {refusal}");
                    ErrorUnauthorized(refusal.to_string())
                }),
        )
    }
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
