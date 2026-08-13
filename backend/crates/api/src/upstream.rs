//! Answering with what the engine said.
//!
//! The backend is the middleman for everything the engine owns. Re-modelling
//! those payloads here would mean editing two places for every field, so a
//! proxy route returns the engine's JSON as it arrived — and its status too: a
//! 404 flattened into a 502 tells a client nothing about what went wrong.

use actix_web::{HttpResponse, ResponseError, http::StatusCode};
use engine::EngineError;
use serde_json::{Value, json};

use crate::extractors::AuthUser;

/// What every proxy route returns.
pub type Answer = Result<HttpResponse, Upstream>;

/// An engine failure, wrapped so it can carry itself into a response.
#[derive(Debug)]
pub struct Upstream(EngineError);

impl From<EngineError> for Upstream {
    fn from(e: EngineError) -> Upstream {
        Upstream(e)
    }
}

impl std::fmt::Display for Upstream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ResponseError for Upstream {
    fn status_code(&self) -> StatusCode {
        match &self.0 {
            // Unreachable is this server's problem to report, not the client's
            // to interpret.
            EngineError::Transport(_) | EngineError::Decode(_) => StatusCode::BAD_GATEWAY,
            EngineError::Status { code, .. } => {
                StatusCode::from_u16(*code).unwrap_or(StatusCode::BAD_GATEWAY)
            }
        }
    }

    fn error_response(&self) -> HttpResponse {
        HttpResponse::build(self.status_code()).json(json!({ "error": self.0.to_string() }))
    }
}

/// The engine's answer, verbatim.
pub fn answer(value: Value) -> Answer {
    Ok(HttpResponse::Ok().json(value))
}

/// Who this request acts for, as the service token spells it.
pub fn caller(user: &AuthUser) -> auth::Caller {
    auth::Caller::User(user.claims.sub)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status_of(e: EngineError) -> u16 {
        Upstream::from(e).status_code().as_u16()
    }

    #[test]
    fn a_services_refusal_reaches_the_client_as_itself() {
        // Flattening these into one code is what makes a client guess.
        for code in [400, 401, 403, 404, 409, 422] {
            assert_eq!(status_of(EngineError::Status { code, detail: "x".into() }), code);
        }
    }

    #[test]
    fn a_service_this_side_cannot_reach_is_this_sides_problem() {
        assert_eq!(status_of(EngineError::Transport("refused".into())), 502);
        assert_eq!(status_of(EngineError::Decode("garbage".into())), 502);
    }

    #[test]
    fn a_status_no_http_client_could_mean_falls_back_rather_than_panicking() {
        assert_eq!(status_of(EngineError::Status { code: 9999, detail: "?".into() }), 502);
    }

    #[test]
    fn the_body_says_what_failed() {
        let error = Upstream::from(EngineError::Status { code: 404, detail: "no run 7".into() });
        assert!(error.to_string().contains("no run 7"));
    }
}
