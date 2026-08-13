//! Answering with what a service said.
//!
//! Re-modelling a payload this side owns nothing of would mean editing two
//! places for every field, so a proxy route returns the JSON as it arrived —
//! and the status too: a 404 flattened into a 502 tells a client nothing.

use actix_web::{HttpRequest, HttpResponse, ResponseError, http::StatusCode};
use engine::EngineError;
use serde_json::{Value, json};

use crate::extractors::AuthUser;

pub type Answer = Result<HttpResponse, Upstream>;

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
            // Not reaching the service is this server's problem, not the
            // client's to interpret.
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

pub fn answer(value: Value) -> Answer {
    Ok(HttpResponse::Ok().json(value))
}

pub fn caller(user: &AuthUser) -> auth::Caller {
    auth::Caller::User(user.claims.sub)
}

/// `path` with the caller's query string appended untouched. Paging and flags
/// belong to the service that defines them, and restating them here would be
/// two places to change one default.
pub fn with_query(path: &str, req: &HttpRequest) -> String {
    match req.query_string() {
        "" => path.to_string(),
        query => format!("{path}?{query}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::test::TestRequest;

    fn status_of(e: EngineError) -> u16 {
        Upstream::from(e).status_code().as_u16()
    }

    #[test]
    fn a_services_refusal_reaches_the_client_as_itself() {
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

    #[test]
    fn a_query_string_rides_along_or_is_absent() {
        let bare = TestRequest::get().uri("/runs").to_http_request();
        assert_eq!(with_query("/runs", &bare), "/runs");

        let paged = TestRequest::get().uri("/runs?limit=2&offset=4").to_http_request();
        assert_eq!(with_query("/runs", &paged), "/runs?limit=2&offset=4");
    }
}
