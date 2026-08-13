//! The two tokens.
//!
//! A **user token** is what a browser holds: long-lived, and it names a person.
//! A **service token** is what the backend mints to call a service it owns:
//! seconds-long, and it names the person on whose behalf the call is made.
//!
//! The user's token is never forwarded. A service that accepted it would be
//! able to act as that user anywhere the token is valid, and the audience claim
//! is what makes that structurally impossible rather than merely discouraged.

use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::access::AccessLevel;

pub type Result<T> = std::result::Result<T, jsonwebtoken::errors::Error>;

const WEB_AUDIENCE: &str = "web";
const SERVICE_ISSUER: &str = "backend";

/// How long a browser session lasts before it has to log in again.
const USER_TTL_HOURS: i64 = 24;

/// How long a service token is good for. Minted per request, one hop over
/// loopback — short enough that a captured one is spent before it is useful.
const SERVICE_TTL_SECS: i64 = 60;

/// Clock skew tolerated on that expiry. `Validation` defaults this to 60s,
/// which silently doubled the window. Both processes share a clock, so this
/// only has to absorb scheduling jitter.
const SERVICE_LEEWAY_SECS: u64 = 5;

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: Uuid,
    pub access_level: AccessLevel,
    pub aud: String,
    pub exp: usize,
}

/// A service token's payload. `act` is who the backend is acting for.
#[derive(Debug, Serialize, Deserialize)]
pub struct ServiceClaims {
    pub iss: String,
    pub aud: String,
    pub act: String,
    pub scope: Vec<String>,
    pub jti: Uuid,
    pub exp: usize,
}

/// Who a service token says is calling.
#[derive(Debug, PartialEq)]
pub enum Caller {
    /// A request made on a person's behalf.
    User(Uuid),
    /// The backend acting for itself — a startup health check, a sweep.
    System,
}

impl Caller {
    fn as_claim(&self) -> String {
        match self {
            Caller::User(id) => id.to_string(),
            Caller::System => "system".into(),
        }
    }

    fn parse(raw: &str) -> Caller {
        Uuid::parse_str(raw).map(Caller::User).unwrap_or(Caller::System)
    }
}

pub fn issue_user_token(user_id: Uuid, access_level: AccessLevel, secret: &str) -> Result<String> {
    let claims = Claims {
        sub: user_id,
        access_level,
        aud: WEB_AUDIENCE.into(),
        exp: expires_in(chrono::Duration::hours(USER_TTL_HOURS)),
    };
    encode(&Header::default(), &claims, &EncodingKey::from_secret(secret.as_bytes()))
}

pub fn verify_user_token(token: &str, secret: &str) -> Result<Claims> {
    let mut validation = Validation::default();
    validation.set_audience(&[WEB_AUDIENCE]);
    let data = decode::<Claims>(token, &DecodingKey::from_secret(secret.as_bytes()), &validation)?;
    Ok(data.claims)
}

/// Mint a token for one outbound call to `audience`.
pub fn issue_service_token(
    audience: &str,
    caller: &Caller,
    scope: &[&str],
    secret: &str,
) -> Result<String> {
    let claims = ServiceClaims {
        iss: SERVICE_ISSUER.into(),
        aud: audience.into(),
        act: caller.as_claim(),
        scope: scope.iter().map(|s| s.to_string()).collect(),
        jti: Uuid::new_v4(),
        exp: expires_in(chrono::Duration::seconds(SERVICE_TTL_SECS)),
    };
    encode(&Header::default(), &claims, &EncodingKey::from_secret(secret.as_bytes()))
}

/// Verify a token addressed to `audience`, returning who it is acting for.
pub fn verify_service_token(token: &str, audience: &str, secret: &str) -> Result<Caller> {
    let mut validation = Validation::default();
    validation.leeway = SERVICE_LEEWAY_SECS;
    validation.set_audience(&[audience]);
    validation.set_issuer(&[SERVICE_ISSUER]);
    let data =
        decode::<ServiceClaims>(token, &DecodingKey::from_secret(secret.as_bytes()), &validation)?;
    Ok(Caller::parse(&data.claims.act))
}

fn expires_in(window: chrono::Duration) -> usize {
    (chrono::Utc::now() + window).timestamp() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    const WEB: &str = "web-secret";
    const SERVICE: &str = "service-secret";

    #[test]
    fn a_user_token_round_trips_its_claims() {
        let id = Uuid::new_v4();
        let token = issue_user_token(id, AccessLevel::Admin, WEB).unwrap();
        let claims = verify_user_token(&token, WEB).unwrap();

        assert_eq!(claims.sub, id);
        assert_eq!(claims.access_level, AccessLevel::Admin);
    }

    #[test]
    fn a_user_token_is_rejected_under_the_wrong_secret() {
        let token = issue_user_token(Uuid::new_v4(), AccessLevel::User, WEB).unwrap();
        assert!(verify_user_token(&token, "not-the-secret").is_err());
    }

    #[test]
    fn a_service_token_names_the_user_it_acts_for() {
        let id = Uuid::new_v4();
        let token = issue_service_token("engine", &Caller::User(id), &["chat:write"], SERVICE).unwrap();
        assert_eq!(verify_service_token(&token, "engine", SERVICE).unwrap(), Caller::User(id));
    }

    #[test]
    fn a_service_token_can_act_for_the_backend_itself() {
        let token = issue_service_token("engine", &Caller::System, &[], SERVICE).unwrap();
        assert_eq!(verify_service_token(&token, "engine", SERVICE).unwrap(), Caller::System);
    }

    /// The property the whole two-secret split exists for: a token good for the
    /// browser must not open the engine, whichever secret is tried.
    #[test]
    fn a_user_token_is_not_a_service_token() {
        let token = issue_user_token(Uuid::new_v4(), AccessLevel::Admin, WEB).unwrap();
        assert!(verify_service_token(&token, "engine", WEB).is_err());
        assert!(verify_service_token(&token, "engine", SERVICE).is_err());
    }

    #[test]
    fn a_service_token_is_not_a_user_token() {
        let token = issue_service_token("engine", &Caller::System, &[], SERVICE).unwrap();
        assert!(verify_user_token(&token, SERVICE).is_err());
        assert!(verify_user_token(&token, WEB).is_err());
    }

    #[test]
    fn a_token_for_another_service_is_refused() {
        let token = issue_service_token("scraper", &Caller::System, &[], SERVICE).unwrap();
        assert!(verify_service_token(&token, "engine", SERVICE).is_err());
    }

    #[test]
    fn an_expired_service_token_is_refused() {
        let claims = ServiceClaims {
            iss: SERVICE_ISSUER.into(),
            aud: "engine".into(),
            act: "system".into(),
            scope: vec![],
            jti: Uuid::new_v4(),
            // Past the leeway, or this asserts nothing.
            exp: expires_in(chrono::Duration::seconds(-(SERVICE_LEEWAY_SECS as i64) - 1)),
        };
        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(SERVICE.as_bytes()),
        )
        .unwrap();
        assert!(verify_service_token(&token, "engine", SERVICE).is_err());
    }

    #[test]
    fn a_token_from_an_unknown_issuer_is_refused() {
        let claims = ServiceClaims {
            iss: "somebody-else".into(),
            aud: "engine".into(),
            act: "system".into(),
            scope: vec![],
            jti: Uuid::new_v4(),
            exp: expires_in(chrono::Duration::seconds(60)),
        };
        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(SERVICE.as_bytes()),
        )
        .unwrap();
        assert!(verify_service_token(&token, "engine", SERVICE).is_err());
    }
}


/// The token seam, verified against the same vector the engine checks.
///
/// Two implementations of one format drift silently: the symptom is a 401 in
/// production, long after whichever side changed. `contracts/service-token.json`
/// is the shared fixture, and both suites read it.
#[cfg(test)]
mod contract {
    use super::*;

    #[derive(Deserialize)]
    struct Vector {
        secret: String,
        audience: String,
        expected_act: String,
        token: String,
    }

    fn vector() -> Vector {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../contracts/service-token.json");
        let raw = std::fs::read_to_string(path).expect("contracts/service-token.json");
        serde_json::from_str(&raw).expect("valid contract fixture")
    }

    #[test]
    fn the_shared_vector_verifies() {
        let v = vector();
        let caller = verify_service_token(&v.token, &v.audience, &v.secret).expect("verifies");
        assert_eq!(caller, Caller::User(Uuid::parse_str(&v.expected_act).unwrap()));
    }
}
