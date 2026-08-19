//! The reverse hop: a service calling the backend.
//!
//! Everywhere else in this crate the backend is the caller and a Python service
//! answers. This is the one direction that runs the other way, and it exists for
//! a single reason: a training run needs its documents at the moment it reaches
//! the front of the engine's queue, which is a time nothing on this side knows
//! in advance. So the engine asks.
//!
//! It changes who talks to whom and nothing about what may be known. The two
//! Python services still never speak — a request arriving here is re-minted
//! toward the other service, carrying the same `act` claim, so a run reads
//! exactly the datasets the person who queued it could have read.
//!
//! The token is the mirror of the one `Proxy` mints outbound: same issuer, same
//! shape, addressed to `backend` instead of to a service.

use auth::Caller;
use auth::token;

/// What a token addressed to the backend says in its `aud`.
pub const AUDIENCE: &str = token::BACKEND_AUDIENCE;

const SECRET_VAR: &str = "BACKEND_SERVICE_SECRET";

/// Why a caller was turned away. Both map to 401; they are separate so the log
/// line says whether a token was absent or wrong.
#[derive(Debug, PartialEq)]
pub enum Refused {
    Missing,
    Invalid(String),
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refused::Missing => write!(f, "missing service token"),
            Refused::Invalid(why) => write!(f, "invalid service token: {why}"),
        }
    }
}

/// The secret inbound service tokens are checked against.
///
/// A newtype rather than a bare `String` for the reason each service's `Proxy`
/// is one: `app_data` is keyed by type, and the JWT secret is already registered
/// as a `String`. Two of those and every route reads whichever was inserted
/// last — here that would mean checking service tokens against the browser's
/// secret, which fails closed but for an invisible reason.
#[derive(Clone, Debug, Default)]
pub struct InboundSecret(Option<String>);

impl InboundSecret {
    pub fn from_env() -> InboundSecret {
        InboundSecret(std::env::var(SECRET_VAR).ok().filter(|s| !s.is_empty()))
    }

    pub fn new(secret: Option<String>) -> InboundSecret {
        InboundSecret(secret)
    }

    /// Whether a caller has to prove anything. Unset is the development pair,
    /// matching what the Python services do with their own secret unset — and
    /// like them, it is announced at startup rather than left to be discovered.
    pub fn enforced(&self) -> bool {
        self.0.is_some()
    }

    /// Who the bearer of this `Authorization` header is acting for, if it was
    /// minted by one of `issuers`.
    ///
    /// The issuers are named by the caller because this crate names no service.
    /// It is the load-bearing half of the check: `aud` keeps a token the backend
    /// minted from opening the backend, and the issuer list keeps a service that
    /// has no business asking from asking.
    ///
    /// With no secret configured the claim is read but not checked, because a
    /// request that cannot name a person cannot be scoped to one and every
    /// dataset route would be unusable without a secret set. That is the same
    /// bargain `api/auth.py` strikes on the other side of the hop.
    pub fn caller(&self, header: Option<&str>, issuers: &[&str]) -> Result<Caller, Refused> {
        let Some(secret) = self.0.as_deref() else {
            let act = bearer(header).and_then(|token| token::read_service_act(token).ok());
            return Ok(act.unwrap_or(Caller::System));
        };

        let token = bearer(header).ok_or(Refused::Missing)?;
        token::verify_inbound_token(token, issuers, secret)
            .map_err(|e| Refused::Invalid(e.to_string()))
    }
}

fn bearer(header: Option<&str>) -> Option<&str> {
    let header = header?;
    let (scheme, token) = header.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then_some(token.trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    const SECRET: &str = "inbound-secret";
    const ENGINE: &str = "engine";
    const ALLOWED: &[&str] = &[ENGINE];

    fn from_engine(caller: &Caller, secret: &str) -> String {
        let token = token::issue_inbound_token(ENGINE, caller, secret).unwrap();
        format!("Bearer {token}")
    }

    fn enforcing() -> InboundSecret {
        InboundSecret::new(Some(SECRET.into()))
    }

    #[test]
    fn a_token_addressed_to_the_backend_names_who_it_acts_for() {
        let id = Uuid::new_v4();

        let caller = enforcing().caller(Some(&from_engine(&Caller::User(id), SECRET)), ALLOWED);

        assert_eq!(caller, Ok(Caller::User(id)));
    }

    /// The property the audience claim exists for, in the direction that did not
    /// have it before: a token the backend minted *for* the engine must not open
    /// the backend, or anything holding one could read as any user it likes.
    #[test]
    fn a_token_the_backend_minted_for_a_service_is_refused() {
        let outbound =
            token::issue_service_token(ENGINE, &Caller::User(Uuid::new_v4()), &[""], SECRET)
                .unwrap();

        let caller = enforcing().caller(Some(&format!("Bearer {outbound}")), ALLOWED);

        assert!(matches!(caller, Err(Refused::Invalid(_))));
    }

    /// The other half: a service that is not on the list cannot ask, even
    /// holding a perfectly valid token under the same secret. The data engine
    /// has no business reading through the backend and this is what says so.
    #[test]
    fn a_token_from_a_service_that_may_not_ask_is_refused() {
        let token =
            token::issue_inbound_token("data-engine", &Caller::System, SECRET).unwrap();

        let caller = enforcing().caller(Some(&format!("Bearer {token}")), ALLOWED);

        assert!(matches!(caller, Err(Refused::Invalid(_))));
    }

    #[test]
    fn a_token_under_another_secret_is_refused() {
        let header = from_engine(&Caller::System, "some-other-secret");

        let caller = enforcing().caller(Some(&header), ALLOWED);

        assert!(matches!(caller, Err(Refused::Invalid(_))));
    }

    #[test]
    fn no_token_is_refused_when_a_secret_is_set() {
        assert_eq!(enforcing().caller(None, ALLOWED), Err(Refused::Missing));
        assert_eq!(enforcing().caller(Some("Basic aGk="), ALLOWED), Err(Refused::Missing));
    }

    #[test]
    fn without_a_secret_the_claim_is_read_but_not_checked() {
        let id = Uuid::new_v4();
        let header = from_engine(&Caller::User(id), "any-secret-at-all");

        let caller = InboundSecret::new(None).caller(Some(&header), ALLOWED);

        assert_eq!(caller, Ok(Caller::User(id)));
    }

    #[test]
    fn without_a_secret_and_without_a_token_the_caller_is_the_system() {
        assert_eq!(InboundSecret::new(None).caller(None, ALLOWED), Ok(Caller::System));
    }

    #[test]
    fn the_scheme_is_matched_case_insensitively_and_the_token_trimmed() {
        let id = Uuid::new_v4();
        let raw = token::issue_inbound_token(ENGINE, &Caller::User(id), SECRET).unwrap();

        let caller = enforcing().caller(Some(&format!("bearer {raw} ")), ALLOWED);

        assert_eq!(caller, Ok(Caller::User(id)));
    }
}
