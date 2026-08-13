//! The HTTP layer: route handlers, the auth extractors they rely on, socket
//! authentication, and JWT issuance/verification.
//!
//! Route files are HTTP-only — parse a request, call `db::`, map the result to
//! an `HttpResponse` with the right status. No SQL lives here, and a `db` struct
//! is never serialized straight back to a client (`db::user::User` carries the
//! password hash).
//!
//! Tokens are minted and verified in `auth`, which also owns `AccessLevel` —
//! that is what lets this layer stay HTTP-only.

pub mod extractors;
pub mod routes;
pub mod upstream;
pub mod ws;

pub use routes::configure;
