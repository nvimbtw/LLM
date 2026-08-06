//! The HTTP layer: route handlers, the auth extractors they rely on, socket
//! authentication, and JWT issuance/verification.
//!
//! Route files are HTTP-only — parse a request, call `db::`, map the result to
//! an `HttpResponse` with the right status. No SQL lives here, and a `db` struct
//! is never serialized straight back to a client (`db::user::User` carries the
//! password hash).
//!
//! `jwt` lives here rather than in `crypto` because its `Claims` carry a
//! `db::user::AccessLevel`, and `db` already depends on `crypto` for password
//! hashing — the other direction would be a cycle. This is the only layer that
//! issues or verifies a token, so it is where it belongs anyway.

pub mod extractors;
pub mod jwt;
pub mod routes;
pub mod ws;

pub use routes::configure;
