//! The HTTP layer: routes, the guards in front of them, and socket handshakes.
//!
//! Routes are HTTP-only. A route parses a request, calls `db::` or forwards to
//! a service, and maps the result to a status. No SQL lives here, and a `db`
//! struct is never serialized straight to a client — `db::user::User` carries
//! the password hash.

pub mod extractors;
pub mod routes;
pub mod search_cache;
pub mod turn;
pub mod upstream;
pub mod ws;

pub use routes::configure;
