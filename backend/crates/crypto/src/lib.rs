//! Self-contained cryptographic helpers: password hashing and the symmetric
//! cipher used to wrap a bearer token before it reaches a client.
//!
//! **No DB and no HTTP awareness**, which is the property that lets every other
//! crate depend on this one. `jwt` deliberately does *not* live here: its
//! `Claims` carry a `db::user::AccessLevel`, so keeping it in this crate would
//! make `crypto` depend on `db` while `db` already depends on `crypto` for
//! password hashing — a cycle Rust will not compile. It lives in the web layer,
//! which is the only place that issues or verifies a token.

pub mod password;
pub mod token_cipher;
