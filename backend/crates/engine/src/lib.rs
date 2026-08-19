//! The LLM engine: training, inference and everything they own, in `engine/`.
//!
//! This crate is thin on purpose. `service` knows how to spawn a Python process
//! and forward signed requests to one; this names *which* process, and adds no
//! behaviour of its own.

use std::path::PathBuf;

use service::{Config, Defaults};

pub use service::{Frame, Method, Result};
/// Kept under its old name: `api` matches on these variants in a dozen places
/// and the type is genuinely shared with every other service.
pub use service::Error as EngineError;

/// What a token addressed to this service says in its `aud`.
pub const AUDIENCE: &str = "engine";

const DEFAULTS: Defaults = Defaults {
    audience: AUDIENCE,
    env_prefix: "ENGINE",
    dir: "../engine",
    // 8000 is commonly taken by a local Apache.
    port: 8001,
};

/// The engine's configuration, including the data root it is the authority on.
///
/// `DATA_DIR` is read here rather than in `service` because no other service has
/// one — the engine owns that tree, and a generic spawner has no business
/// knowing it exists.
pub fn config() -> Config {
    let config = Config::from_env(&DEFAULTS);
    let data_dir = std::env::var("DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data"));
    config.with_env_path("DATA_DIR", data_dir)
}

/// The engine's proxy, distinct from every other service's.
///
/// A newtype rather than a bare `service::Proxy` because Actix keys `app_data`
/// by type: two services registering the same type means one of them silently
/// wins and every route reaches it.
#[derive(Clone)]
pub struct Proxy(service::Proxy);

impl Proxy {
    pub fn new(base_url: impl Into<String>, secret: Option<String>) -> Proxy {
        Proxy(service::Proxy::new(base_url, AUDIENCE, secret))
    }
}

impl std::ops::Deref for Proxy {
    type Target = service::Proxy;

    fn deref(&self) -> &service::Proxy {
        &self.0
    }
}
