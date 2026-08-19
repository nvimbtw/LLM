//! The data engine: datasets, where they came from and what was done to them,
//! in `data-engine/`.
//!
//! A sibling of `engine`, not a layer above or below it. The two Python services
//! never speak to each other — everything either of them knows about the other
//! arrives through the backend — so this crate depends on `engine` for nothing.

use service::{Config, Defaults};

pub use service::{Frame, Method, Result};

/// What a token addressed to this service says in its `aud`.
pub const AUDIENCE: &str = "data-engine";

const DEFAULTS: Defaults = Defaults {
    audience: AUDIENCE,
    env_prefix: "DATA_ENGINE",
    dir: "../data-engine",
    // 8001 is the LLM engine's.
    port: 8002,
};

pub fn config() -> Config {
    Config::from_env(&DEFAULTS)
}

/// The data engine's proxy, distinct from every other service's — see
/// `engine::Proxy` for why the newtype is load-bearing rather than decorative.
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
