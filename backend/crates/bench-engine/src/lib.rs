use service::{Config, Defaults};

pub use service::Error as BenchEngineError;
pub use service::{Blob, FilePart, Frame, Method, Result};

pub const AUDIENCE: &str = "bench-engine";

const DEFAULTS: Defaults = Defaults {
    audience: AUDIENCE,
    env_prefix: "BENCH_ENGINE",
    dir: "../bench-engine",
    port: 8004,
};

pub fn config() -> Config {
    Config::from_env(&DEFAULTS)
}

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
