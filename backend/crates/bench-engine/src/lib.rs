use std::path::PathBuf;
use service::{Config, Defaults};
pub use service::{Blob, FilePart, Frame, Method, Result};
pub use service::Error as BenchEngineError;
pub const AUDIENCE: &str = "bench-engine";
const DEFAULTS: Defaults = Defaults {
    audience: AUDIENCE,
    env_prefix: "BENCH_ENGINE",
    dir: "../bench-engine",
    port: 8003,
};
pub fn config() -> Config {
    let config = Config::from_env(&DEFAULTS);
    let data_dir = std::env::var("BENCH_ENGINE_DATA_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("data"));
    config.with_env_path("BENCH_ENGINE_DATA_DIR", data_dir)
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
    fn deref(&self) -> &service::Proxy { &self.0 }
}
