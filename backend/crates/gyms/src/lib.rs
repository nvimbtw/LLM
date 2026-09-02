use service::{Config, Defaults};

pub use service::{Blob, FilePart, Frame, Method, Result};
pub use service::Error as GymsError;

pub const AUDIENCE: &str = "gyms";

const DEFAULTS: Defaults = Defaults {
    audience: AUDIENCE,
    env_prefix: "GYMS",
    dir: "../gyms",
    port: 8003,
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
