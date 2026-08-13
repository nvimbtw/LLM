//! The boundary to the Python engine: the process, and the hop into it.
//!
//! Training, inference and everything they own live in `engine/` (PyTorch).
//! This crate spawns that process, keeps it alive, and forwards signed requests
//! to it. It models none of what the engine owns — see `proxy`.

pub mod config;
pub mod error;
pub mod paths;
pub mod proxy;
pub mod supervisor;

/// What a token addressed to this service says in its `aud`.
pub const AUDIENCE: &str = "engine";

pub use config::EngineConfig;
pub use error::{EngineError, Result};
pub use proxy::{Frame, Method, Proxy};
pub use supervisor::Engine;
