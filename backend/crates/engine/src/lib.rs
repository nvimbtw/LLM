//! The boundary to the Python engine: the process, and the HTTP calls into it.
//!
//! Training and inference run in `engine/` (PyTorch) rather than in this
//! workspace. This crate is the only place that knows that — everything above
//! it calls typed functions and never learns there is a subprocess involved.

pub mod chat;
pub mod client;
pub mod config;
pub mod paths;
pub mod supervisor;

pub use chat::{Role, SamplingParams, ThinkBudget, Turn};
pub use client::{Client, EngineError, RunEvent, RunSnapshot};
pub use config::EngineConfig;
pub use supervisor::Engine;
