//! Where the engine keeps its data, for the few things on this side that still
//! write into that tree.
//!
//! Resolved from the same environment the supervisor uses, so the two cannot
//! disagree about where the engine is looking. This is deliberately small: the
//! engine owns the data, and everything that only *reads* it asks over HTTP
//! instead. What is left here is the synthetic-data generator, which produces
//! files the engine will later train on.

use std::path::PathBuf;

use crate::config::EngineConfig;

/// Root of the engine's data tree.
pub fn data_root() -> PathBuf {
    let config = EngineConfig::from_env();
    if config.data_dir.is_absolute() {
        return config.data_dir;
    }
    config.dir.join(&config.data_dir)
}

/// Chat conversations for supervised fine-tuning.
pub fn sft_dir() -> PathBuf {
    data_root().join("sft")
}

/// A trained model's weights, under its storage id.
///
/// The name is the engine's (`llm/storage.py:weights_path`), not this side's: the
/// engine writes the file and this side only ever asks whether it is there. Spelled
/// once, here — the previous caller had the pre-port `model.bin` inlined, which
/// resolved to nothing after the rename and failed every fine-tune at validation.
pub fn model_weights(storage_id: &str) -> PathBuf {
    data_root().join("artifacts/models").join(storage_id).join("model.pt")
}
