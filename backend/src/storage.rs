// Several path helpers are defined ahead of the pipeline stages that will use
// them (crawler, tokenizer, training). Allow the unused ones until those land.
#![allow(dead_code)]

use std::path::PathBuf;

// Single source of truth for every on-disk data path. Pipeline stages, the
// crawler, and the HTTP routes all build paths through here instead of hardcoding
// string literals, so the layout can change in one place.
//
// Layout (see backend/data/):
//   raw/{uploads,crawled,downloads}  original bytes, as acquired
//   processed/                       extracted, cleaned text
//   tokens/                          per-document token-id streams
//   datasets/                        training-ready packs (train.bin / val.bin)
//   artifacts/tokenizer/             trained tokenizer (vocab + merges)
//   artifacts/models/<run>/          trained model checkpoints + weights

// Root of all data. Override with DATA_DIR (e.g. an absolute path or a mounted
// volume in production); defaults to `data` relative to the working directory.
fn root() -> PathBuf {
    std::env::var("DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data"))
}

pub fn uploads_dir() -> PathBuf {
    root().join("raw").join("uploads")
}

// Synthetic training documents, kept separate from user uploads so a run can mix
// in a chosen number of them independently.
pub fn synthetic_dir() -> PathBuf {
    root().join("raw").join("synthetic")
}

pub fn crawled_dir() -> PathBuf {
    root().join("raw").join("crawled")
}

pub fn downloads_dir() -> PathBuf {
    root().join("raw").join("downloads")
}

pub fn processed_dir() -> PathBuf {
    root().join("processed")
}

pub fn tokens_dir() -> PathBuf {
    root().join("tokens")
}

pub fn datasets_dir() -> PathBuf {
    root().join("datasets")
}

pub fn tokenizer_dir() -> PathBuf {
    root().join("artifacts").join("tokenizer")
}

pub fn models_dir() -> PathBuf {
    root().join("artifacts").join("models")
}
