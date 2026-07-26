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

// Parent of the pretrain dataset subfolders. Each immediate subdirectory
// (e.g. uploads/TinyStories, uploads/Synthetic) is one selectable dataset; a run
// picks which subfolders — and how many files from each — go into its corpus
// (see training::list_datasets / build_dataset).
pub fn uploads_dir() -> PathBuf {
    root().join("raw").join("uploads")
}

// Chat-conversation datasets (JSONL) for supervised fine-tuning, kept separate
// from the raw prose corpora since SFT consumes a different format.
pub fn sft_dir() -> PathBuf {
    root().join("raw").join("sft")
}

// The training side of the thumbs feedback, exported from `messages.feedback` as
// JSONL and consumed by `training::build_kto_dataset`.
//
// Deliberately a sibling of `sft_dir` under `raw/` rather than living beside the
// held-out slice in `eval/preference/`: the two must never be read by the same
// consumer, and putting them in one directory means one glob away from a benchmark
// that scores the examples the tune trained on.
pub fn preference_dir() -> PathBuf {
    root().join("raw").join("preference")
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

// Benchmark fixtures. `eval/corpus/*.txt` is the reference prose a run scores
// against — held-out text the model was **not** trained on — and `eval/blimp/`
// optionally holds real BLiMP JSONL alongside the built-in minimal-pair suite.
// Kept out of `raw/uploads` on purpose: anything under uploads is selectable as
// training data, and a held-out set that can be accidentally trained on is not a
// held-out set.
pub fn eval_dir() -> PathBuf {
    root().join("eval")
}
