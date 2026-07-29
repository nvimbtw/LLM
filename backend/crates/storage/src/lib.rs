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

/// Serializes tests that point `DATA_DIR` at a temp root.
///
/// `DATA_DIR` is process-global and cargo runs tests multi-threaded, so any test
/// that sets it must hold this first. It lives here, beside the one function that
/// reads the variable, because it has to be **one** lock: `training` and `synth`
/// each had their own, which guarded nothing against each other — a `synth` test
/// could repoint `DATA_DIR` mid-way through a `training` build and fail it
/// intermittently.
///
/// **Not `#[cfg(test)]`, and it cannot be.** That attribute means "when *this*
/// crate is compiled as a test", so the moment `storage` became a crate of its
/// own the lock would exist only for `storage`'s own tests and vanish for the
/// consumers that are the entire reason it exists. Two of them — `training` and
/// `synth` — are separate crates too, and a lock each is the exact bug the
/// paragraph above describes. The cost of compiling it always is one
/// `Mutex<()>` in the binary.
pub static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
// against — held-out text the model was **not** trained on — `eval/blimp/`
// optionally holds real BLiMP JSONL alongside the built-in minimal-pair suite,
// and `eval/chat/*.jsonl` holds held-out conversations in the `raw/sft` shape,
// scored through the SFT loss mask so a fine-tune has an instrument.
// Kept out of `raw/uploads` on purpose: anything under uploads is selectable as
// training data, and a held-out set that can be accidentally trained on is not a
// held-out set. The same applies to `eval/chat` against `raw/sft`.
pub fn eval_dir() -> PathBuf {
    root().join("eval")
}
