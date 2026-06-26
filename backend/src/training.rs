// Orchestrates the upload -> tokenize -> train pipeline. The `llm` and
// `tokenize` crates are deliberately layout-agnostic (they take explicit paths);
// this module is where `storage.rs` (the single source of truth for the data
// layout) is mapped onto those crate calls. Plain free functions, no traits.

use std::fs;
use std::io;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::storage;

// Hyperparameters + dataset knobs for one run. Deserialized from the POST body,
// echoed back in progress snapshots, and persisted to the run's meta.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub model_name: String,
    pub dimensions: i64,
    pub ctx_window: i64,
    pub heads: i64,
    pub layers: i64,
    pub epochs: i64,
    pub lr: f32,
    pub ff_multiplier: i64,
    pub max_vocab: u32,
}

// Summary of what tokenization produced, for the dashboard + meta.json.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct DatasetStats {
    pub files: usize,
    pub chars: usize,
    pub vocab_size: i64,
    pub merges: usize,
    pub corpus_len: usize,
}

// What `build_dataset` reports as it runs, so the caller can stream progress.
pub enum TokenizeEvent<'a> {
    Start { files: usize, chars: usize },
    Merge { done: u32, total: u32 },
    Log(&'a str),
}

// Reads every `.txt` in the uploads dir, trains a fresh BPE tokenizer, and
// writes the two files `llm::train` consumes: a flat token-id `corpus.bin` in
// `datasets/` and `merges.bin` in `artifacts/tokenizer/`. Progress is pushed
// through `on_event` (synchronous; runs on the caller's thread).
pub fn build_dataset(
    max_vocab: u32,
    on_event: impl Fn(TokenizeEvent),
) -> io::Result<DatasetStats> {
    let uploads = storage::uploads_dir();
    let mut paths: Vec<_> = if uploads.is_dir() {
        fs::read_dir(&uploads)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("txt"))
            .collect()
    } else {
        Vec::new()
    };
    paths.sort(); // deterministic concatenation order

    if paths.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no .txt files found in uploads",
        ));
    }

    let mut combined = String::new();
    for path in &paths {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&fs::read_to_string(path)?);
    }

    let files = paths.len();
    let chars = combined.chars().count();
    on_event(TokenizeEvent::Start { files, chars });

    // Pre-tokenize, then train BPE. The crate reports progress as a plain log
    // string ("BPE merge x/y"); we forward the raw line for the log panel and
    // parse the counts out for the numeric dashboard.
    let pre = tokenize::pretokenize::pre_tokenize_text(&combined);
    let (vocab_size, corpus, merges) = tokenize::bpe::train(pre, Some(max_vocab), |line| {
        if let Some((done, total)) = parse_merge_progress(line) {
            on_event(TokenizeEvent::Merge { done, total });
        }
        on_event(TokenizeEvent::Log(line));
    })?;

    // `bpe::train` keeps the corpus split per word; flatten into the single
    // token stream `llm::train` reads via chunks_exact(4).
    let flat: Vec<u32> = corpus.into_iter().flatten().collect();
    let corpus_len = flat.len();

    let datasets = storage::datasets_dir();
    let tokenizer = storage::tokenizer_dir();
    fs::create_dir_all(&datasets)?;
    fs::create_dir_all(&tokenizer)?;
    tokenize::bpe::save_corpus(&flat, &datasets.join("corpus.bin").to_string_lossy())?;
    tokenize::bpe::save_merges(&merges, &tokenizer.join("merges.bin").to_string_lossy())?;

    Ok(DatasetStats {
        files,
        chars,
        vocab_size,
        merges: merges.len(),
        corpus_len,
    })
}

// Parses the tokenize crate's "BPE merge {done}/{total}" line into numbers.
// Returns None for any other log line (kept tolerant of format drift).
fn parse_merge_progress(line: &str) -> Option<(u32, u32)> {
    let rest = line.strip_prefix("BPE merge ")?;
    let (done, total) = rest.split_once('/')?;
    Some((done.trim().parse().ok()?, total.trim().parse().ok()?))
}

// Maps a run onto the on-disk layout and invokes `llm::train`. Returns Ok(true)
// if the run was cancelled, Ok(false) if it completed.
pub fn run_training(
    cfg: &TrainConfig,
    on_progress: llm::train::ProgressCallback,
    cancel: Arc<AtomicBool>,
) -> io::Result<bool> {
    let model_dir = storage::models_dir().join(&cfg.model_name);
    let corpus_path = storage::datasets_dir().join("corpus.bin");
    let merges_path = storage::tokenizer_dir().join("merges.bin");
    let checkpoints_dir = model_dir.join("checkpoints");

    llm::train::train(
        cfg.dimensions,
        cfg.ctx_window,
        cfg.heads,
        cfg.layers,
        cfg.epochs,
        cfg.lr,
        cfg.ff_multiplier,
        &model_dir,
        &corpus_path,
        &merges_path,
        Some(&checkpoints_dir),
        on_progress,
        Some(cancel),
    )
}

#[derive(Serialize)]
struct RunMeta<'a> {
    config: &'a TrainConfig,
    dataset: DatasetStats,
    status: &'a str,
    created_at: &'a str,
    finished_at: String,
}

// Writes a durable record of the run next to its weights, so a run is
// inspectable on disk without a database.
pub fn write_meta(
    cfg: &TrainConfig,
    dataset: DatasetStats,
    status: &str,
    created_at: &str,
) -> io::Result<()> {
    let model_dir = storage::models_dir().join(&cfg.model_name);
    fs::create_dir_all(&model_dir)?;
    let meta = RunMeta {
        config: cfg,
        dataset,
        status,
        created_at,
        finished_at: Utc::now().to_rfc3339(),
    };
    let json = serde_json::to_string_pretty(&meta)?;
    fs::write(model_dir.join("meta.json"), json)
}
