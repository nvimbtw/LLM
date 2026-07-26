// Orchestrates the upload -> tokenize -> train pipeline. The `llm` and
// `tokenize` crates are deliberately layout-agnostic (they take explicit paths);
// this module is where `storage.rs` (the single source of truth for the data
// layout) is mapped onto those crate calls. Plain free functions, no traits.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::storage;

// Which kind of run this is: pretrain a fresh model, or supervised-fine-tune
// (SFT) an existing one on chat data. Serialized lowercase (`"pretrain"` /
// `"sft"`) in the POST body and meta.json.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TrainMode {
    #[default]
    Pretrain,
    Sft,
    // Preference tuning on thumbs-rated replies (KTO). Like SFT it fine-tunes an
    // existing model named by `base_model_id`, but from the chat feedback rather
    // than from a conversation corpus.
    Kto,
}

// The KTO default, matching `llm::kto::KtoParams::default`. 0.1 is the paper's
// value and the usual starting point: low enough that the tune can move, high
// enough that the KL anchor still holds.
fn default_beta() -> f64 {
    0.1
}

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
    // Pretrain (default) vs SFT. `#[serde(default)]` so a body/meta.json written
    // before this field existed still deserializes as a pretrain run (back-compat).
    #[serde(default)]
    pub mode: TrainMode,
    // For SFT only: the id of the already-trained model to fine-tune from (names
    // its on-disk dir under models/). Ignored for pretrain; validated present at
    // the route boundary for SFT.
    #[serde(default)]
    pub base_model_id: Option<Uuid>,
    // KTO only: strength of the KL anchor holding the tune near the base model.
    // Defaulted rather than required so a body written for the other two modes
    // still deserializes; see `llm::kto::KtoParams::beta` for what it does.
    #[serde(default = "default_beta")]
    pub beta: f64,
    // Which upload datasets (subfolders of uploads/) go into this run's corpus,
    // each with how many of its .txt files to sample at random (0 => all of that
    // dataset). Pretrain only — SFT ignores it. `#[serde(default)]` keeps
    // pre-existing meta.json (written before this field, or from an SFT run)
    // deserializing as an empty selection.
    #[serde(default)]
    pub datasets: Vec<DatasetSelection>,
    // Curriculum stages: train the model on each stage's datasets in order, for
    // that stage's own epoch count (e.g. TinyStories for 3 epochs, then Wikipedia
    // for 2). Empty => a single implicit stage from `datasets` + `epochs` (the
    // legacy behaviour). Pretrain only. `#[serde(default)]` for meta.json
    // back-compat.
    #[serde(default)]
    pub curriculum: Vec<CurriculumStage>,
    // Reorder the documents fed to the model each epoch (with an <|endoftext|>
    // separator between them), so it doesn't see a fixed data order and never
    // learns to predict across a document boundary. `#[serde(default)]` => a body
    // or meta.json written before this field deserializes as `false`, exactly
    // reproducing the old fixed-order runs.
    #[serde(default)]
    pub shuffle: bool,
    // PRNG seed for the per-epoch shuffle, persisted so a run is reproducible. `0`
    // means "pick a fresh time-based seed at run start" (resolved server-side into
    // this field before the run's meta.json is written).
    #[serde(default)]
    pub seed: u64,
    // The model's id, which names its on-disk directory. Set server-side by the
    // enqueue route (reused on retrain, freshly generated for a new model), which
    // overwrites whatever arrived in the request body — so a client cannot choose
    // it even though this deserializes. It has to deserialize: the config makes a
    // round trip through the job row's JSONB between being queued and being run,
    // and `skip_deserializing` silently reset the id to nil on the way back out,
    // which would have pointed the run at the wrong weights directory.
    #[serde(default)]
    pub storage_id: Uuid,
}

// One curriculum stage: a set of datasets trained for a fixed number of epochs
// before moving on to the next stage. Reuses `DatasetSelection` for per-dataset
// file sampling, exactly like the flat (non-curriculum) selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumStage {
    #[serde(default)]
    pub name: Option<String>,
    pub datasets: Vec<DatasetSelection>,
    pub epochs: i64,
}

// One entry in a run's dataset selection: the name of an uploads/ subfolder and
// how many of its .txt files to sample at random. `count == 0` means "use all of
// them" (see `select_txt`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSelection {
    pub name: String,
    #[serde(default)]
    pub count: usize,
}

// A dataset the UI can offer: one subfolder of uploads/ and its total .txt count
// (so the picker can show "n / <files>" when dialing a per-dataset sample size).
#[derive(Debug, Clone, Serialize)]
pub struct DatasetInfo {
    pub name: String,
    pub files: usize,
}

// Summary of what tokenization produced, for the dashboard + meta.json.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct DatasetStats {
    pub files: usize,
    pub chars: usize,
    pub vocab_size: i64,
    pub merges: usize,
    pub corpus_len: usize,
    // Number of documents (source files) whose token spans were recorded in
    // segments.json for per-epoch shuffling / curriculum staging.
    pub documents: usize,
}

// Per-run sidecar written next to corpus.bin: the document boundaries (content
// spans into the flat corpus, exclusive of the baked-in <|endoftext|>
// separators) tagged with the curriculum stage they belong to. The training loop
// reads this to shuffle documents per epoch and to feed one stage at a time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segments {
    pub eot_id: u32,
    pub stages: Vec<StageMeta>,
    pub documents: Vec<DocSpan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageMeta {
    #[serde(default)]
    pub name: Option<String>,
    pub epochs: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocSpan {
    pub stage: usize,
    pub dataset: String,
    pub start: usize,
    pub len: usize,
}

// What `build_dataset` reports as it runs, so the caller can stream progress.
pub enum TokenizeEvent<'a> {
    Start { files: usize, chars: usize },
    Merge { done: u32, total: u32 },
    Log(&'a str),
}

// Reads every `.txt` in the uploads dir, trains a fresh BPE tokenizer, and
// A time-seeded xorshift64 state. Avoids pulling in the `rand` crate for the few
// spots we need randomness (picking a book subset, sampling synthetic templates)
// — the bar here is "different each run", not cryptographic quality. The `| 1`
// guarantees the seed is never zero, which xorshift can never leave.
fn rng_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
        | 1
}

// Advances an xorshift64 state in place and returns the new value.
fn xorshift_next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

// Fisher–Yates shuffle using the shared xorshift PRNG, seeded fresh each run.
fn shuffle<T>(items: &mut [T]) {
    shuffle_seeded(items, rng_seed());
}

// Fisher–Yates shuffle from an explicit seed, so callers that need a
// reproducible order (the per-epoch training shuffle) can derive one from a
// stored seed. `| 1` keeps xorshift out of its fixed zero state.
fn shuffle_seeded<T>(items: &mut [T], seed: u64) {
    let mut state = seed | 1;
    for i in (1..items.len()).rev() {
        let j = (xorshift_next(&mut state) % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

// Resolve a run's shuffle seed: `0` (the request/meta default) means "pick a
// fresh time seed"; any other value is used as-is. Called once at run start so
// the resolved seed can be persisted to meta.json for reproducibility.
pub fn resolve_seed(seed: u64) -> u64 {
    if seed == 0 { rng_seed() } else { seed }
}

// The stages a run trains over. A curriculum run uses its stages verbatim; a
// plain run collapses to a single implicit stage from the flat dataset selection
// and top-level epoch count. Every downstream consumer (dataset build + training
// loop) works purely in terms of stages, so the two paths share one code path.
pub fn resolve_stages(cfg: &TrainConfig) -> Vec<CurriculumStage> {
    if cfg.curriculum.is_empty() {
        vec![CurriculumStage {
            name: None,
            datasets: cfg.datasets.clone(),
            epochs: cfg.epochs,
        }]
    } else {
        cfg.curriculum.clone()
    }
}

// Collects the `.txt` files in `dir`, optionally sampling `num` of them at
// random. `zero_means_all` distinguishes "0 = use everything" (books) from
// "0 = none" (synthetic data, which is opt-in). A missing dir yields no files.
fn select_txt(dir: &Path, num: usize, zero_means_all: bool) -> io::Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = if dir.is_dir() {
        fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("txt"))
            .collect()
    } else {
        Vec::new()
    };
    paths.sort(); // deterministic base order before any sampling

    let take = if zero_means_all && num == 0 {
        paths.len()
    } else {
        num.min(paths.len())
    };
    if take < paths.len() {
        shuffle(&mut paths);
        paths.truncate(take);
        paths.sort(); // stable concatenation order for the chosen subset
    }
    Ok(paths)
}

// Rejects a dataset name that isn't a plain, single-segment folder name, so a
// request can't escape the uploads dir via path traversal (`..`, absolute paths,
// nested segments). Every selected name is checked before it's joined to
// uploads/.
fn is_safe_dataset_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains('\0')
}

// Scans uploads/ for its immediate subfolders — each one a selectable dataset —
// and counts the `.txt` files in each. Sorted by name for a stable UI order; a
// missing uploads dir yields an empty list rather than an error.
pub fn list_datasets() -> io::Result<Vec<DatasetInfo>> {
    let uploads = storage::uploads_dir();
    if !uploads.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(&uploads)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue; // loose files at the uploads root aren't datasets
        }
        // A non-UTF-8 dir name can't round-trip to JSON / a request — skip it.
        let name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let files = fs::read_dir(entry.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("txt"))
            .count();
        out.push(DatasetInfo { name, files });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

// writes the files `llm::train` consumes: a flat token-id `corpus.bin` in
// `datasets/`, `merges.bin` in `artifacts/tokenizer/`, and a `segments.json`
// sidecar recording each document's token span + curriculum stage. Progress is
// pushed through `on_event` (synchronous; runs on the caller's thread).
//
// One BPE tokenizer is trained over ALL stages' files (a shared vocabulary is
// mandatory — the token ids must line up across stages so the model's embeddings
// mean the same thing throughout). We pre-tokenize each file on its own so we can
// map its words back to a token span after BPE, and bake a single <|endoftext|>
// separator between documents so the default (unshuffled) corpus already reads
// with document boundaries.
pub fn build_dataset(
    max_vocab: u32,
    stages: &[CurriculumStage],
    on_event: impl Fn(TokenizeEvent),
) -> io::Result<DatasetStats> {
    // Gather every selected file tagged with the stage it belongs to. A dataset
    // is a subfolder of uploads/; `select_txt` pulls `count` .txt files from it
    // (0 => all). Files keep their per-stage grouping so the training loop can
    // feed one stage at a time.
    let uploads = storage::uploads_dir();
    // (stage index, dataset name, file path)
    let mut tagged: Vec<(usize, String, PathBuf)> = Vec::new();
    for (stage_idx, stage) in stages.iter().enumerate() {
        for sel in &stage.datasets {
            if !is_safe_dataset_name(&sel.name) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "invalid dataset name",
                ));
            }
            for path in select_txt(&uploads.join(&sel.name), sel.count, true)? {
                tagged.push((stage_idx, sel.name.clone(), path));
            }
        }
    }

    if tagged.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no .txt files found in the selected datasets",
        ));
    }

    // Pre-tokenize each file individually into words, remembering each file's
    // contiguous word range in the concatenated `pre` list, so once BPE has run
    // we can sum the per-word token lengths back into a per-document token span.
    struct DocWords {
        stage: usize,
        dataset: String,
        word_start: usize,
        words: usize,
    }
    let mut pre: Vec<Vec<char>> = Vec::new();
    let mut doc_words: Vec<DocWords> = Vec::new();
    let mut chars = 0usize;
    for (stage_idx, dataset, path) in &tagged {
        let text = fs::read_to_string(path)?;
        chars += text.chars().count();
        let words = tokenize::pretokenize::pre_tokenize_text(&text);
        let word_start = pre.len();
        let count = words.len();
        pre.extend(words);
        doc_words.push(DocWords {
            stage: *stage_idx,
            dataset: dataset.clone(),
            word_start,
            words: count,
        });
    }

    let files = tagged.len();
    on_event(TokenizeEvent::Start { files, chars });

    // Train BPE once over all files' words. The crate reports progress as a plain
    // log string ("BPE merge x/y"); forward the raw line for the log panel and
    // parse the counts out for the numeric dashboard. `corpus` comes back split
    // per word, in the same order as `pre`.
    let (vocab_size, corpus, merges) = tokenize::bpe::train(pre, Some(max_vocab), |line| {
        if let Some((done, total)) = parse_merge_progress(line) {
            on_event(TokenizeEvent::Merge { done, total });
        }
        on_event(TokenizeEvent::Log(line));
    })?;

    // A lone <|endoftext|> encodes to its single reserved id (256); use it as the
    // document separator, exactly like the SFT boundary between conversations.
    let eot = tokenize::bpe::encode("<|endoftext|>", &merges);
    let eot_id = *eot.first().unwrap_or(&256);

    // Assemble the flat corpus by concatenating each document's tokens with an
    // <|endoftext|> baked in between documents, recording each document's content
    // span (exclusive of separators) tagged with its stage.
    let mut flat: Vec<u32> = Vec::new();
    let mut documents: Vec<DocSpan> = Vec::new();
    for d in &doc_words {
        let tokens: usize = corpus[d.word_start..d.word_start + d.words]
            .iter()
            .map(|w| w.len())
            .sum();
        if tokens == 0 {
            continue; // empty file — no span to record, no separator to add
        }
        if !flat.is_empty() {
            flat.push(eot_id);
        }
        let start = flat.len();
        for w in &corpus[d.word_start..d.word_start + d.words] {
            flat.extend(w);
        }
        documents.push(DocSpan {
            stage: d.stage,
            dataset: d.dataset.clone(),
            start,
            len: tokens,
        });
    }
    let corpus_len = flat.len();

    let segments = Segments {
        eot_id,
        stages: stages
            .iter()
            .map(|s| StageMeta {
                name: s.name.clone(),
                epochs: s.epochs,
            })
            .collect(),
        documents,
    };

    let datasets_dir = storage::datasets_dir();
    let tokenizer = storage::tokenizer_dir();
    fs::create_dir_all(&datasets_dir)?;
    fs::create_dir_all(&tokenizer)?;
    tokenize::bpe::save_corpus(&flat, &datasets_dir.join("corpus.bin").to_string_lossy())?;
    tokenize::bpe::save_merges(&merges, &tokenizer.join("merges.bin").to_string_lossy())?;
    save_segments(&segments, &datasets_dir.join("segments.json"))?;

    Ok(DatasetStats {
        files,
        chars,
        vocab_size,
        merges: merges.len(),
        corpus_len,
        documents: segments.documents.len(),
    })
}

// Writes the document-boundary sidecar as pretty JSON next to corpus.bin.
fn save_segments(segments: &Segments, path: &Path) -> io::Result<()> {
    let json = serde_json::to_vec_pretty(segments)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(path, json)
}

// Loads the document-boundary sidecar. Returns `Ok(None)` when it's absent — a
// dataset built before this feature existed — so the training loop can fall back
// to treating the whole corpus as one document (full back-compat).
fn load_segments(path: &Path) -> io::Result<Option<Segments>> {
    match fs::read(path) {
        Ok(bytes) => {
            let segments: Segments = serde_json::from_slice(&bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(Some(segments))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
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
    let model_dir = storage::models_dir().join(cfg.storage_id.to_string());
    let datasets_dir = storage::datasets_dir();
    let corpus_path = datasets_dir.join("corpus.bin");
    let merges_path = storage::tokenizer_dir().join("merges.bin");
    let checkpoints_dir = model_dir.join("checkpoints");

    // The document/stage plan for the training loop. A dataset built before this
    // feature (no segments.json) falls back to one stage of `epochs` epochs over
    // the whole corpus as a single document — identical to the old behaviour.
    let plan = match load_segments(&datasets_dir.join("segments.json"))? {
        Some(segments) => llm::train::DataPlan {
            stages: (0..segments.stages.len())
                .map(|si| llm::train::Stage {
                    name: segments.stages[si].name.clone(),
                    epochs: segments.stages[si].epochs,
                    docs: segments
                        .documents
                        .iter()
                        .filter(|d| d.stage == si)
                        .map(|d| llm::train::DocSpan { start: d.start, len: d.len })
                        .collect(),
                })
                .collect(),
            eot_id: segments.eot_id,
            shuffle: cfg.shuffle,
            seed: resolve_seed(cfg.seed),
        },
        None => llm::train::DataPlan::whole_corpus(cfg.epochs),
    };

    llm::train::train(
        cfg.dimensions,
        cfg.ctx_window,
        cfg.heads,
        cfg.layers,
        cfg.lr,
        cfg.ff_multiplier,
        &model_dir,
        &corpus_path,
        &merges_path,
        Some(&checkpoints_dir),
        plan,
        on_progress,
        Some(cancel),
    )
}

// Maps an SFT run onto the on-disk layout and invokes `llm::sft_train`. Mirrors
// `run_training` but loads a pretrained base (resolved from `cfg.base_model_id`)
// and drives the masked SFT path. Returns Ok(true) if cancelled, Ok(false) if it
// completed. The dataset (`sft_corpus.bin` + `sft_mask.bin`) must already have
// been built by `build_sft_dataset` — the caller runs that as phase 1.
pub fn run_sft_training(
    cfg: &TrainConfig,
    on_progress: llm::train::ProgressCallback,
    cancel: Arc<AtomicBool>,
) -> io::Result<bool> {
    // The route validates presence, but this crate is also called directly (tests,
    // future callers), so guard here rather than trust the invariant.
    let base_id = cfg.base_model_id.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "SFT run requires a base_model_id")
    })?;
    let base_model_dir = storage::models_dir().join(base_id.to_string());
    let sft_corpus_path = storage::datasets_dir().join("sft_corpus.bin");
    let sft_mask_path = storage::datasets_dir().join("sft_mask.bin");
    let out_model_dir = storage::models_dir().join(cfg.storage_id.to_string());
    let checkpoints_dir = out_model_dir.join("checkpoints");

    llm::train::sft_train(
        &base_model_dir,
        &sft_corpus_path,
        &sft_mask_path,
        &out_model_dir,
        cfg.epochs,
        cfg.lr,
        Some(&checkpoints_dir),
        on_progress,
        Some(cancel),
    )
}

#[derive(Serialize)]
struct RunMeta<'a> {
    config: &'a TrainConfig,
    // Serialized dataset summary — `DatasetStats` for pretrain, `SftDatasetStats`
    // for SFT — so one meta.json shape covers both modes.
    dataset: &'a serde_json::Value,
    status: &'a str,
    created_at: &'a str,
    finished_at: String,
}

// Overwrite an SFT run's architecture fields with the base model's *actual*
// architecture, read from the base's `meta.json`. An SFT fine-tune loads the base
// weights wholesale, so it inherits the base's dimensions/context/heads/layers —
// the run's own arch fields are just placeholders from the request form (the SFT
// UI doesn't collect them). Without this, the fine-tuned model is *registered*
// with those placeholders (e.g. 64ctx, ~200k params) instead of the real inherited
// architecture, so the registry/param-count display is wrong even though the
// weights are correct. `max_vocab` is set from the base's actual vocab so the
// derived parameter count matches the base exactly. Best-effort: on any read/parse
// failure the caller logs and proceeds (metadata stays as-is; training is
// unaffected).
pub fn apply_base_architecture(cfg: &mut TrainConfig, base_model_dir: &Path) -> io::Result<()> {
    let raw = fs::read_to_string(base_model_dir.join("meta.json"))?;
    let meta: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let config = meta
        .get("config")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "base meta.json has no config"))?;

    if let Some(v) = config.get("dimensions").and_then(|v| v.as_i64()) {
        cfg.dimensions = v;
    }
    if let Some(v) = config.get("ctx_window").and_then(|v| v.as_i64()) {
        cfg.ctx_window = v;
    }
    if let Some(v) = config.get("heads").and_then(|v| v.as_i64()) {
        cfg.heads = v;
    }
    if let Some(v) = config.get("layers").and_then(|v| v.as_i64()) {
        cfg.layers = v;
    }
    if let Some(v) = config.get("ff_multiplier").and_then(|v| v.as_i64()) {
        cfg.ff_multiplier = v;
    }
    // Prefer the base's *actual* trained vocab (dataset.vocab_size) over its
    // configured cap, so the SFT model's derived param count matches the base's.
    let vocab = meta
        .get("dataset")
        .and_then(|d| d.get("vocab_size"))
        .and_then(|v| v.as_i64())
        .or_else(|| config.get("max_vocab").and_then(|v| v.as_i64()));
    if let Some(v) = vocab.filter(|v| *v > 0) {
        cfg.max_vocab = v as u32;
    }
    Ok(())
}

// Writes a durable record of the run next to its weights, so a run is
// inspectable on disk without a database. `dataset` is the already-serialized
// per-mode stats (see `RunMeta::dataset`).
pub fn write_meta(
    cfg: &TrainConfig,
    dataset: &serde_json::Value,
    status: &str,
    created_at: &str,
) -> io::Result<()> {
    let model_dir = storage::models_dir().join(cfg.storage_id.to_string());
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

// ── Supervised fine-tuning (SFT) dataset ────────────────────────────────────
//
// SFT trains a *pretrained* model on chat conversations, computing loss only on
// the assistant's replies. Its dataset is two aligned files in `datasets/`:
//   sft_corpus.bin  flat little-endian u32 token ids (same format as corpus.bin)
//   sft_mask.bin    one byte per token, 1=train / 0=ignore, aligned 1:1
// The mask comes straight from `llm::chat_template::build_training_example`, the
// single source of the format the model is prompted with (see ISSUE-0031/0030).
// Wired into the run orchestration (`run_sft_training` + `jobs::run_sft_job`) and
// the route by ISSUE-0037.

// One turn of a conversation, as it appears in the input JSONL.
#[derive(Debug, Deserialize)]
struct SftTurn {
    role: String,
    text: String,
}

// One conversation: an optional system prompt plus alternating turns, ending on
// an assistant reply. One JSON object per line in the input `.jsonl` files.
#[derive(Debug, Deserialize)]
struct SftConversation {
    #[serde(default)]
    system: Option<String>,
    turns: Vec<SftTurn>,
}

// Summary of what the SFT build produced, for the dashboard + meta.json. The
// trainable fraction is a useful sanity check — it should sit well under 50%
// (most tokens are system/user context the model only conditions on).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SftDatasetStats {
    pub conversations: usize,
    pub total_tokens: usize,
    pub trainable_tokens: usize,
}

// Map the JSONL role string onto the chat-template Role. Unknown roles are a
// hard error — bad training data should fail loudly, not train on the wrong mask.
fn parse_role(role: &str) -> io::Result<llm::chat_template::Role> {
    match role {
        "user" => Ok(llm::chat_template::Role::User),
        "assistant" => Ok(llm::chat_template::Role::Assistant),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown conversation role {:?} (expected user|assistant)", other),
        )),
    }
}

// Reads every `.jsonl` conversation file in the SFT dir, renders each conversation
// through the canonical chat template using the **base model's** tokenizer, and
// writes the aligned `sft_corpus.bin` + `sft_mask.bin` to `datasets/`.
//
// The tokenizer is loaded from `base_model_dir/merges.bin` and is NOT retrained:
// SFT is a fine-tune, so the token ids must match exactly what the base model
// already learned. Training a fresh BPE here would renumber the vocab (and the
// special-marker ids) and the fine-tune would be gibberish.
pub fn build_sft_dataset(base_model_dir: &Path) -> io::Result<SftDatasetStats> {
    let merges_path = base_model_dir.join("merges.bin");
    let merges = tokenize::bpe::load_merges(&merges_path.to_string_lossy())?;

    // Between conversations, a single <|endoftext|> with mask 0 marks the boundary
    // so the model sees a separator but is never trained to predict across two
    // unrelated conversations.
    let eot = tokenize::bpe::encode("<|endoftext|>", &merges);

    let sft = storage::sft_dir();
    let mut files: Vec<PathBuf> = if sft.is_dir() {
        fs::read_dir(&sft)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("jsonl"))
            .collect()
    } else {
        Vec::new()
    };
    files.sort();
    if files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no .jsonl conversation files found in the SFT dir",
        ));
    }

    let mut corpus: Vec<u32> = Vec::new();
    let mut mask: Vec<u8> = Vec::new();
    let mut conversations = 0usize;

    for path in &files {
        let content = fs::read_to_string(path)?;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let conv: SftConversation = serde_json::from_str(line).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bad conversation JSON in {}: {}", path.display(), e),
                )
            })?;

            let mut turns = Vec::with_capacity(conv.turns.len());
            for t in &conv.turns {
                turns.push(llm::chat_template::Turn {
                    role: parse_role(&t.role)?,
                    text: t.text.clone(),
                });
            }

            let (ids, m) = llm::chat_template::build_training_example(
                conv.system.as_deref(),
                &turns,
                &merges,
            );
            if ids.is_empty() {
                continue;
            }

            // Boundary before every conversation after the first.
            if !corpus.is_empty() {
                corpus.extend(&eot);
                mask.extend(std::iter::repeat_n(0u8, eot.len()));
            }
            corpus.extend(&ids);
            mask.extend(&m);
            conversations += 1;
        }
    }

    if corpus.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "SFT input produced no tokens (empty conversations?)",
        ));
    }

    let datasets = storage::datasets_dir();
    fs::create_dir_all(&datasets)?;
    tokenize::bpe::save_corpus(&corpus, &datasets.join("sft_corpus.bin").to_string_lossy())?;
    fs::write(datasets.join("sft_mask.bin"), &mask)?;

    let trainable_tokens = mask.iter().filter(|&&b| b != 0).count();
    Ok(SftDatasetStats {
        conversations,
        total_tokens: corpus.len(),
        trainable_tokens,
    })
}

#[derive(Debug, Serialize)]
pub struct KtoDatasetStats {
    pub examples: usize,
    pub desirable: usize,
    pub undesirable: usize,
    pub total_tokens: usize,
    pub trainable_tokens: usize,
    // Rated replies that were read but could not be used, and why — surfaced
    // because on a dataset this small, silently dropping half the feedback is the
    // difference between a tune that can work and one that cannot.
    pub skipped_too_long: usize,
    pub skipped_no_reply: usize,
}

// One rated reply as exported by `queue::export_preference_data`.
#[derive(Deserialize)]
struct RatedLine {
    prompt: String,
    response: String,
    label: String,
}

// Builds the KTO dataset from the exported thumbs feedback: token spans, the reply
// mask, and each example's reference log-probability under the base model.
//
// **Why the base model is loaded here.** KTO needs `log π_ref(reply | prompt)` for
// every example. Computing it now, once, and storing it in the index means the
// training loop never holds two models on the GPU — the alternative (a resident
// frozen copy) would double weight memory and add a forward pass per step for a
// number that never changes.
//
// **Why examples longer than the context window are dropped rather than truncated.**
// A KTO reward is a property of a whole reply. A truncated example would be trained
// against a reference log-probability computed over a different span, so its reward
// would be measuring two different things — worse than not training on it at all.
pub fn build_kto_dataset(base_model_dir: &Path) -> io::Result<KtoDatasetStats> {
    let merges_path = base_model_dir.join("merges.bin");
    let merges = tokenize::bpe::load_merges(&merges_path.to_string_lossy())?;

    // The *training* slice only. The held-out slice lives under `eval/preference/`
    // and is the benchmark's alone — training on it would make the reported margin
    // meaningless.
    let dir = storage::preference_dir();
    let mut files: Vec<PathBuf> = if dir.is_dir() {
        fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("jsonl"))
            .collect()
    } else {
        Vec::new()
    };
    files.sort();

    let mut lines: Vec<RatedLine> = Vec::new();
    for path in &files {
        let content = fs::read_to_string(path)?;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // A malformed line is skipped rather than fatal: this file is a machine
            // export, and one bad row should not block a tune over hundreds of good
            // ones.
            if let Ok(parsed) = serde_json::from_str::<RatedLine>(line) {
                lines.push(parsed);
            }
        }
    }

    if lines.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no rated replies found — collect some thumbs up/down in the chat first",
        ));
    }

    let model = llm::training::transformer::Model::load(&base_model_dir.to_string_lossy())?;
    let ctx = model.positional_table.len();
    let layers = model.layers as usize;

    let mut corpus: Vec<u32> = Vec::new();
    let mut mask: Vec<u8> = Vec::new();
    let mut examples: Vec<llm::kto::KtoExample> = Vec::new();
    let mut skipped_too_long = 0usize;
    let mut skipped_no_reply = 0usize;

    for line in &lines {
        let desirable = match line.label.as_str() {
            "up" => true,
            "down" => false,
            _ => continue,
        };

        let turns = vec![
            llm::chat_template::Turn {
                role: llm::chat_template::Role::User,
                text: line.prompt.clone(),
            },
            llm::chat_template::Turn {
                role: llm::chat_template::Role::Assistant,
                text: line.response.clone(),
            },
        ];
        let (ids, m) = llm::chat_template::build_training_example(None, &turns, &merges);

        if ids.len() < 2 || ids.len() > ctx {
            skipped_too_long += 1;
            continue;
        }
        if !m.iter().any(|&b| b != 0) {
            skipped_no_reply += 1;
            continue;
        }

        // The reference log-probability over exactly the masked (reply) positions —
        // the same positions `train_step_kto` will sum over, so the two are
        // comparable.
        let Some(ref_logprob) = llm::kto::reference_logprob(&model, &ids, &m, layers) else {
            skipped_no_reply += 1;
            continue;
        };

        examples.push(llm::kto::KtoExample {
            offset: corpus.len(),
            len: ids.len(),
            desirable,
            ref_logprob,
        });
        corpus.extend(&ids);
        mask.extend(&m);
    }

    if examples.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "every rated reply was unusable ({skipped_too_long} too long for the \
                 {ctx}-token context, {skipped_no_reply} with no reply tokens)"
            ),
        ));
    }

    let datasets = storage::datasets_dir();
    fs::create_dir_all(&datasets)?;
    tokenize::bpe::save_corpus(&corpus, &datasets.join("kto_corpus.bin").to_string_lossy())?;
    fs::write(datasets.join("kto_mask.bin"), &mask)?;

    let desirable = examples.iter().filter(|e| e.desirable).count();
    let undesirable = examples.len() - desirable;
    let index = llm::kto::KtoIndex { examples };
    fs::write(
        datasets.join("kto_index.json"),
        serde_json::to_string(&index).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("could not encode index: {e}"))
        })?,
    )?;

    Ok(KtoDatasetStats {
        examples: index.examples.len(),
        desirable,
        undesirable,
        total_tokens: corpus.len(),
        trainable_tokens: mask.iter().filter(|&&b| b != 0).count(),
        skipped_too_long,
        skipped_no_reply,
    })
}

// `run_sft_training`'s preference twin. The dataset must already have been built
// by `build_kto_dataset` — the caller runs that as phase 1.
pub fn run_kto_training(
    cfg: &TrainConfig,
    on_progress: llm::train::ProgressCallback,
    cancel: Arc<AtomicBool>,
) -> io::Result<bool> {
    let base_id = cfg.base_model_id.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "KTO run requires a base_model_id")
    })?;
    let base_model_dir = storage::models_dir().join(base_id.to_string());
    let datasets = storage::datasets_dir();
    let out_model_dir = storage::models_dir().join(cfg.storage_id.to_string());
    let checkpoints_dir = out_model_dir.join("checkpoints");

    llm::train::kto_train(
        &base_model_dir,
        &datasets.join("kto_corpus.bin"),
        &datasets.join("kto_mask.bin"),
        &datasets.join("kto_index.json"),
        &out_model_dir,
        cfg.epochs,
        cfg.lr,
        cfg.beta,
        Some(&checkpoints_dir),
        on_progress,
        Some(cancel),
    )
}

// ── Synthetic SFT data generator (ISSUE-0036) ───────────────────────────────
//
// Produces templated instruction/response conversations whose correct assistant
// reply is known *by construction* (arithmetic, sorting, factual lookup, echo),
// so the SFT smoke path has clean signal without a data-collection project. Opt-in
// and independent of the training loop — it just writes JSONL that
// `build_sft_dataset` later consumes. No LLM-in-the-loop, no external corpora, no
// template-engine dependency (`format!` is enough at this scale).
//
// Not yet wired to a runtime trigger (a dev/admin route or CLI is a future
// issue), so in this binary crate its items read as dead code until then — the
// same "defined ahead of its caller" state `build_sft_dataset` sat in before
// ISSUE-0037. One justified module-level allow rather than scattering it (R7).
#[allow(dead_code)]
mod synthetic {
    use super::{rng_seed, storage, xorshift_next};
    use serde::Serialize;
    use std::fs;
    use std::io;

// A conversation in the on-disk JSONL shape (`SftConversation`'s serialize twin;
// that one is deserialize-only). `system` is omitted from the line when absent.
#[derive(Debug, Serialize)]
struct GenConversation {
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    turns: Vec<GenTurn>,
}

#[derive(Debug, Serialize)]
struct GenTurn {
    role: &'static str,
    text: String,
}

// Small factual table — capital cities. Verifiable lookups with short answers
// that fit the tiny models' context.
const CAPITALS: &[(&str, &str)] = &[
    ("France", "Paris"),
    ("Japan", "Tokyo"),
    ("Egypt", "Cairo"),
    ("Peru", "Lima"),
    ("Norway", "Oslo"),
    ("Kenya", "Nairobi"),
];

// Roughly one in `SYSTEM_PROMPT_ODDS` examples gets a system prompt, so the model
// sees (and learns to ignore for loss) the system scaffold on some fraction.
const SYSTEM_PROMPT_ODDS: u64 = 3;

/// Generates `count` templated chat conversations and writes them as one JSONL
/// file (`synthetic.jsonl`) into `storage::sft_dir()`, the input dir that
/// `build_sft_dataset` reads. Each line is a `{system?, turns:[…]}` object ending
/// on an assistant turn, matching the SFT dataset schema. Returns `count`.
pub fn generate_sft_examples(count: usize) -> io::Result<usize> {
    let sft = storage::sft_dir();
    fs::create_dir_all(&sft)?;

    let mut state = rng_seed();
    let mut out = String::new();
    for _ in 0..count {
        let conv = make_example(&mut state);
        // Serializing a struct we built ourselves can only fail on an allocator
        // fault; propagate anyway rather than unwrap (R3).
        out.push_str(&serde_json::to_string(&conv)?);
        out.push('\n');
    }
    fs::write(sft.join("synthetic.jsonl"), out)?;
    Ok(count)
}

// Builds one conversation: pick a category, optionally prepend a system prompt.
fn make_example(state: &mut u64) -> GenConversation {
    let (question, answer) = match xorshift_next(state) % 4 {
        0 => arithmetic_example(state),
        1 => sort_example(state),
        2 => capital_example(state),
        _ => echo_example(state),
    };
    let system = if xorshift_next(state).is_multiple_of(SYSTEM_PROMPT_ODDS) {
        Some("You are a helpful assistant. Answer concisely.".to_string())
    } else {
        None
    };
    GenConversation {
        system,
        turns: vec![
            GenTurn { role: "user", text: question },
            GenTurn { role: "assistant", text: answer },
        ],
    }
}

// `a op b` for +/-/* with small operands; subtraction is ordered so the answer is
// never negative (keeps replies to a single short token run).
fn arithmetic_example(state: &mut u64) -> (String, String) {
    let a = (xorshift_next(state) % 20) as i64;
    let b = (xorshift_next(state) % 20) as i64;
    match xorshift_next(state) % 3 {
        0 => (format!("What is {} + {}?", a, b), format!("{}", a + b)),
        1 => {
            let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
            (format!("What is {} - {}?", hi, lo), format!("{}", hi - lo))
        }
        _ => (format!("What is {} * {}?", a, b), format!("{}", a * b)),
    }
}

// Sort 3–5 small numbers ascending. Both prompt and answer are comma-separated.
fn sort_example(state: &mut u64) -> (String, String) {
    let n = 3 + (xorshift_next(state) % 3) as usize; // 3..=5
    let nums: Vec<i64> = (0..n).map(|_| (xorshift_next(state) % 100) as i64).collect();
    let prompt = nums.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
    let mut sorted = nums;
    sorted.sort();
    let answer = sorted.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
    (format!("Sort these numbers in ascending order: {}", prompt), answer)
}

// "What is the capital of X?" from the CAPITALS table.
fn capital_example(state: &mut u64) -> (String, String) {
    let (country, capital) = CAPITALS[(xorshift_next(state) as usize) % CAPITALS.len()];
    (format!("What is the capital of {}?", country), capital.to_string())
}

// Echo a short word N times — a format-following task with a known answer.
fn echo_example(state: &mut u64) -> (String, String) {
    const WORDS: &[&str] = &["cat", "sun", "book", "tree", "star"];
    let word = WORDS[(xorshift_next(state) as usize) % WORDS.len()];
    let times = 2 + (xorshift_next(state) % 3) as usize; // 2..=4
    let answer = std::iter::repeat_n(word, times).collect::<Vec<_>>().join(" ");
    (format!("Repeat the word \"{}\" {} times.", word, times), answer)
}
} // mod synthetic

#[cfg(test)]
mod sft_tests {
    use super::*;
    use std::sync::Mutex;

    // `DATA_DIR` is a process-global that `storage::root()` reads, so tests that
    // set it must not run concurrently. cargo runs tests multi-threaded by
    // default; serialize the env-touching ones through this lock. (`.lock()`
    // recovers from a poisoned guard so one failing test doesn't cascade.)
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // Build a dataset from a fixture SFT dir + a fixture merges.bin (empty merges
    // → byte-level tokenization, deterministic), and check the corpus/mask are
    // aligned and the trained-token count matches the assistant spans.
    #[test]
    fn test_build_sft_dataset_aligns_and_counts() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Isolate storage + a base-model dir under a unique temp root.
        let tmp = std::env::temp_dir().join(format!("sft_build_{}", std::process::id()));
        let sft = tmp.join("raw").join("sft");
        let base = tmp.join("base");
        fs::create_dir_all(&sft).unwrap();
        fs::create_dir_all(&base).unwrap();
        // SAFETY: single-threaded test; DATA_DIR drives storage::root().
        unsafe { std::env::set_var("DATA_DIR", &tmp); }

        let merges: Vec<((u32, u32), u32)> = Vec::new();
        tokenize::bpe::save_merges(&merges, &base.join("merges.bin").to_string_lossy()).unwrap();

        // Two conversations, one per line.
        let jsonl = "{\"turns\":[{\"role\":\"user\",\"text\":\"q1\"},{\"role\":\"assistant\",\"text\":\"a1\"}]}\n\
                     {\"system\":\"be nice\",\"turns\":[{\"role\":\"user\",\"text\":\"q2\"},{\"role\":\"assistant\",\"text\":\"a2\"}]}\n";
        fs::write(sft.join("data.jsonl"), jsonl).unwrap();

        let stats = build_sft_dataset(&base).unwrap();
        assert_eq!(stats.conversations, 2);

        let corpus_bytes = fs::read(storage::datasets_dir().join("sft_corpus.bin")).unwrap();
        let mask_bytes = fs::read(storage::datasets_dir().join("sft_mask.bin")).unwrap();
        // corpus is u32 (4 bytes) per token; mask is 1 byte per token — same count.
        assert_eq!(corpus_bytes.len(), mask_bytes.len() * 4, "corpus/mask must align");
        assert_eq!(stats.total_tokens, mask_bytes.len());

        // Trainable tokens = the two assistant replies + their closing markers.
        let expected = llm::chat_template::build_training_example(
            None,
            &[
                llm::chat_template::Turn { role: llm::chat_template::Role::User, text: "q1".into() },
                llm::chat_template::Turn { role: llm::chat_template::Role::Assistant, text: "a1".into() },
            ],
            &merges,
        ).1.iter().filter(|&&b| b != 0).count()
        + llm::chat_template::build_training_example(
            Some("be nice"),
            &[
                llm::chat_template::Turn { role: llm::chat_template::Role::User, text: "q2".into() },
                llm::chat_template::Turn { role: llm::chat_template::Role::Assistant, text: "a2".into() },
            ],
            &merges,
        ).1.iter().filter(|&&b| b != 0).count();
        assert_eq!(stats.trainable_tokens, expected);
        assert!(stats.trainable_tokens < stats.total_tokens);

        fs::remove_dir_all(&tmp).ok();
    }

    // Back-compat: a request body written before `mode`/`base_model_id` existed
    // (no such keys) must still deserialize, defaulting to a pretrain run. This is
    // the regression ISSUE-0037 most needs to guard.
    #[test]
    fn test_trainconfig_defaults_to_pretrain_without_mode() {
        let body = r#"{
            "model_name": "legacy", "dimensions": 64, "ctx_window": 16, "heads": 4,
            "layers": 2, "epochs": 1, "lr": 0.01, "ff_multiplier": 2, "max_vocab": 512
        }"#;
        let cfg: TrainConfig = serde_json::from_str(body).unwrap();
        assert_eq!(cfg.mode, TrainMode::Pretrain);
        assert!(cfg.base_model_id.is_none());
    }

    // An explicit SFT body parses the lowercase tag and carries the base id.
    #[test]
    fn test_trainconfig_parses_sft_mode() {
        let id = Uuid::new_v4();
        let body = format!(
            r#"{{"model_name":"ft","dimensions":64,"ctx_window":16,"heads":4,"layers":2,
                "epochs":1,"lr":0.0001,"ff_multiplier":2,"max_vocab":512,
                "mode":"sft","base_model_id":"{id}"}}"#
        );
        let cfg: TrainConfig = serde_json::from_str(&body).unwrap();
        assert_eq!(cfg.mode, TrainMode::Sft);
        assert_eq!(cfg.base_model_id, Some(id));
    }

    #[test]
    fn test_build_sft_dataset_rejects_unknown_role() {
        assert!(parse_role("system").is_err());
        assert!(parse_role("user").is_ok());
        assert!(parse_role("assistant").is_ok());
    }

    // Every generated example is a well-formed JSONL line that round-trips through
    // `SftConversation` and renders (via ISSUE-0031's `build_training_example`) to
    // a non-empty masked span — i.e. it carries a trainable assistant reply.
    #[test]
    fn test_generate_sft_examples_are_well_formed() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("sft_gen_{}", std::process::id()));
        fs::create_dir_all(&tmp).unwrap();
        // SAFETY: env access serialized by ENV_LOCK; DATA_DIR drives storage::root().
        unsafe { std::env::set_var("DATA_DIR", &tmp); }

        let n = 50;
        assert_eq!(synthetic::generate_sft_examples(n).unwrap(), n);

        // Empty merges → deterministic byte-level tokenization, enough to assert the
        // mask is non-empty per example.
        let merges: Vec<((u32, u32), u32)> = Vec::new();

        let content = fs::read_to_string(storage::sft_dir().join("synthetic.jsonl")).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), n, "one JSONL line per example");

        for line in lines {
            let conv: SftConversation = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("line not valid SftConversation: {} ({})", line, e));
            // Must end on an assistant turn.
            assert_eq!(conv.turns.last().unwrap().role, "assistant");

            let turns: Vec<llm::chat_template::Turn> = conv
                .turns
                .iter()
                .map(|t| llm::chat_template::Turn {
                    role: parse_role(&t.role).unwrap(),
                    text: t.text.clone(),
                })
                .collect();
            let (ids, mask) =
                llm::chat_template::build_training_example(conv.system.as_deref(), &turns, &merges);
            assert!(!ids.is_empty(), "example rendered to no tokens: {}", line);
            assert!(
                mask.iter().any(|&b| b != 0),
                "example has no trainable (assistant) span: {}",
                line
            );
        }

        fs::remove_dir_all(&tmp).ok();
    }
}
