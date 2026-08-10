// Orchestrates the upload -> tokenize -> train pipeline. The `llm` and
// `tokenize` crates are deliberately layout-agnostic (they take explicit paths);
// this module is where `storage.rs` (the single source of truth for the data
// layout) is mapped onto those crate calls. Plain free functions, no traits.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;


// Which kind of run this is: pretrain a fresh model, or supervised-fine-tune
// (SFT) an existing one on chat data. Serialized lowercase (`"pretrain"` /
// `"sft"`) in the POST body and meta.json.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TrainMode {
    #[default]
    Pretrain,
    Sft,
}

/// Whether a run of `mode` renders its chat data with reasoning, given what the
/// request asked for.
///
/// The one place the "pretrain always strips" rule lives, so the dataset builder
/// and anything reconstructing a model's rendering cannot disagree about it. A
/// pretrain corpus carries chat to expose the *scaffold* — the markers a later SFT
/// would otherwise meet at random init — and a thinking block is not scaffold.
pub fn strip_thinking_for(mode: TrainMode, requested: bool) -> bool {
    match mode {
        TrainMode::Pretrain => true,
        _ if requested => true,
        _ => false,
    }
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
    // Drop the `<|thinking|>` blocks (and their budget markers) from the chat data
    // this run trains on, teaching the model to answer directly.
    //
    // **SFT only as a choice.** Pretrain always strips, whatever this says: the
    // chat share exists in a pretrain corpus to expose the *scaffold*, and
    // reasoning a model that small cannot hold together is capacity spent on noise
    // either way. `strip_thinking_for(mode)` is the one place that rule lives.
    //
    // `#[serde(default)]` => a body or meta.json written before this field
    // deserializes as `false`, exactly reproducing the runs already on disk.
    #[serde(default)]
    pub strip_thinking: bool,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub files: usize,
    pub chars: usize,
    pub vocab_size: i64,
    pub merges: usize,
    pub corpus_len: usize,
    // Number of documents (source files) whose token spans were recorded in
    // segments.json for per-epoch shuffling / curriculum staging.
    pub documents: usize,
    // Which reserved special-token ids this corpus actually contained, sorted.
    //
    // Recorded because it is the only durable answer to "which marker embeddings
    // did this model never train?", which a later SFT needs in order to warm-start
    // them (`llm::train::warm_start_embeddings`). It cannot be recovered afterwards:
    // not from the weights (a small trained embedding is indistinguishable from an
    // untrained one — the bug this replaced), and not from `corpus.bin`, which a
    // later pretrain overwrites. `#[serde(default)]` leaves a model trained before
    // this field with an empty list, which the caller reads as "unknown" rather
    // than as "saw nothing".
    #[serde(default)]
    pub special_ids_seen: Vec<u32>,
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

// The SFT counterpart of `Segments`, written next to `sft_corpus.bin`: one
// span per *conversation*, exclusive of the `<|endoftext|>` baked in between them.
//
// Separate from `Segments` rather than a reuse of it because the two answer
// different questions. A pretrain corpus is documents belonging to curriculum
// stages, and the stage is what its sidecar exists to record. An SFT corpus has no
// stages and no source datasets — it is a flat list of conversations, and the only
// thing worth recording is where each one starts, so the training loop can deal
// them in a different order every epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftSegments {
    pub eot_id: u32,
    pub conversations: Vec<SftSpan>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SftSpan {
    pub start: usize,
    pub len: usize,
}


// How much text trains the BPE merge table.
//
// `bpe::train` now collapses the sample to its **distinct** pretokens before
// learning anything, so its footprint scales with the sample's vocabulary rather
// than its length — and natural text repeats itself heavily, so that is a far
// slower curve. Measured end to end (train a full 8192 table on a 10.6 MiB sample,
// then encode 8 MiB of held-out text) peak RSS was 267 MB, against the ~1 GB that
// 16 MiB used to cost when the trainer held the whole corpus as chars *and* as ids
// alongside a never-compacted positional index.
//
// This constant is therefore no longer the memory cliff it was, and its *size* is
// not the lever either — see `plan_bpe_sample`, whose composition is. Measured at
// vocab 8192 and 65536, held-out bytes/token over the real corpora:
//
//   sample    vocab 8k    vocab 64k
//   16 MiB      3.967       4.872
//   64 MiB      3.945       4.889
//
// Identical inside noise at 4x the tokenizer-training time. Raising it is safe on
// the memory side and buys nothing; leave it here.
const BPE_SAMPLE_BYTES: usize = 16 * 1024 * 1024;

// The least of the BPE sample any one dataset may be cut to, as a percent of
// `BPE_SAMPLE_BYTES`. `plan_bpe_sample` splits the budget by corpus share, which
// left to itself would give a 1.5 GB web-text dataset ~98% of it and leave a small
// early-curriculum dataset tokenizing at nearly byte level. The floor bounds that
// without giving up the gains proportional sampling actually delivers; it is
// capped at an equal split, so it stays meaningful however many datasets a run
// selects.
const MIN_DATASET_SAMPLE_SHARE: usize = 5;

// Bytes per streaming read. Chunks end on a line break, so this is a floor rather
// than an exact size.
const CHUNK_TARGET_BYTES: usize = 4 * 1024 * 1024;

// A chunk that grows past this without meeting a line break means a file with no
// line structure, and so no safe point to cut. Refuse it rather than read it whole
// and exhaust memory.
const CHUNK_MAX_BYTES: usize = 64 * 1024 * 1024;

// Roughly how many `Encode` events a whole build emits, regardless of file count.
const PROGRESS_REPORTS_PER_BUILD: usize = 200;

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

// There is deliberately no wall-clock-seeded `shuffle` wrapper. Its only caller
// was `select_txt`, where it meant a dataset whose `count` was smaller than what
// is on disk drew a *different random subset on every run* — so two runs of one
// config trained on different corpora, and any A/B holding the corpus constant
// was measuring two things at once. Every shuffle takes an explicit seed; the
// run's own resolved seed is threaded down from `build_dataset`.

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
fn select_txt(
    dir: &Path,
    num: usize,
    zero_means_all: bool,
    seed: u64,
) -> io::Result<Vec<PathBuf>> {
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
        // Seeded, not wall-clock: two runs of one config must select the same
        // files. See `build_dataset`'s `seed` parameter.
        shuffle_seeded(&mut paths, seed);
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




// One selected file, with the size the sample planner needs. Sizes come from
// `metadata` so planning costs no reads.
struct TaggedFile {
    stage: usize,
    dataset: String,
    path: PathBuf,
    bytes: u64,
}


// How much of one file to feed the tokenizer trainer.
struct SampleSlice {
    file: usize,
    bytes: usize,
}



// What the streaming pass produced.
struct Streamed {
    documents: Vec<DocSpan>,
    corpus_len: usize,
    chars: usize,
    special_ids: Vec<u32>,
}



fn cancelled(cancel: &Option<Arc<AtomicBool>>) -> bool {
    cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst))
}

/// Reads whole lines into `buf` (cleared first) until it holds at least `target`
/// bytes or the file ends. `Ok(false)` means end of file with nothing read.
///
/// Cutting only after a `'\n'` is what makes chunked encoding exact rather than
/// approximate: `replace_special_characters` maps a newline to `'Ċ'` and
/// `split_punctuation` splits on it, so a newline is already a pretoken boundary
/// and no token can span one. It also means a cut can never land inside a
/// multi-byte character or a `<|marker|>`, neither of which contains a newline.
///
/// `max` is a parameter rather than read from the constant so the no-newline
/// rejection can be tested without a 64 MiB fixture.
fn read_chunk(
    reader: &mut BufReader<File>,
    target: usize,
    max: usize,
    buf: &mut String,
    path: &Path,
) -> io::Result<bool> {
    buf.clear();
    while buf.len() < target {
        let read = reader.read_line(buf)?;
        if read == 0 {
            break;
        }
        if buf.len() > max {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{}: no line break in the first {} MiB — there is no safe point to \
                     split this file, so shard it or add line breaks before uploading",
                    path.display(),
                    max >> 20
                ),
            ));
        }
    }
    Ok(!buf.is_empty())
}

// The reserved special-token ids present in a token stream, sorted and deduped.
//
// Membership is asked of the tokenizer (`special_char_from_id`) rather than
// tested as a range, so the unallocated tail of the reserved block counts as
// absent — those slots hold no marker and nothing can have emitted them.
fn special_ids_in(tokens: &[u32]) -> Vec<u32> {
    let mut seen: Vec<u32> = tokens
        .iter()
        .copied()
        .filter(|&id| tokenize::pretokenize::special_char_from_id(id).is_some())
        .collect();
    seen.sort_unstable();
    seen.dedup();
    seen
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

// Writes the conversation-boundary sidecar next to sft_corpus.bin. Best-effort at
// the call sites: a dataset without it still trains, just in a fixed order.
fn save_sft_segments(segments: &SftSegments, path: &Path) -> io::Result<()> {
    let json = serde_json::to_vec(segments)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(path, json)
}

// `Ok(None)` when the sidecar is absent — a dataset built before this existed —
// so the SFT run falls back to the whole corpus as one fixed-order document.
fn load_sft_segments(path: &Path) -> io::Result<Option<SftSegments>> {
    match fs::read(path) {
        Ok(bytes) => {
            let segments: SftSegments = serde_json::from_slice(&bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(Some(segments))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

// The name of the sidecar, as both builders write it and `run_sft_training` reads
// it. Named once so the two halves cannot drift.
const SFT_SEGMENTS_FILE: &str = "sft_segments.json";

// Parses the tokenize crate's "BPE merge {done}/{total}" line into numbers.
// Returns None for any other log line (kept tolerant of format drift).
fn parse_merge_progress(line: &str) -> Option<(u32, u32)> {
    let rest = line.strip_prefix("BPE merge ")?;
    let (done, total) = rest.split_once('/')?;
    Some((done.trim().parse().ok()?, total.trim().parse().ok()?))
}



// The chat scaffold, as marker text. Used only as the fallback reset set for a
// base model whose meta.json predates `DatasetStats::special_ids_seen`: before
// that field existed the pretrain corpora were prose + document markup, so the
// scaffold is the part we can still assert was absent. Anything richer would be a
// guess about a corpus no longer on disk.
const LEGACY_SCAFFOLD_MARKERS: [&str; 6] = [
    "<|user|>",
    "<|/user|>",
    "<|assistant|>",
    "<|/assistant|>",
    "<|system|>",
    "<|/system|>",
];

// Which marker embeddings a fine-tune will feed that its base model never trained:
// the special ids used by the fine-tune's corpus, minus those the base has seen.
//
// The base's side comes from its `meta.json` — see `DatasetStats::special_ids_seen`
// for why it cannot be recovered any other way. A base with no such record (every
// model trained before the field existed) falls back to the chat scaffold, which is
// exactly the set the previous hardcoded implementation reset, so an old model
// behaves as it did before.
//
// A missing or unparseable base meta.json yields the fallback set, not an error:
// warm-starting six rows that may already be trained is a far smaller mistake than
// refusing to fine-tune at all. A missing *corpus* is a real failure and propagates.
fn untrained_marker_ids(base_model_dir: &Path, corpus_path: &Path) -> io::Result<Vec<u32>> {
    let used = special_ids_in(&read_corpus(corpus_path)?);

    match model_special_ids_seen(base_model_dir) {
        Some(seen) => Ok(used.into_iter().filter(|id| !seen.contains(id)).collect()),
        None => {
            eprintln!(
                "warm-start: base model records no marker set; \
                 falling back to the chat scaffold"
            );
            Ok(scaffold_ids())
        }
    }
}

/// How a trained model's chat data was rendered, from its own `meta.json`.
///
/// A benchmark has to score a model the way it was trained: the assistant suite is
/// bits per assistant-reply token, so measuring a stripped model against replies
/// that still carry their `<|thinking|>` blocks charges it for tokens it was taught
/// never to emit. That is a property of the model, not a choice about the run, so
/// it is read here rather than asked for on the benchmark form.
///
/// Best-effort by design: a missing, unreadable or pre-`strip_thinking` meta.json
/// yields `Keep`, which is what every model already on disk was trained under.
pub fn model_reasoning(model_dir: &Path) -> bool {
    let stripped = fs::read_to_string(model_dir.join("meta.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|meta| meta.get("config")?.get("strip_thinking")?.as_bool())
        .unwrap_or(false);
    if stripped {
        true
    } else {
        false
    }
}

// A whole corpus file as ids, widened to the u32 the rest of this module speaks.
//
// Only for the SFT corpus, which is bounded by the conversation set and so
// are small. The pretrain corpus is hundreds of millions of tokens and must never
// come through here — `append_prose_replay` reads the spans it needs instead.
fn read_corpus(path: &Path) -> io::Result<Vec<u32>> {
    Ok(tokenize::bpe::load_corpus(&path.to_string_lossy())?
        .into_iter()
        .map(u32::from)
        .collect())
}

// The markers a model has trained on, from its own `meta.json`, or `None` when it
// carries no usable record.
//
// Read as a loose JSON field rather than through `DatasetStats`, because the
// `dataset` object is a different shape per mode (`DatasetStats` for pretrain,
// `SftDatasetStats` for a fine-tune) and only the one field is wanted from either.
// Each mode records the markers its model has now seen — a fine-tune unions its own
// corpus with its base's — so the record composes down a pretrain → SFT
// chain instead of only describing the last hop.
//
// An empty list is treated as absent rather than as "saw no markers": that is what
// a meta.json written before the field deserializes to, and the two are
// indistinguishable from here.
pub fn model_special_ids_seen(model_dir: &Path) -> Option<Vec<u32>> {
    let raw = fs::read_to_string(model_dir.join("meta.json")).ok()?;
    let meta: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let seen: Vec<u32> = meta
        .get("dataset")?
        .get("special_ids_seen")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as u32))
        .collect();
    if seen.is_empty() {
        return None;
    }
    Some(seen)
}

fn scaffold_ids() -> Vec<u32> {
    LEGACY_SCAFFOLD_MARKERS
        .iter()
        // A lone special marker encodes to exactly its one reserved id, so the
        // merge list is irrelevant — deriving the ids through `encode` keeps this
        // correct if `SPECIAL_TOKENS` is ever reordered.
        .filter_map(|m| tokenize::bpe::encode(m, &[]).first().copied())
        .collect()
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
    let model_dir = engine::paths::data_root()
        .join("artifacts/models")
        .join(cfg.storage_id.to_string());
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
// `Serialize` too, so the held-out share can be written back out in the same shape
// `llm::eval::chat` reads.
#[derive(Debug, Serialize, Deserialize)]
pub struct SftTurn {
    role: String,
    text: String,
}

// One conversation: an optional system prompt plus alternating turns, ending on
// an assistant reply. One JSON object per line in the input `.jsonl` files.
#[derive(Debug, Serialize, Deserialize)]
pub struct SftConversation {
    // Skipped when absent so a re-exported conversation is the same JSON it came
    // in as, rather than gaining a `"system": null`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    turns: Vec<SftTurn>,
    // How long the reply was written to reason for, when the generator recorded it
    // — `synth::teach` writes `"budget": "low"|"med"|"high"` on every conversation
    // in `think-*.jsonl`. Rendered as one marker in front of the assistant turn, so
    // the fine-tune learns reasoning length as a *lever it is told to pull* rather
    // than as unexplained variance between two otherwise identical replies.
    //
    // Absent everywhere else, and absent renders as nothing, so the conversations
    // that predate this are tokenized exactly as they were.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    budget: Option<engine::ThinkBudget>,
}

// Summary of what the SFT build produced, for the dashboard + meta.json.
//
// The trainable fraction is a useful sanity check, but read it against the data
// rather than a fixed bar. It sat well under 50% while the set was conversations
// written here, where most tokens were user context the model only conditions on;
// on smol-smoltalk it is ~86%, because the replies are long against their prompts
// and half of what remains is prose replay, which is trainable by design.
#[derive(Debug, Clone, Serialize)]
pub struct SftDatasetStats {
    pub conversations: usize,
    pub total_tokens: usize,
    pub trainable_tokens: usize,
    // Prose tokens mixed in from the base model's pretraining corpus, all of them
    // trainable. Reported because 0 is a meaningful and easily-missed outcome: it
    // means the tokenizer guard refused the replay, and the run is the
    // unanchored fine-tune that replay exists to prevent.
    pub replay_tokens: usize,
    // Conversations refused for rendering past the base model's context window.
    //
    // SFT was the only builder without this guard — `build_kto_dataset` and
    // `build_raft_dataset` have always had it — and it did not show while the SFT
    // set was a couple of thousand conversations written against this model. An
    // over-long conversation is not merely wasteful: it becomes a span in
    // `sft_segments.json` longer than the window `run_sft_training` hands
    // `sft_train`, so the run trains on a silently truncated example.
    //
    // Expect a small number. A large one means the import that produced `raw/sft/`
    // is mis-tuned for this model's window, which is a data problem rather than
    // something to absorb here — see `scripts/prepare_smoltalk.py`.
    pub skipped_too_long: usize,
    // Every marker the fine-tuned model will have trained on: this corpus's,
    // unioned with the base's own record. The union is what makes the record
    // compose down a pretrain → SFT chain — without it, a fine-tune would read
    // its SFT base as having seen only the SFT corpus and warm-start markers the
    // pretrain had already taught it. See `DatasetStats::special_ids_seen`.
    pub special_ids_seen: Vec<u32>,
}

// Longest user turn kept, in bytes; beyond this the head is dropped and the tail
// kept.
//
// Conversations captured from a real session carry pasted logs, file listings and
// terminal transcripts. In the curated set the median user turn is 126 bytes and
// the largest is 109,118 — and those giants are almost pure context. They fill
// whole context windows containing no assistant token at all, so the chunk's loss
// is exactly zero and the step teaches nothing (`llm::train::trainable_chunks_in`).
// Before this cap, 85% of the SFT chunks were such windows, one run of 184 of them
// consecutively.
//
// 4000 was chosen against the data rather than picked round: it truncates 5 of 47
// turns while lifting the assistant's share of the dataset from 12% to ~45% —
// under the 50% the trainable fraction is documented to sit below. A tighter cap
// buys little and cuts three times as many turns.
//
// **The tail is kept, not the head.** A long turn is pasted context followed by
// the actual request, and the request is what the reply answers.
const MAX_USER_TURN_BYTES: usize = 4000;

fn cap_user_turn(text: &str) -> String {
    if text.len() <= MAX_USER_TURN_BYTES {
        return text.to_string();
    }
    let mut start = text.len() - MAX_USER_TURN_BYTES;
    // `text.len()` is bytes, so the cut can land inside a multi-byte character.
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    let tail = &text[start..];
    // Prefer starting at a line break when one is close, so the kept text doesn't
    // begin mid-sentence. Far-away breaks are ignored — a log with 3000-byte lines
    // would otherwise lose most of the budget.
    match tail.find('\n') {
        Some(i) if i < 200 => tail[i + 1..].to_string(),
        _ => tail.to_string(),
    }
}


// ── The chat corpus split ───────────────────────────────────────────────────
//
// One set of conversations has to serve three jobs that must not overlap:
//
//   * **Pretrain** — rendered into the corpus so the model meets the chat scaffold
//     during pretraining. Without this the marker embeddings arrive at SFT at
//     random init and wreck the trunk the moment they are fed.
//   * **SFT** — what the fine-tune actually trains on.
//   * **Eval** — held back entirely, scored by `llm::eval::chat`.
//
// The overlap is the part worth being careful about. Seeding the *same*
// conversations into pretraining that SFT then trains on turns the fine-tune into
// a memorization pass over text the model already fit: the forgetting cost is paid
// and nothing new is learned. And a held-out set that leaked into either is not a
// held-out set.
//
// So membership is decided by one function, keyed on a hash of the conversation's
// own content rather than on its position in a file. That makes the split
// idempotent — re-running the export moves nothing — and stable as the corpus
// grows: adding conversations does not reshuffle the existing assignments and
// quietly promote an eval conversation into training.

// Destination shares, as percentages of the whole set. Pretrain takes the largest
// share because teaching the markers is what unblocks SFT at all; eval takes the
// smallest that still scores.
//
// **Changing these re-partitions the set**, which moves conversations across the
// train/eval boundary and makes chat metrics incomparable with any run recorded
// before the change. Treat it as invalidating the benchmark history, not as tuning.
const CHAT_SHARE_PRETRAIN: u64 = 50;
const CHAT_SHARE_SFT: u64 = 35;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatDestination {
    Pretrain,
    Sft,
    Eval,
}

/// Most conversations `export_chat_eval_set` will write, however large the eval
/// share grows.
///
/// 1,000 rather than the 305 the suite ran on before, because the error bar is what
/// the cycle's gate is built on. Measured: `bits_per_byte_stderr` fell from ~0.015
/// to ~0.012 and the score's from ~0.83 points to ~0.66. Less than the 1/sqrt(n)
/// that count suggests, because the sampling unit is the conversation and the
/// clustering is what it is — which is also why pushing to ten thousand would buy
/// little against a linear cost in benchmark time.
///
/// **Changing it re-partitions the holdout and invalidates comparability**, exactly
/// as the share constants do. Bump `ASSISTANT_SCORE_VERSION` with it.
const MAX_HELDOUT_CONVERSATIONS: usize = 1_000;


// FNV-1a over the conversation's content.
//
// Hand-rolled rather than `DefaultHasher` because the split has to be reproducible
// across builds, and `DefaultHasher`'s output is explicitly not stable between Rust
// releases — a toolchain upgrade would silently re-partition the corpus and leak
// the eval set into training.
fn conversation_hash(conv: &SftConversation) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    let mut absorb = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    if let Some(system) = &conv.system {
        absorb(system.as_bytes());
    }
    for turn in &conv.turns {
        absorb(turn.role.as_bytes());
        absorb(turn.text.as_bytes());
    }
    // `budget` is deliberately **not** absorbed. It is a label describing a
    // conversation, not part of it, and every conversation already on disk was
    // assigned a destination without it — folding it in would re-partition the
    // whole set on the next run, moving conversations across the train/eval
    // boundary and invalidating comparability with every score already recorded.
    hash
}

// Which of the three jobs a conversation belongs to. The single source of truth:
// the exporter and `build_sft_dataset` both call it, so they cannot disagree about
// where a conversation went.
pub fn chat_destination(conv: &SftConversation) -> ChatDestination {
    match conversation_hash(conv) % 100 {
        n if n < CHAT_SHARE_PRETRAIN => ChatDestination::Pretrain,
        n if n < CHAT_SHARE_PRETRAIN + CHAT_SHARE_SFT => ChatDestination::Sft,
        _ => ChatDestination::Eval,
    }
}

/// The name the chat conversations appear under in the curriculum picker.
///
/// Not a folder in `uploads/`. Every other dataset is text someone uploaded, but
/// this one is *derived* from `raw/sft/*.jsonl`, and derived data in the source
/// directory is how a corpus goes stale — you add conversations, forget to
/// re-export, and the run silently trains on last week's set. So the name is
/// virtual: `list_datasets` reports it, and `build_dataset` renders it fresh.
pub const CHAT_DATASET: &str = "Chat";




fn reset_dir(dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    for entry in fs::read_dir(dir)?.flatten() {
        if entry.path().is_file() {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}






// ── Prose replay ────────────────────────────────────────────────────────────
//
// How many prose tokens to mix in per trainable chat token. At 1.0 half the
// gradient signal in an SFT run comes from the pretraining distribution.
//
// Without replay, a fine-tune on a few hundred conversations is a full-parameter
// pass over a corpus a fraction of a percent the size of the pretrain set, with
// nothing anchoring the model to prose — the model trades its language ability for
// the fine-tune's register, which is catastrophic forgetting doing exactly what it
// says. Interleaving the base's own pretraining text is the standard remedy.
const REPLAY_TOKENS_PER_CHAT_TOKEN: f64 = 1.0;

// Longest span taken from any one pretrain document. A Wikipedia article can be
// tens of thousands of tokens, and taking whole documents would spend the entire
// replay budget on two of them — the point is breadth of the old distribution, not
// depth in one article.
const MAX_REPLAY_DOC_TOKENS: usize = 2048;



// Fisher–Yates over an explicit seed, so a rebuild picks the same documents.
// Mirrors `llm::train::shuffle_seeded`; `| 1` keeps xorshift out of its zero state.
fn shuffle_indices(order: &mut [usize], seed: u64) {
    let mut state = seed | 1;
    for i in (1..order.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(i, (state % (i as u64 + 1)) as usize);
    }
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
    use super::{rng_seed, xorshift_next};
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

    #[test]
    fn a_short_user_turn_is_left_alone() {
        let text = "why is the sky blue?";
        assert_eq!(cap_user_turn(text), text);
    }

    #[test]
    fn a_long_user_turn_keeps_its_tail() {
        // The request comes after the pasted context, so the tail is the half that
        // the assistant's reply actually answers.
        let text = format!("{}\nso what is wrong with it?", "x".repeat(MAX_USER_TURN_BYTES * 2));
        let capped = cap_user_turn(&text);
        assert!(capped.len() <= MAX_USER_TURN_BYTES);
        assert!(capped.ends_with("so what is wrong with it?"), "got {capped:?}");
    }

    #[test]
    fn capping_never_splits_a_multibyte_character() {
        // A paste of emoji: every cut offset lands mid-character unless guarded.
        for extra in 0..8 {
            let text = "🙂".repeat(MAX_USER_TURN_BYTES / 2 + extra);
            let capped = cap_user_turn(&text);
            assert!(capped.len() <= MAX_USER_TURN_BYTES);
            assert!(capped.chars().all(|c| c == '🙂'), "cut produced a broken char");
        }
    }

    #[test]
    fn a_far_away_line_break_does_not_eat_the_budget() {
        // One enormous line: there is no nearby break to align to, so the tail is
        // kept as-is rather than being trimmed to the next break thousands of
        // bytes later.
        let text = "y".repeat(MAX_USER_TURN_BYTES * 3);
        let capped = cap_user_turn(&text);
        assert_eq!(capped.len(), MAX_USER_TURN_BYTES);
    }

    // ── streaming dataset build ──

    fn tagged(dataset: &str, bytes: u64) -> TaggedFile {
        TaggedFile {
            stage: 0,
            dataset: dataset.to_string(),
            path: PathBuf::from(format!("/tmp/{dataset}-{bytes}.txt")),
            bytes,
        }
    }

    // Two runs of one config must select the same files. `select_txt` shuffled
    // with a wall-clock seed, so any dataset whose `count` was smaller than what
    // is on disk drew a *different random subset every run* — which quietly makes
    // any A/B a comparison of two corpora as well as of whatever was under test.
    #[test]
    fn a_seeded_selection_picks_the_same_files_every_time() {
        let dir = std::env::temp_dir().join(format!("seltxt_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for i in 0..40 {
            fs::write(dir.join(format!("shard{i:03}.txt")), "x\n").unwrap();
        }

        let pick = |seed: u64| -> Vec<PathBuf> { select_txt(&dir, 8, true, seed).unwrap() };

        assert_eq!(pick(1337), pick(1337), "the same seed must select the same files");
        assert_ne!(pick(1337), pick(99), "a different seed should select differently");
        assert_eq!(pick(1337).len(), 8);

        // `count: 0` means everything, so it is seed-independent by construction.
        assert_eq!(select_txt(&dir, 0, true, 1).unwrap(), select_txt(&dir, 0, true, 2).unwrap());

        let _ = fs::remove_dir_all(&dir);
    }

    // A file with no line break has no safe cut point, so it must be refused
    // rather than read whole — that read is exactly the OOM this design avoids.
    #[test]
    fn a_file_without_line_breaks_is_refused_by_name() {
        let dir = std::env::temp_dir().join(format!("chunk_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("oneline.txt");
        fs::write(&path, "x".repeat(4096)).unwrap();

        let mut reader = BufReader::new(File::open(&path).unwrap());
        let mut buf = String::new();
        let err = read_chunk(&mut reader, 64, 128, &mut buf, &path).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("oneline.txt"), "message must name the file: {err}");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn chunks_end_on_line_breaks_and_cover_the_file() {
        let dir = std::env::temp_dir().join(format!("chunk_ok_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lines.txt");
        let text: String = (0..500).map(|i| format!("line number {i}\n")).collect();
        fs::write(&path, &text).unwrap();

        let mut reader = BufReader::new(File::open(&path).unwrap());
        let mut buf = String::new();
        let mut rebuilt = String::new();
        let mut chunks = 0;
        while read_chunk(&mut reader, 256, 1 << 20, &mut buf, &path).unwrap() {
            assert!(buf.ends_with('\n'), "chunk must end on a line break");
            rebuilt.push_str(&buf);
            chunks += 1;
        }

        assert!(chunks > 1, "fixture should need several chunks, got {chunks}");
        assert_eq!(rebuilt, text, "chunks must reassemble into the original file");

        fs::remove_dir_all(&dir).ok();
    }

    // Sets up a temp DATA_DIR with `files` under one uploads dataset and returns
    // the stages that select it. Caller must hold ENV_LOCK.
    fn stage_over(tmp: &Path, dataset: &str, files: &[(&str, &str)]) -> Vec<CurriculumStage> {
        let dir = tmp.join("raw").join("uploads").join(dataset);
        fs::create_dir_all(&dir).unwrap();
        for (name, text) in files {
            fs::write(dir.join(name), text).unwrap();
        }
        // SAFETY: single-threaded under ENV_LOCK; DATA_DIR drives storage::root().
        unsafe { std::env::set_var("DATA_DIR", tmp) };
        vec![CurriculumStage {
            name: None,
            datasets: vec![DatasetSelection { name: dataset.to_string(), count: 0 }],
            epochs: 1,
        }]
    }

    // Mirrors the tokenizer's private header size; asserted against here so a
    // change to the format shows up as a failing measurement rather than silently.
    const CORPUS_HEADER_BYTES_FOR_TEST: usize = 8;

    // ── the chat split ──

    fn conversation(user: &str, assistant: &str) -> SftConversation {
        SftConversation {
            system: None,
            turns: vec![
                SftTurn { role: "user".into(), text: user.into() },
                SftTurn { role: "assistant".into(), text: assistant.into() },
            ],
            budget: None,
        }
    }

    // Labelling a conversation with a thinking budget must not move it across the
    // train/eval boundary. The tiered files landed on disk *before* the label was
    // read back, so hashing it would re-partition a set that has already been
    // trained and scored against.
    #[test]
    fn a_budget_label_does_not_repartition_the_split() {
        let plain = conversation("why is the sky blue?", "rayleigh scattering");
        let before = chat_destination(&plain);
        for budget in engine::ThinkBudget::ALL {
            let mut labelled = conversation("why is the sky blue?", "rayleigh scattering");
            labelled.budget = Some(budget);
            assert_eq!(chat_destination(&labelled), before, "{budget:?} moved the split");
        }
    }

    // The label `synth::teach` writes must survive the round trip through
    // `SftConversation` — it was being silently dropped, which is the whole reason
    // the tiers taught nothing.
    #[test]
    fn the_on_disk_budget_label_is_read_back() {
        let line = r#"{"turns":[{"role":"user","text":"q"},{"role":"assistant","text":"a"}],"group":"g","budget":"med","think_tokens":269}"#;
        let conv: SftConversation = serde_json::from_str(line).unwrap();
        assert_eq!(conv.budget, Some(engine::ThinkBudget::Med));

        // …and a conversation without one still round-trips as the JSON it came in
        // as, rather than gaining a `"budget": null`.
        let plain: SftConversation =
            serde_json::from_str(r#"{"turns":[{"role":"user","text":"q"},{"role":"assistant","text":"a"}]}"#)
                .unwrap();
        assert_eq!(plain.budget, None);
        let out = serde_json::to_string(&plain).unwrap();
        assert!(!out.contains("budget"), "{out}");
    }

    // The property the whole split rests on: a conversation's destination follows
    // its content, so re-running the export cannot move it and leak an eval
    // conversation into training.
    #[test]
    fn a_conversations_destination_is_stable_across_calls() {
        let conv = conversation("why is the sky blue?", "rayleigh scattering");
        let first = chat_destination(&conv);
        for _ in 0..10 {
            assert_eq!(chat_destination(&conv), first);
        }
        // And it follows the content, not the identity of the value.
        assert_eq!(
            chat_destination(&conversation("why is the sky blue?", "rayleigh scattering")),
            first,
        );
    }

    #[test]
    fn different_conversations_do_not_all_land_together() {
        // A hash that ignored its input would put everything in one share, which
        // would silently leave SFT or eval with nothing.
        let mut seen = std::collections::HashSet::new();
        for i in 0..200 {
            seen.insert(chat_destination(&conversation(&format!("q{i}"), &format!("a{i}"))));
        }
        assert_eq!(seen.len(), 3, "expected all three destinations to be used");
    }

    #[test]
    fn the_shares_are_roughly_the_configured_proportions() {
        let mut pretrain = 0;
        let mut sft = 0;
        let mut eval = 0;
        const N: usize = 4000;
        for i in 0..N {
            match chat_destination(&conversation(&format!("question {i}"), &format!("reply {i}"))) {
                ChatDestination::Pretrain => pretrain += 1,
                ChatDestination::Sft => sft += 1,
                ChatDestination::Eval => eval += 1,
            }
        }
        // Wide bounds: this asserts the buckets are wired to the right constants,
        // not that FNV-1a is a good hash.
        let share = |n: usize| 100.0 * n as f64 / N as f64;
        assert!((share(pretrain) - CHAT_SHARE_PRETRAIN as f64).abs() < 5.0, "{}", share(pretrain));
        assert!((share(sft) - CHAT_SHARE_SFT as f64).abs() < 5.0, "{}", share(sft));
        let expected_eval = 100 - CHAT_SHARE_PRETRAIN - CHAT_SHARE_SFT;
        assert!((share(eval) - expected_eval as f64).abs() < 5.0, "{}", share(eval));
    }

    // The eval share is a fraction of the source set, so an import of a few hundred
    // thousand conversations makes the assistant suite a multi-hour job. The cap is
    // what stops that, and it has to select the *same* conversations every time or
    // the benchmark measures a different holdout on every run.
    #[test]
    fn the_held_out_cap_is_bounded_and_reproducible() {
        let all: Vec<SftConversation> = (0..MAX_HELDOUT_CONVERSATIONS * 20)
            .map(|i| conversation(&format!("question {i}"), &format!("reply {i}")))
            .collect();

        let select = || {
            let mut held: Vec<&SftConversation> = all
                .iter()
                .filter(|c| chat_destination(c) == ChatDestination::Eval)
                .collect();
            assert!(
                held.len() > MAX_HELDOUT_CONVERSATIONS,
                "the fixture must be large enough for the cap to bite",
            );
            held.sort_by_key(|c| conversation_hash(c));
            held.truncate(MAX_HELDOUT_CONVERSATIONS);
            held.iter().map(|c| conversation_hash(c)).collect::<Vec<_>>()
        };

        let first = select();
        assert_eq!(first.len(), MAX_HELDOUT_CONVERSATIONS);
        assert_eq!(first, select(), "the same set must produce the same holdout");

        // Every selected conversation is genuinely in the eval share — the cap must
        // narrow the holdout, never reach into what training is allowed to see.
        for conversation in all
            .iter()
            .filter(|c| first.contains(&conversation_hash(c)))
        {
            assert_eq!(chat_destination(conversation), ChatDestination::Eval);
        }
    }

    // The system prompt is part of the conversation, so two otherwise-identical
    // conversations under different system prompts are different examples and must
    // be hashed apart — otherwise they would always share a destination.
    #[test]
    fn the_system_prompt_participates_in_the_hash() {
        let mut with_system = conversation("hi", "hello");
        with_system.system = Some("be terse".into());
        assert_ne!(
            conversation_hash(&with_system),
            conversation_hash(&conversation("hi", "hello")),
        );
    }

    // The `nth` (user, assistant) pair under `system` that the split assigns to the
    // SFT share. Fixtures for `build_sft_dataset` have to land there, and hardcoding
    // strings that happen to would silently rot the moment the shares change — so
    // the fixture asks rather than assumes.
    fn sft_share_conversation(system: Option<&str>, nth: usize) -> (String, String) {
        let mut found = 0;
        for i in 0.. {
            let mut conv = conversation(&format!("q{i}"), &format!("a{i}"));
            conv.system = system.map(str::to_string);
            if chat_destination(&conv) == ChatDestination::Sft {
                if found == nth {
                    return (format!("q{i}"), format!("a{i}"));
                }
                found += 1;
            }
        }
        unreachable!("the SFT share is a third of the space; one exists")
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
        // The same back-compat rule for `strip_thinking`: absent means the runs
        // already on disk, which kept their reasoning.
        assert!(!cfg.strip_thinking);
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
}

