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
    // Preference tuning on thumbs-rated replies (KTO). Like SFT it fine-tunes an
    // existing model named by `base_model_id`, but from the chat feedback rather
    // than from a conversation corpus.
    Kto,
    // Rejection-sampling fine-tuning (RAFT): ordinary SFT, but on the model's *own*
    // generations that cleared the catalogue's bar.
    //
    // **Why this exists next to KTO rather than instead of it.** They are not
    // alternatives, they are an engine and a brake. RAFT's signal is the full
    // token distribution over a good reply — hundreds of bits — so it moves the
    // model toward the good region fast, which is what teaches a model at this
    // stage to form English at all. But it can only reinforce; it has no way to
    // express "don't do this". KTO's signal is one bit per example, far weaker,
    // but its undesirable term is the only mechanism here that actively pushes
    // probability *down* on a failure mode.
    //
    // Run in that order (RAFT then KTO on the same catalogue) they compose rather
    // than compete: after RAFT the good examples already sit at high probability,
    // so KTO's gradient on them saturates (see `llm::kto::step`) and it spends its
    // budget almost entirely on suppressing the bad half.
    Raft,
}

// The KTO default, matching `llm::kto::KtoParams::default`. 0.1 is the paper's
// value and the usual starting point: low enough that the tune can move, high
// enough that the KL anchor still holds.
fn default_beta() -> f64 {
    0.1
}

/// Whether a run of `mode` renders its chat data with reasoning, given what the
/// request asked for.
///
/// The one place the "pretrain always strips" rule lives, so the dataset builder
/// and anything reconstructing a model's rendering cannot disagree about it. A
/// pretrain corpus carries chat to expose the *scaffold* — the markers a later SFT
/// would otherwise meet at random init — and a thinking block is not scaffold.
pub fn strip_thinking_for(mode: TrainMode, requested: bool) -> llm::chat_template::Reasoning {
    match mode {
        TrainMode::Pretrain => llm::chat_template::Reasoning::Strip,
        _ if requested => llm::chat_template::Reasoning::Strip,
        _ => llm::chat_template::Reasoning::Keep,
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
    // KTO only: strength of the KL anchor holding the tune near the base model.
    // Defaulted rather than required so a body written for the other two modes
    // still deserializes; see `llm::kto::KtoParams::beta` for what it does.
    #[serde(default = "default_beta")]
    pub beta: f64,
    // RAFT and KTO only: which *round's* generations to train on, as the cycle it
    // belongs to plus the round number within it. The catalogue file is append-only
    // and spans every round every cycle has ever run, so without this a round trains
    // on the union of all of them — the oldest and worst generations accumulating the
    // most total gradient, round after round.
    //
    // **This was keyed on the generating model and that did not work.** Every re-run
    // of a cycle starts from the same seed and generates with it, so every run stamped
    // the same `id_model` and the scope did not separate them: measured across four
    // runs of one cycle, the catalogue went 320 → 604 → 900 examples and the round
    // score fell 58.39 → 56.85 → 54.41 in step with it. A round is the unit that is
    // actually unique, and `(id_cycle, round)` names one without a database round-trip
    // — the cycle holds both before it enqueues anything.
    //
    // `None` means "the whole catalogue", which is what a hand-started run from the
    // API gets: outside a cycle there is no round to scope to, and the old behaviour
    // is the only one that has a meaning there.
    #[serde(default)]
    pub catalogue_cycle: Option<Uuid>,
    #[serde(default)]
    pub catalogue_round: Option<i32>,
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

impl TrainConfig {
    /// The round this run's catalogue is scoped to, or `None` for the whole file.
    ///
    /// The two halves are stored as separate fields because that is what survives a
    /// JSONB round trip through the job row cleanly, but every consumer wants them
    /// together — and a half-set pair is meaningless, so this refuses one without the
    /// other rather than inventing a round 0.
    pub fn catalogue_scope(&self) -> Option<(Uuid, i32)> {
        self.catalogue_cycle.zip(self.catalogue_round)
    }
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

// The SFT/RAFT counterpart of `Segments`, written next to `sft_corpus.bin`: one
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

// What `build_dataset` reports as it runs, so the caller can stream progress.
pub enum TokenizeEvent<'a> {
    // Fires before any file is opened, so the dashboard populates immediately.
    // `chars` is the summed byte length from `metadata` — exact for ASCII, up to
    // ~2x high for heavy unicode — and is corrected by `Encode` as files complete.
    Start { files: usize, chars: usize },
    Merge { done: u32, total: u32 },
    // The streaming pass. `chars` is the running exact count over files finished.
    Encode { files_done: usize, files: usize, chars: usize },
    Log(&'a str),
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
// This constant is therefore no longer the memory cliff it was, and it is now the
// binding constraint on tokenizer *quality* instead: see `plan_bpe_sample` for why
// the sample's composition matters more than its size. Raising it is safe on the
// memory side; still measure, since the encode pass that follows is unchanged.
const BPE_SAMPLE_BYTES: usize = 16 * 1024 * 1024;

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
        // A real `uploads/Chat` folder would be shadowed by the virtual entry
        // below, so don't offer it twice under one name.
        if name == CHAT_DATASET {
            continue;
        }
        out.push(DatasetInfo { name, files });
    }

    // The chat conversations, offered like any other dataset even though no folder
    // holds them: they are rendered from `raw/sft/` when a run starts. Counted from
    // the source rather than from the last render, so the number is right before
    // any run has happened and moves the moment conversations are added.
    let chat = count_chat_documents().unwrap_or(0);
    if chat > 0 {
        out.push(DatasetInfo { name: CHAT_DATASET.to_string(), files: chat });
    }

    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

// How many conversations the pretrain share holds. `Ok(0)` rather than an error
// when there are none — an install with no conversations yet should simply not be
// offered the dataset.
fn count_chat_documents() -> io::Result<usize> {
    match read_sft_conversations() {
        Ok(conversations) => Ok(conversations
            .iter()
            .filter(|c| chat_destination(c) == ChatDestination::Pretrain)
            .count()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e),
    }
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
// `Ok(None)` means the caller asked for a cancel and the build stopped early — a
// distinct outcome from `Err`, matching `llm::eval::run`. A cancelled or failed
// build leaves the previous corpus/merges/segments untouched.
pub fn build_dataset(
    max_vocab: u32,
    stages: &[CurriculumStage],
    cancel: Option<Arc<AtomicBool>>,
    on_event: impl Fn(TokenizeEvent),
) -> io::Result<Option<DatasetStats>> {
    // Before anything else, so the run's chat documents are derived from the
    // conversations as they are now rather than from whenever someone last
    // exported them. Cheap (a few hundred small files) and unconditional: gating
    // it on the curriculum naming `Chat` would mean a run that added the stage
    // could still read a stale directory.
    render_chat_documents()?;

    let files = collect_tagged(stages)?;
    // Byte lengths, not character counts: the exact count needs a full read of
    // every file, and this fires before any of them are opened so the dashboard
    // is populated immediately. `Encode` events correct it as files complete.
    let estimated_chars: usize = files.iter().map(|f| f.bytes as usize).sum();
    on_event(TokenizeEvent::Start { files: files.len(), chars: estimated_chars });

    let sample = plan_bpe_sample(&files, BPE_SAMPLE_BYTES);
    let Some(merges) = train_sample_merges(&files, &sample, max_vocab, &cancel, &on_event)? else {
        return Ok(None);
    };

    // A lone <|endoftext|> encodes to its single reserved id; use it as the
    // document separator, exactly like the SFT boundary between conversations.
    let eot_id = *tokenize::bpe::encode("<|endoftext|>", &merges).first().unwrap_or(&256);

    let datasets_dir = storage::datasets_dir();
    let tokenizer_dir = storage::tokenizer_dir();
    fs::create_dir_all(&datasets_dir)?;
    fs::create_dir_all(&tokenizer_dir)?;
    let corpus_partial = datasets_dir.join("corpus.bin.partial");

    let streamed = stream_encode(&files, &merges, eot_id, &corpus_partial, &cancel, &on_event);
    let streamed = match streamed {
        Ok(Some(streamed)) => streamed,
        // Nothing is renamed into place on either path, so the previous dataset
        // survives; only the partial needs clearing.
        other => {
            let _ = fs::remove_file(&corpus_partial);
            return other.map(|_| None);
        }
    };

    let segments = Segments {
        eot_id,
        stages: stages
            .iter()
            .map(|s| StageMeta { name: s.name.clone(), epochs: s.epochs })
            .collect(),
        documents: streamed.documents,
    };
    finalize_dataset(&corpus_partial, &merges, &segments)?;

    Ok(Some(DatasetStats {
        files: files.len(),
        chars: streamed.chars,
        vocab_size: tokenize::bpe::vocab_size(merges.len()),
        merges: merges.len(),
        corpus_len: streamed.corpus_len,
        documents: segments.documents.len(),
        special_ids_seen: streamed.special_ids,
    }))
}

// One selected file, with the size the sample planner needs. Sizes come from
// `metadata` so planning costs no reads.
struct TaggedFile {
    stage: usize,
    dataset: String,
    path: PathBuf,
    bytes: u64,
}

// Every selected file, tagged with the stage it belongs to. A dataset is a
// subfolder of uploads/; `select_txt` pulls `count` .txt files from it (0 => all).
// Files keep their per-stage grouping so the training loop can feed one stage at
// a time.
fn collect_tagged(stages: &[CurriculumStage]) -> io::Result<Vec<TaggedFile>> {
    let uploads = storage::uploads_dir();
    let mut tagged: Vec<TaggedFile> = Vec::new();
    for (stage, spec) in stages.iter().enumerate() {
        for sel in &spec.datasets {
            if !is_safe_dataset_name(&sel.name) {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid dataset name"));
            }
            // `Chat` is rendered from `raw/sft/`, not uploaded, so it is read from
            // where this build just wrote it. The name takes precedence over an
            // `uploads/Chat` folder if one exists.
            let dir = if sel.name == CHAT_DATASET {
                chat_documents_dir()
            } else {
                uploads.join(&sel.name)
            };
            for path in select_txt(&dir, sel.count, true)? {
                let bytes = fs::metadata(&path)?.len();
                tagged.push(TaggedFile { stage, dataset: sel.name.clone(), path, bytes });
            }
        }
    }

    if tagged.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no .txt files found in the selected datasets",
        ));
    }
    Ok(tagged)
}

// How much of one file to feed the tokenizer trainer.
struct SampleSlice {
    file: usize,
    bytes: usize,
}

// Which slices of which files train the BPE merges.
//
// The tokenizer learns from a bounded sample rather than the whole corpus. That
// was originally a memory constraint; since `bpe::train` started deduplicating its
// input it is mostly a time one, and `BPE_SAMPLE_BYTES` says what the budget now
// buys.
//
// Shares are **equal per dataset, not proportional to size**. Proportional would
// hand a large web-text dataset essentially the whole budget and leave a small
// early-curriculum dataset with kilobytes, so that stage would tokenize at nearly
// byte level — precisely where token efficiency matters most for a small model.
//
// **This is worth re-examining, and has not been.** On the corpus that trained
// `Large 1` the equal split gave FineWeb 27.9% of the sample while it was 98.4% of
// the text actually encoded, so the merges were fitted to a distribution the corpus
// does not have. The stated worry cuts the other way too: simple English is a
// *subset* of general English, so merges learned from web text cover TinyStories
// far better than the reverse. Measure before changing it — but do not read the
// current split as a settled result.
// Water-filling from the smallest group up means a dataset too small to use its
// share donates the remainder to the larger ones automatically.
fn plan_bpe_sample(files: &[TaggedFile], budget: usize) -> Vec<SampleSlice> {
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, f) in files.iter().enumerate() {
        groups.entry(f.dataset.as_str()).or_default().push(i);
    }

    // Smallest total first, so each step's `remaining / groups_left` share already
    // accounts for what the groups before it could not use.
    let mut ordered: Vec<(u64, Vec<usize>)> = groups
        .into_values()
        .map(|idx| (idx.iter().map(|&i| files[i].bytes).sum::<u64>(), idx))
        .collect();
    ordered.sort_by_key(|(total, _)| *total);

    let mut slices = Vec::new();
    let mut remaining = budget;
    let mut groups_left = ordered.len();
    for (total, idx) in ordered {
        if groups_left == 0 {
            break;
        }
        let share = (remaining / groups_left).min(total as usize);
        groups_left -= 1;
        remaining -= share;

        // Stride across the group rather than taking its first files, which on a
        // sharded dataset would sample one shard and miss the rest of its range.
        let mut left = share;
        let wanted = idx.len().min(share.div_ceil(1).max(1));
        let stride = (idx.len() / wanted.max(1)).max(1);
        for &i in idx.iter().step_by(stride) {
            if left == 0 {
                break;
            }
            let take = (files[i].bytes as usize).min(left);
            if take > 0 {
                slices.push(SampleSlice { file: i, bytes: take });
                left -= take;
            }
        }
        // Anything the stride could not place (every strided file was smaller than
        // its share) goes back into the pool for the groups behind.
        remaining += left;
    }
    slices
}

// Reads the sampled slices and trains the merge table on them.
//
// Cancellation is checked between files. It is deliberately **not** checked inside
// `bpe::train`, which would mean threading a flag through the tokenizer crate: the
// sample is bounded, so that call is the one uninterruptible window in the build.
fn train_sample_merges(
    files: &[TaggedFile],
    sample: &[SampleSlice],
    max_vocab: u32,
    cancel: &Option<Arc<AtomicBool>>,
    on_event: &impl Fn(TokenizeEvent),
) -> io::Result<Option<Vec<((u32, u32), u32)>>> {
    let mut pre: Vec<Vec<char>> = Vec::new();
    let mut buf = String::new();
    for slice in sample {
        if cancelled(cancel) {
            return Ok(None);
        }
        let mut reader = BufReader::new(File::open(&files[slice.file].path)?);
        // One chunk capped at the slice size: whole lines only, so the sample never
        // ends mid-marker or mid-character.
        if read_chunk(&mut reader, slice.bytes, CHUNK_MAX_BYTES, &mut buf, &files[slice.file].path)? {
            pre.extend(tokenize::pretokenize::pre_tokenize_text(&buf));
        }
    }

    if pre.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the selected datasets hold no text to train a tokenizer on",
        ));
    }

    // The crate reports progress as a plain log string ("BPE merge x/y"); forward
    // the raw line for the log panel and parse the counts out for the dashboard.
    // The corpus it returns is discarded — the full corpus, including these bytes,
    // is re-encoded in the streaming pass so there is exactly one code path
    // producing token ids.
    let (_, _, merges) = tokenize::bpe::train(pre, Some(max_vocab), |line| {
        if let Some((done, total)) = parse_merge_progress(line) {
            on_event(TokenizeEvent::Merge { done, total });
        }
        on_event(TokenizeEvent::Log(line));
    })?;
    Ok(Some(merges))
}

// What the streaming pass produced.
struct Streamed {
    documents: Vec<DocSpan>,
    corpus_len: usize,
    chars: usize,
    special_ids: Vec<u32>,
}

// Encodes every file into `corpus_path`, one chunk at a time.
//
// This is the pass that makes a multi-gigabyte corpus fit: nothing larger than one
// chunk is ever resident, so peak memory is constant in corpus size. Chunk cuts
// land only on newlines, which `pretokenize::split_punctuation` already treats as
// a hard pretoken boundary, so the ids are identical to encoding each file whole
// (`bpe::tests::test_chunking_at_newlines_is_seam_free` is the guard on that).
fn stream_encode(
    files: &[TaggedFile],
    merges: &[((u32, u32), u32)],
    eot_id: u32,
    corpus_path: &Path,
    cancel: &Option<Arc<AtomicBool>>,
    on_event: &impl Fn(TokenizeEvent),
) -> io::Result<Option<Streamed>> {
    let encoder = tokenize::bpe::Encoder::new(merges);
    let mut writer = tokenize::bpe::CorpusWriter::create(&corpus_path.to_string_lossy())?;
    let mut documents: Vec<DocSpan> = Vec::new();
    let mut seen = vec![false; tokenize::pretokenize::NUM_SPECIAL];
    let mut chars = 0usize;

    // Reused across every chunk of every file, so the streaming loop allocates
    // nothing per iteration.
    let mut text = String::new();
    let mut ids: Vec<u32> = Vec::new();

    // At tens of thousands of files one event per file is tens of thousands of
    // broadcast frames and state-lock acquisitions; ~200 over a build is plenty to
    // animate a progress bar.
    let report_every = (files.len() / PROGRESS_REPORTS_PER_BUILD).max(1);

    for (index, file) in files.iter().enumerate() {
        let mut reader = BufReader::new(File::open(&file.path)?);
        // Reserved before the first chunk: the separator belongs between files, and
        // a file that turns out to be empty must not leave a stray one behind.
        let start = writer.len() + usize::from(!writer.is_empty());
        let mut len = 0usize;

        while read_chunk(&mut reader, CHUNK_TARGET_BYTES, CHUNK_MAX_BYTES, &mut text, &file.path)? {
            // Between chunks, not just between files: one large file is minutes of
            // work, and a cancel that waits for it looks broken.
            if cancelled(cancel) {
                return Ok(None);
            }
            chars += text.chars().count();
            ids.clear();
            encoder.encode_into(&text, &mut ids);
            if ids.is_empty() {
                continue;
            }
            if len == 0 && !writer.is_empty() {
                writer.write(&[eot_id])?;
            }
            for &id in &ids {
                if let Some(slot) = (id as usize).checked_sub(256) {
                    if let Some(flag) = seen.get_mut(slot) {
                        *flag = true;
                    }
                }
            }
            writer.write(&ids)?;
            len += ids.len();
        }

        if len > 0 {
            documents.push(DocSpan {
                stage: file.stage,
                dataset: file.dataset.clone(),
                start,
                len,
            });
        }
        if index % report_every == 0 || index + 1 == files.len() {
            on_event(TokenizeEvent::Encode {
                files_done: index + 1,
                files: files.len(),
                chars,
            });
        }
    }

    let corpus_len = writer.finish()?;
    let special_ids = seen
        .iter()
        .enumerate()
        .filter(|&(_, &flag)| flag)
        .map(|(slot, _)| 256 + slot as u32)
        .collect();
    Ok(Some(Streamed { documents, corpus_len, chars, special_ids }))
}

// Publishes the build by renaming the partials into place.
//
// The corpus goes last so its appearance is the commit point. Three renames are
// not atomic together and nothing here pretends otherwise: there is no I/O between
// them, and every failure that actually happens — a cancel, an OOM, a full disk —
// happens during the streaming pass minutes earlier, before any of this runs.
fn finalize_dataset(
    corpus_partial: &Path,
    merges: &[((u32, u32), u32)],
    segments: &Segments,
) -> io::Result<()> {
    let datasets_dir = storage::datasets_dir();
    let tokenizer_dir = storage::tokenizer_dir();
    let merges_partial = tokenizer_dir.join("merges.bin.partial");
    let segments_partial = datasets_dir.join("segments.json.partial");

    tokenize::bpe::save_merges(merges, &merges_partial.to_string_lossy())?;
    save_segments(segments, &segments_partial)?;

    fs::rename(&merges_partial, tokenizer_dir.join("merges.bin"))?;
    fs::rename(&segments_partial, datasets_dir.join("segments.json"))?;
    fs::rename(corpus_partial, datasets_dir.join("corpus.bin"))?;
    Ok(())
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

    let warm_start_ids = untrained_marker_ids(&base_model_dir, &sft_corpus_path)?;

    llm::train::sft_train(
        &base_model_dir,
        &sft_corpus_path,
        &sft_mask_path,
        &out_model_dir,
        sft_plan(cfg)?,
        &warm_start_ids,
        cfg.lr,
        Some(&checkpoints_dir),
        on_progress,
        Some(cancel),
    )
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
pub fn model_reasoning(model_dir: &Path) -> llm::chat_template::Reasoning {
    let stripped = fs::read_to_string(model_dir.join("meta.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|meta| meta.get("config")?.get("strip_thinking")?.as_bool())
        .unwrap_or(false);
    if stripped {
        llm::chat_template::Reasoning::Strip
    } else {
        llm::chat_template::Reasoning::Keep
    }
}

// A whole corpus file as ids, widened to the u32 the rest of this module speaks.
//
// Only for the SFT/KTO corpora, which are bounded by the conversation set and so
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
// corpus with its base's — so the record composes down a pretrain → SFT → KTO
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

// The conversation order for an SFT/RAFT run: one document per conversation,
// reshuffled every epoch.
//
// A fine-tune is where a fixed order bites hardest. The corpus is small enough
// that every epoch is the same few hundred conversations, and generation writes
// them out grouped — a whole call's batch of one topic lands consecutively, so a
// fixed order trains topic-by-topic and the last conversations in the file are the
// ones the model ends every epoch believing. Reordering costs one permutation per
// epoch and removes that.
//
// Unlike pretraining, this doesn't ask the config: `cfg.shuffle` is a pretrain knob
// (its composer offers it, the fine-tune composers don't), and there is no
// curriculum here whose order a shuffle would destroy. A dataset built before the
// sidecar existed, or one holding a single conversation, has nothing to permute and
// falls back to the whole corpus in built order.
fn sft_plan(cfg: &TrainConfig) -> io::Result<llm::train::DataPlan> {
    let sidecar = storage::datasets_dir().join(SFT_SEGMENTS_FILE);
    let segments = match load_sft_segments(&sidecar)? {
        Some(segments) if segments.conversations.len() > 1 => segments,
        _ => return Ok(llm::train::DataPlan::whole_corpus(cfg.epochs)),
    };

    Ok(llm::train::DataPlan {
        stages: vec![llm::train::Stage {
            name: None,
            epochs: cfg.epochs,
            docs: segments
                .conversations
                .iter()
                .map(|c| llm::train::DocSpan { start: c.start, len: c.len })
                .collect(),
        }],
        eot_id: segments.eot_id,
        shuffle: true,
        seed: resolve_seed(cfg.seed),
    })
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
    budget: Option<llm::chat_template::ThinkBudget>,
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
    // compose down a pretrain → SFT → KTO chain — without it, a KTO run would read
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

/// Where a run's rendered chat documents land: under `datasets/`, with the rest of
/// what a build produces, and rebuilt from scratch every time.
fn chat_documents_dir() -> PathBuf {
    storage::datasets_dir().join("chat")
}

// Renders this run's share of the conversations into `datasets/chat/`, one file
// per conversation, and returns how many.
//
// Called at the top of every `build_dataset` so a pretrain is self-contained: it
// derives its chat documents from `raw/sft/` the same way it derives its
// tokenizer and its corpus, rather than depending on someone having run an export
// first. Adding conversations and starting a run is enough.
//
// Only the pretrain share (see `chat_destination`). The SFT share is left alone —
// `build_sft_dataset` reads `raw/sft/` and filters it directly, so the
// conversations exist in exactly one place and there is nothing to keep in sync.
//
// One file per conversation because `build_dataset` records a document span per
// file: that is what lets the training loop shuffle and stage conversations
// individually rather than as one undifferentiated blob.
fn render_chat_documents() -> io::Result<usize> {
    // No conversations is a normal state — a first pretrain, or an install that
    // does not use chat data at all — so it renders nothing rather than failing a
    // run that never asked for the `Chat` dataset in the first place.
    let conversations = match read_sft_conversations() {
        Ok(conversations) => conversations,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };

    let dir = chat_documents_dir();
    // A full rebuild, not an append: a conversation removed from the source must
    // not survive as an orphan file from a previous run.
    reset_dir(&dir)?;

    let mut written = 0usize;
    for (index, conv) in conversations.iter().enumerate() {
        if chat_destination(conv) != ChatDestination::Pretrain {
            continue;
        }
        let turns = conversation_turns(conv)?;
        // Unconditionally stripped — see `strip_thinking_for`. This is the only
        // path that writes chat into a *pretrain* corpus.
        let text = llm::chat_template::render_conversation(
            conv.system.as_deref(),
            &turns,
            conv.budget,
            llm::chat_template::Reasoning::Strip,
        );
        fs::write(dir.join(format!("chat{index:05}.txt")), text)?;
        written += 1;
    }
    Ok(written)
}

// Empties a derived directory, creating it if absent. Only the files are removed,
// so a directory someone is watching keeps its identity.
/// Writes the held-out share of `raw/sft/` to `eval/chat/heldout.jsonl`.
///
/// `chat_destination` has split conversations three ways since it was written, and
/// the `Eval` 15% had **no consumer at all** — it was computed, excluded from both
/// pretraining and SFT, and then dropped on the floor. `llm::eval::chat` has been
/// reading `eval/chat/` for just as long and finding nothing, so `"chat": null`
/// appeared in every benchmark report ever produced, and the one instrument built to
/// show a fine-tune working never ran. This is the missing half.
///
/// Called by `jobs::queue` before a benchmark, next to `export_preference_data`, for
/// the reason the module header gives about `render_chat_documents`: a run derives its
/// own inputs, rather than depending on someone having remembered to run an export.
/// The split is a hash of conversation *content*, so this is deterministic and
/// idempotent — running it twice writes the same bytes, and adding conversations never
/// re-partitions the existing ones.
///
/// One file rather than one per conversation: `eval::chat::load` reads whole JSONL
/// files and this set is consumed as a unit, so there is no span bookkeeping to do.
pub fn export_chat_eval_set() -> io::Result<usize> {
    // `read_sft_conversations` reports "no conversations" as `NotFound`, which is the
    // right answer for a *training* run — it cannot build an SFT dataset out of
    // nothing. Here it is the ordinary state of a fresh install, and the markup suite
    // still scores its pair stage without a chat set, so it must not surface as an
    // error the queue logs before every single benchmark.
    let conversations = match read_sft_conversations() {
        Ok(conversations) => conversations,
        Err(e) if e.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(e),
    };
    let held_out: Vec<&SftConversation> = conversations
        .iter()
        .filter(|c| chat_destination(c) == ChatDestination::Eval)
        .collect();


    let dir = storage::eval_dir().join("chat");
    reset_dir(&dir)?;
    if held_out.is_empty() {
        // No conversations at all is the normal state of a fresh install, and an
        // empty directory is exactly how `eval::chat::load` expects to be told so.
        return Ok(0);
    }

    let mut out = String::new();
    for conversation in &held_out {
        // Serialized back into the shape it came in as — `SftConversation` skips a
        // `None` system field for precisely this round trip.
        out.push_str(&serde_json::to_string(conversation).map_err(io::Error::other)?);
        out.push('\n');
    }
    fs::write(dir.join("heldout.jsonl"), out)?;
    Ok(held_out.len())
}

fn reset_dir(dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    for entry in fs::read_dir(dir)?.flatten() {
        if entry.path().is_file() {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

// Every conversation in `raw/sft/*.jsonl`, in a stable order.
//
// Shared by the exporter and the SFT dataset builder so both see the same set; the
// order is by sorted filename then line, which keeps a rebuild byte-identical.
fn read_sft_conversations() -> io::Result<Vec<SftConversation>> {
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

    let mut conversations = Vec::new();
    for path in &files {
        let content = fs::read_to_string(path)?;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            conversations.push(serde_json::from_str(line).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bad conversation JSON in {}: {}", path.display(), e),
                )
            })?);
        }
    }
    Ok(conversations)
}

// Whether the conversation still has something for the model to learn to say once
// `reasoning` has been applied. Always true under `Keep`, where nothing is removed.
fn any_reply_survives(
    turns: &[llm::chat_template::Turn],
    reasoning: llm::chat_template::Reasoning,
) -> bool {
    if reasoning == llm::chat_template::Reasoning::Keep {
        return true;
    }
    turns.iter().any(|t| {
        t.role == llm::chat_template::Role::Assistant
            && !llm::chat_template::strip_thinking(&t.text).is_empty()
    })
}

// A conversation's turns as chat-template turns, applying the long-user-turn cap.
fn conversation_turns(conv: &SftConversation) -> io::Result<Vec<llm::chat_template::Turn>> {
    let mut turns = Vec::with_capacity(conv.turns.len());
    for t in &conv.turns {
        let role = parse_role(&t.role)?;
        let text = match role {
            llm::chat_template::Role::User => cap_user_turn(&t.text),
            llm::chat_template::Role::Assistant => t.text.clone(),
        };
        turns.push(llm::chat_template::Turn { role, text });
    }
    Ok(turns)
}

// Reads every `.jsonl` conversation file in the SFT dir, renders each conversation
// through the canonical chat template using the **base model's** tokenizer, and
// writes the aligned `sft_corpus.bin` + `sft_mask.bin` to `datasets/`.
//
// The tokenizer is loaded from `base_model_dir/merges.bin` and is NOT retrained:
// SFT is a fine-tune, so the token ids must match exactly what the base model
// already learned. Training a fresh BPE here would renumber the vocab (and the
// special-marker ids) and the fine-tune would be gibberish.
//
// `reasoning` comes from the run's `strip_thinking`, via `strip_thinking_for`. A
// conversation whose assistant turns are *all* empty once stripped is dropped
// rather than trained on: it would otherwise teach "answer with nothing and stop",
// which is a worse lesson than the reasoning it replaced. (Measured against the
// corpus on disk when this landed: 0 of 1,849 assistant turns, so this is
// insurance, not a filter that is doing quiet work.)
pub fn build_sft_dataset(
    base_model_dir: &Path,
    reasoning: llm::chat_template::Reasoning,
) -> io::Result<SftDatasetStats> {
    let merges_path = base_model_dir.join("merges.bin");
    let merges = tokenize::bpe::load_merges(&merges_path.to_string_lossy())?;

    // Between conversations, a single <|endoftext|> with mask 0 marks the boundary
    // so the model sees a separator but is never trained to predict across two
    // unrelated conversations.
    let eot = tokenize::bpe::encode("<|endoftext|>", &merges);

    // Only this base model's SFT share — the rest of the set is seeded into
    // pretraining or held back for eval. See the chat-split header for why the
    // three must not overlap.
    let all = read_sft_conversations()?;
    let share: Vec<&SftConversation> = all
        .iter()
        .filter(|c| chat_destination(c) == ChatDestination::Sft)
        .collect();
    if share.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no conversations fell in the SFT share of the chat split",
        ));
    }

    // The window every rendered conversation has to fit. Read off the base model,
    // because SFT is a fine-tune and the only window that matters is the one the
    // weights were built for.
    let ctx = base_ctx_window(base_model_dir);

    let mut corpus: Vec<u32> = Vec::new();
    let mut mask: Vec<u8> = Vec::new();
    let mut spans: Vec<SftSpan> = Vec::new();
    let mut conversations = 0usize;
    let mut skipped_too_long = 0usize;

    for conv in share {
        let turns = conversation_turns(conv)?;
        if !any_reply_survives(&turns, reasoning) {
            continue;
        }

        let (ids, m) = llm::chat_template::build_training_example(
            conv.system.as_deref(),
            &turns,
            conv.budget,
            &merges,
            reasoning,
        );
        if ids.is_empty() {
            continue;
        }
        if ids.len() > ctx {
            // Not merely wasteful: this would become a span in `sft_segments.json`
            // longer than the window `sft_train` is given, so the run would train on
            // a silently truncated example. The same guard KTO and RAFT already have.
            skipped_too_long += 1;
            continue;
        }

        // Boundary before every conversation after the first.
        if !corpus.is_empty() {
            corpus.extend(&eot);
            mask.extend(std::iter::repeat_n(0u8, eot.len()));
        }
        spans.push(SftSpan { start: corpus.len(), len: ids.len() });
        corpus.extend(&ids);
        mask.extend(&m);
        conversations += 1;
    }

    let chat_trainable = mask.iter().filter(|&&b| b != 0).count();
    let replay_tokens =
        append_prose_replay(&mut corpus, &mut mask, &mut spans, base_model_dir, &eot, chat_trainable)?;

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
    save_sft_segments(
        &SftSegments { eot_id: *eot.first().unwrap_or(&256), conversations: spans },
        &datasets.join(SFT_SEGMENTS_FILE),
    )?;

    let trainable_tokens = mask.iter().filter(|&&b| b != 0).count();
    Ok(SftDatasetStats {
        conversations,
        total_tokens: corpus.len(),
        trainable_tokens,
        replay_tokens,
        skipped_too_long,
        special_ids_seen: markers_after_training(base_model_dir, &corpus),
    })
}

/// The context window a base model was built for, from its own `meta.json`.
///
/// Best-effort, and `usize::MAX` when it cannot be read — an unknown window means
/// there is no basis for calling anything too long, and refusing every conversation
/// would be a far worse failure than keeping one that overruns. Every model this
/// has ever written records it; the fallback is for a hand-assembled directory.
fn base_ctx_window(base_model_dir: &Path) -> usize {
    fs::read_to_string(base_model_dir.join("meta.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|meta| meta.get("config")?.get("ctx_window")?.as_u64())
        .map(|ctx| ctx as usize)
        .unwrap_or(usize::MAX)
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

// Appends prose from the base model's pretraining corpus, fully masked in (every
// position trains, as it did during pretraining), and records a span per document
// so the epoch shuffle interleaves replay with conversations rather than leaving it
// in one block at the end.
//
// Returns the number of replay tokens added, which is 0 when replay is skipped.
//
// **The guard is the important part.** `corpus.bin` is only meaningful under the
// tokenizer that produced it, and it is overwritten by every pretrain run. If a
// later pretrain rebuilt it with a fresh BPE, its ids mean something else entirely
// under this base model's `merges.bin` and replaying it would train the model on
// noise — silently, since nothing downstream can tell. So the two tokenizers are
// compared byte for byte and replay is skipped loudly on a mismatch.
fn append_prose_replay(
    corpus: &mut Vec<u32>,
    mask: &mut Vec<u8>,
    spans: &mut Vec<SftSpan>,
    base_model_dir: &Path,
    eot: &[u32],
    chat_trainable: usize,
) -> io::Result<usize> {
    let budget = (chat_trainable as f64 * REPLAY_TOKENS_PER_CHAT_TOKEN) as usize;
    if budget == 0 {
        return Ok(0);
    }

    let datasets = storage::datasets_dir();
    if !same_tokenizer(base_model_dir, &storage::tokenizer_dir().join("merges.bin"))? {
        eprintln!(
            "SFT replay: the pretrain corpus was tokenized with a different tokenizer \
             than this base model — skipping replay rather than training on ids that \
             mean something else"
        );
        return Ok(0);
    }

    let Some(segments) = load_segments(&datasets.join("segments.json"))? else {
        eprintln!("SFT replay: no segments.json next to corpus.bin — skipping replay");
        return Ok(0);
    };
    // The spans are read individually rather than loading corpus.bin, which at
    // pretrain scale is hundreds of millions of tokens against the few hundred
    // thousand replay actually samples.
    let prose_path = datasets.join("corpus.bin").to_string_lossy().to_string();
    if segments.documents.is_empty() {
        return Ok(0);
    }

    // Shuffled so the replay is a sample of the whole corpus rather than its first
    // documents, which are one dataset and one curriculum stage. Seeded from the
    // budget so a rebuild of the same dataset picks the same documents.
    let mut order: Vec<usize> = (0..segments.documents.len()).collect();
    shuffle_indices(&mut order, budget as u64);

    // Breadth first, then depth. One pass takes at most `MAX_REPLAY_DOC_TOKENS` from
    // each document, so replay is a wide sample of the corpus before it is a deep one;
    // a further pass takes the *next* slice of each, at a fresh offset.
    //
    // The passes exist because one is not enough any more. A pass over the 11,104
    // documents on disk yields 7.1M tokens, which was ample against an SFT set of a
    // couple of thousand conversations and is 14% of the budget against 454,000 —
    // the fine-tune would have been anchored at a seventh of the intended strength,
    // silently, with `replay_tokens` the only sign and nothing to compare it to.
    //
    // Bounded: `offset` grows by `MAX_REPLAY_DOC_TOKENS` every pass, so every
    // document is exhausted in finite time and `progressed` ends the loop when they
    // all are — replay stops short of its budget rather than repeating the corpus,
    // because the same tokens twice is not more of the old distribution.
    let mut added = 0usize;
    let mut pass = 0usize;
    while added < budget {
        let offset = pass * MAX_REPLAY_DOC_TOKENS;
        let mut progressed = false;
        for &index in &order {
            if added >= budget {
                break;
            }
            let doc = &segments.documents[index];
            let Some(left) = doc.len.checked_sub(offset) else {
                continue;
            };
            let want = left.min(MAX_REPLAY_DOC_TOKENS).min(budget - added);
            // A one-token span has nothing to predict and would be a chunk the
            // training loop skips; not worth a document boundary.
            if want < 2 {
                continue;
            }
            let span = tokenize::bpe::load_corpus_span(&prose_path, doc.start + offset, want)?;
            if span.len() < 2 {
                continue;
            }
            progressed = true;

            if !corpus.is_empty() {
                corpus.extend(eot);
                mask.extend(std::iter::repeat_n(0u8, eot.len()));
            }
            spans.push(SftSpan { start: corpus.len(), len: span.len() });
            corpus.extend(span.iter().map(|&t| u32::from(t)));
            // Every position trains: this is ordinary language modelling, the
            // objective the tokens were pretrained under. The masking convention is a
            // property of chat examples, not of the file format.
            mask.extend(std::iter::repeat_n(1u8, span.len()));
            added += span.len();
        }
        if !progressed {
            break;
        }
        pass += 1;
    }

    Ok(added)
}

// Whether the base model's tokenizer is byte-identical to the one that produced
// `corpus.bin`. A missing file on either side is a mismatch, not an error — the
// caller's response to both is the same, and "cannot prove they match" is the
// answer that keeps a run from training on misinterpreted ids.
fn same_tokenizer(base_model_dir: &Path, corpus_merges: &Path) -> io::Result<bool> {
    let (Ok(base), Ok(built)) = (
        fs::read(base_model_dir.join("merges.bin")),
        fs::read(corpus_merges),
    ) else {
        return Ok(false);
    };
    Ok(base == built)
}

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

// The marker set a fine-tune's output model should record: everything its corpus
// contains, plus everything its base already knew. A base with no record
// contributes nothing, so the union degrades to "what this corpus taught" rather
// than inventing history for a model that never wrote any down.
fn markers_after_training(base_model_dir: &Path, corpus: &[u32]) -> Vec<u32> {
    let mut seen = special_ids_in(corpus);
    seen.extend(model_special_ids_seen(base_model_dir).unwrap_or_default());
    seen.sort_unstable();
    seen.dedup();
    seen
}

#[derive(Debug, Serialize)]
pub struct RaftDatasetStats {
    pub examples: usize,
    pub total_tokens: usize,
    pub trainable_tokens: usize,
    // Winners dropped as exact repeats of an earlier one. Reported because a high
    // number is the model saying the same thing to every prompt — which trains fine
    // and narrows hard, and is invisible from the example count alone.
    pub duplicates: usize,
    pub skipped_too_long: usize,
    // As `SftDatasetStats::replay_tokens`. Zero here means the tokenizer guard
    // refused the replay, and the run is the unanchored self-train that replay
    // exists to prevent — which for RAFT is the more dangerous of the two modes,
    // since its examples are the model's own output.
    pub replay_tokens: usize,
    // As `SftDatasetStats::special_ids_seen` — RAFT writes the same `sft_corpus.bin`
    // and trains through the same entry point, so its output model must carry the
    // same record or a later fine-tune reads it as having seen nothing.
    pub special_ids_seen: Vec<u32>,
}

// Builds the RAFT dataset: every catalogue entry labelled `up`, rendered through
// the same chat template and masking convention SFT uses.
//
// **This deliberately writes `sft_corpus.bin` / `sft_mask.bin`.** RAFT *is* SFT —
// identical loss, identical masking, identical training loop — differing only in
// where the conversations came from. A separate pair of files would mean a second
// copy of `sft_train` reading them, and two loops that must not drift apart. Phase
// 1 always rebuilds these before phase 2 reads them, so an SFT run and a RAFT run
// overwriting each other's dataset is no more of a problem than two SFT runs are.
//
// **Exact duplicates are dropped.** A model that answers three prompts with the
// same sentence contributes that sentence three times, and max-likelihood training
// would weight it triple — accelerating exactly the collapse this loop has to avoid.
// Near-duplicates (same sentence, one word different) are *not* caught here; that
// wants an edit-distance or n-gram overlap pass, and is the obvious next guard.
//
// **It replays prose, for the same reason `build_sft_dataset` does and more
// urgently.** RAFT is plain masked cross-entropy through `sft_train` — no KL term,
// nothing holding it near the base model — over a few hundred short replies. KTO
// at least has `ref_logprob` anchoring it. Left unanchored, this stage is a
// full-parameter fine-tune on the model's own output, which is the textbook
// recipe for collapsing onto it: the model's language ability decays into the
// register of whatever cleared the judge's bar, and the language benchmark reads
// that as a regression because it is one. See `append_prose_replay`.
pub fn build_raft_dataset(
    base_model_dir: &Path,
    catalogue_round: Option<(Uuid, i32)>,
) -> io::Result<RaftDatasetStats> {
    let merges_path = base_model_dir.join("merges.bin");
    let merges = tokenize::bpe::load_merges(&merges_path.to_string_lossy())?;
    let eot = tokenize::bpe::encode("<|endoftext|>", &merges);

    // The same lines `build_kto_dataset` reads — RAFT just keeps one half of the
    // split instead of both.
    let lines = read_catalogue(catalogue_round)?;

    let model = llm::training::transformer::Model::load(&base_model_dir.to_string_lossy())?;
    let ctx = model.ctx_window();

    let mut corpus: Vec<u32> = Vec::new();
    let mut mask: Vec<u8> = Vec::new();
    let mut spans: Vec<SftSpan> = Vec::new();
    let mut examples = 0usize;
    let mut duplicates = 0usize;
    let mut skipped_too_long = 0usize;
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();

    for line in &lines {
        if line.label != "up" {
            continue;
        }
        if line.prompt.trim().is_empty() || line.response.trim().is_empty() {
            continue;
        }
        let key = (
            line.prompt.trim().to_string(),
            line.response.trim().to_string(),
        );
        if !seen.insert(key) {
            duplicates += 1;
            continue;
        }

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
        // `Keep` is the identity here: these are the model's *own* generations,
        // produced unbudgeted (see `labeling::autolabel`), so there is no thinking
        // block to strip — and a model trained under `Strip` does not emit one.
        let (ids, m) = llm::chat_template::build_training_example(
            None,
            &turns,
            None,
            &merges,
            llm::chat_template::Reasoning::Keep,
        );

        // Unlike KTO — where a truncated example would be scored against a whole-reply
        // reference log-probability and mean two different things — a long example
        // here is merely one SFT can't fit in a window. Dropped for the same reason
        // either way: a half-rendered conversation teaches half a conversation.
        if ids.len() < 2 || ids.len() > ctx {
            skipped_too_long += 1;
            continue;
        }
        if !m.iter().any(|&b| b != 0) {
            continue;
        }

        if !corpus.is_empty() {
            corpus.extend(&eot);
            mask.extend(std::iter::repeat_n(0u8, eot.len()));
        }
        spans.push(SftSpan { start: corpus.len(), len: ids.len() });
        corpus.extend(&ids);
        mask.extend(&m);
        examples += 1;
    }

    if examples == 0 {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no approved replies in the catalogue — build the catalogue first, or \
             the bar is set above anything this model currently produces",
        ));
    }

    // Counted before replay: the budget is a multiple of the *reply* tokens, and
    // replay's own fully-masked-in tokens must not enlarge the budget they are
    // being measured against.
    let reply_trainable = mask.iter().filter(|&&b| b != 0).count();
    let replay_tokens =
        append_prose_replay(&mut corpus, &mut mask, &mut spans, base_model_dir, &eot, reply_trainable)?;

    let datasets = storage::datasets_dir();
    fs::create_dir_all(&datasets)?;
    tokenize::bpe::save_corpus(&corpus, &datasets.join("sft_corpus.bin").to_string_lossy())?;
    fs::write(datasets.join("sft_mask.bin"), &mask)?;
    save_sft_segments(
        &SftSegments { eot_id: *eot.first().unwrap_or(&256), conversations: spans },
        &datasets.join(SFT_SEGMENTS_FILE),
    )?;

    Ok(RaftDatasetStats {
        examples,
        total_tokens: corpus.len(),
        trainable_tokens: mask.iter().filter(|&&b| b != 0).count(),
        duplicates,
        skipped_too_long,
        replay_tokens,
        special_ids_seen: markers_after_training(base_model_dir, &corpus),
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
    // As `SftDatasetStats::special_ids_seen`, so the record survives a third hop.
    pub special_ids_seen: Vec<u32>,
}

// One rated reply, as written by `queue::export_preference_data` (the human half)
// or `labeling::autolabel` (the generated half).
#[derive(Deserialize)]
struct RatedLine {
    prompt: String,
    response: String,
    label: String,
    // Which model generated this reply. Absent on the human-rated export and on
    // every catalogue line written before the field existed — see
    // `read_catalogue` for what that absence means.
    #[serde(default)]
    id_cycle: Option<Uuid>,
    #[serde(default)]
    round: Option<i32>,
}

// Every rated reply in `raw/preference/`, optionally narrowed to one round's
// generations. `only` is `(id_cycle, round)`; see `TrainConfig::catalogue_cycle` for
// why the round rather than the generating model.
//
// **The exemption is by file, not by missing stamp.** `feedback.jsonl` is kept
// whole under every filter: it is what a person clicked, it is the seed and the
// audit rather than any round's output, and it is a few dozen lines against
// thousands. Everything else is generated, and a generated line with no round stamp
// is one written before the stamp existed — which is precisely the accumulated
// backlog the scoping exists to stop training on, so under a filter it is dropped.
//
// Reading the stamp alone cannot make that distinction: human lines have no stamp
// either, so "keep the unstamped" grandfathers the entire pre-existing catalogue
// and the filter selects nothing it was meant to select.
//
// Both dataset builders read through this so they can never disagree about what
// the catalogue *is*; they differ only in which half of it they keep.
fn read_catalogue(only: Option<(Uuid, i32)>) -> io::Result<Vec<RatedLine>> {
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
        let human = path.file_name().and_then(|n| n.to_str())
            == Some(storage::HUMAN_FEEDBACK_FILE);
        let content = fs::read_to_string(path)?;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // A malformed line is skipped rather than fatal: this file is a machine
            // export, and one bad row should not block a tune over hundreds of good
            // ones.
            let Ok(parsed) = serde_json::from_str::<RatedLine>(line) else {
                continue;
            };
            // An unstamped generated line is backlog: it predates the stamp, so
            // there is no round it can be attributed to and no way to tell it from
            // one written six rounds ago.
            if let Some((want_cycle, want_round)) = only {
                let mine =
                    parsed.id_cycle == Some(want_cycle) && parsed.round == Some(want_round);
                if !human && !mine {
                    continue;
                }
            }
            lines.push(parsed);
        }
    }
    Ok(lines)
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
pub fn build_kto_dataset(
    base_model_dir: &Path,
    catalogue_round: Option<(Uuid, i32)>,
) -> io::Result<KtoDatasetStats> {
    let merges_path = base_model_dir.join("merges.bin");
    let merges = tokenize::bpe::load_merges(&merges_path.to_string_lossy())?;

    // The *training* slice only. The held-out slice lives under `eval/preference/`
    // and is the benchmark's alone — training on it would make the reported margin
    // meaningless.
    //
    // **`catalogue_round` is the round whose candidates these are, not this run's
    // base.** KTO runs after RAFT, so its base is the model RAFT just produced, which
    // has never generated a catalogue line — keying off the base would select nothing.
    // The cycle passes its own round, which is the same set RAFT trained on and is
    // what makes the two stages compose.
    let lines = read_catalogue(catalogue_round)?;

    if lines.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no rated replies found — collect some thumbs up/down in the chat first",
        ));
    }

    let model = llm::training::transformer::Model::load(&base_model_dir.to_string_lossy())?;
    let ctx = model.ctx_window();
    // One session for every reference log-probability in this pass, rather than
    // re-uploading the weights per scored example.
    let mut session = llm::training::infer_session::InferSession::new(&model, ctx);

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
        // `Keep` is the identity here: these are the model's *own* generations,
        // produced unbudgeted (see `labeling::autolabel`), so there is no thinking
        // block to strip — and a model trained under `Strip` does not emit one.
        let (ids, m) = llm::chat_template::build_training_example(
            None,
            &turns,
            None,
            &merges,
            llm::chat_template::Reasoning::Keep,
        );

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
        let Some(ref_logprob) = llm::kto::reference_logprob(&mut session, &ids, &m) else {
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
        special_ids_seen: markers_after_training(base_model_dir, &corpus),
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

    let kto_corpus_path = datasets.join("kto_corpus.bin");
    let warm_start_ids = untrained_marker_ids(&base_model_dir, &kto_corpus_path)?;

    llm::train::kto_train(
        &base_model_dir,
        &kto_corpus_path,
        &datasets.join("kto_mask.bin"),
        &datasets.join("kto_index.json"),
        &out_model_dir,
        cfg.epochs,
        &warm_start_ids,
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

    fn sampled_per_dataset(files: &[TaggedFile], budget: usize) -> BTreeMap<String, usize> {
        let mut totals: BTreeMap<String, usize> = BTreeMap::new();
        for slice in plan_bpe_sample(files, budget) {
            *totals.entry(files[slice.file].dataset.clone()).or_default() += slice.bytes;
        }
        totals
    }

    // The reason the sample is stratified at all: proportional shares would give a
    // huge dataset essentially the whole budget and leave a small one with
    // kilobytes, so the small dataset's stage would tokenize at near byte level.
    #[test]
    fn a_small_dataset_is_not_drowned_out_by_a_huge_one() {
        let files = vec![tagged("tiny", 8 << 20), tagged("huge", 1600 << 20)];
        let totals = sampled_per_dataset(&files, 16 << 20);

        assert_eq!(totals["tiny"], 8 << 20, "the small dataset must contribute all it has");
        assert_eq!(totals["huge"], 8 << 20, "the large one takes only the rest");
    }

    // A dataset too small to fill its share hands the remainder back rather than
    // shrinking the sample — the water-filling half of the same property.
    #[test]
    fn an_undersized_dataset_donates_its_leftover_share() {
        let files = vec![tagged("tiny", 1 << 20), tagged("big", 100 << 20), tagged("mid", 20 << 20)];
        let totals = sampled_per_dataset(&files, 30 << 20);

        assert_eq!(totals.values().sum::<usize>(), 30 << 20, "budget should be spent");
        assert_eq!(totals["tiny"], 1 << 20);
    }

    #[test]
    fn a_sample_never_exceeds_the_budget_or_the_corpus() {
        // Everything available is under budget: take it all, no padding.
        let files = vec![tagged("a", 1000), tagged("b", 1000)];
        let totals = sampled_per_dataset(&files, 16 << 20);
        assert_eq!(totals.values().sum::<usize>(), 2000);

        // One dataset, budget smaller than it: capped at the budget.
        let files = vec![tagged("only", 500 << 20)];
        let totals = sampled_per_dataset(&files, 4 << 20);
        assert_eq!(totals["only"], 4 << 20);
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

    // The invariant a reordered epoch depends on: every document's span must name
    // exactly its own tokens, with one separator between neighbours and nothing
    // unaccounted for. A drift here silently trains across document boundaries.
    #[test]
    fn document_spans_tile_the_streamed_corpus() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("build_spans_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let stages = stage_over(
            &tmp,
            "Prose",
            &[
                ("a.txt", "the cat sat on the mat\nand then it slept\n"),
                ("b.txt", "dogs bark loudly in the yard\nevery single morning\n"),
                ("c.txt", "birds sing at dawn\nover the quiet fields\n"),
            ],
        );

        let stats = build_dataset(512, &stages, None, |_| {}).unwrap().expect("not cancelled");
        assert_eq!(stats.files, 3);
        assert_eq!(stats.documents, 3);

        let corpus =
            tokenize::bpe::load_corpus(&storage::datasets_dir().join("corpus.bin").to_string_lossy())
                .unwrap();
        assert_eq!(corpus.len(), stats.corpus_len);

        let segments = load_segments(&storage::datasets_dir().join("segments.json"))
            .unwrap()
            .expect("build must write segments.json");
        assert_eq!(segments.documents.len(), 3);

        let mut previous: Option<&DocSpan> = None;
        for doc in &segments.documents {
            if let Some(prev) = previous {
                assert_eq!(
                    doc.start,
                    prev.start + prev.len + 1,
                    "exactly one separator between documents",
                );
                assert_eq!(
                    u32::from(corpus[prev.start + prev.len]),
                    segments.eot_id,
                    "the token between spans must be the separator",
                );
            } else {
                assert_eq!(doc.start, 0, "the first document starts at zero");
            }
            previous = Some(doc);
        }
        let last = previous.unwrap();
        assert_eq!(last.start + last.len, corpus.len(), "spans must reach the end");

        fs::remove_dir_all(&tmp).ok();
    }

    // Streaming must produce exactly what encoding each file in one shot produces.
    // This is what proves chunked reads and the reused encoder changed nothing —
    // the failure it guards against is a corpus that trains but is subtly wrong.
    #[test]
    fn streamed_documents_match_a_single_shot_encode() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("build_encode_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        // Enough repetition that BPE learns real merges, and several lines per file
        // so the streaming path actually crosses chunk boundaries.
        let a: String = (0..200).map(|i| format!("the quick brown fox {i} jumps\n")).collect();
        let b: String = (0..200).map(|i| format!("a lazy dog {i} sleeps soundly\n")).collect();
        let stages = stage_over(&tmp, "Prose", &[("a.txt", a.as_str()), ("b.txt", b.as_str())]);

        build_dataset(600, &stages, None, |_| {}).unwrap().expect("not cancelled");

        let corpus =
            tokenize::bpe::load_corpus(&storage::datasets_dir().join("corpus.bin").to_string_lossy())
                .unwrap();
        let merges = tokenize::bpe::load_merges(
            &storage::tokenizer_dir().join("merges.bin").to_string_lossy(),
        )
        .unwrap();
        let segments = load_segments(&storage::datasets_dir().join("segments.json"))
            .unwrap()
            .unwrap();
        let encoder = tokenize::bpe::Encoder::new(&merges);

        for (doc, text) in segments.documents.iter().zip([&a, &b]) {
            let streamed: Vec<u32> = corpus[doc.start..doc.start + doc.len]
                .iter()
                .map(|&t| u32::from(t))
                .collect();
            assert_eq!(streamed, encoder.encode(text), "streamed corpus differs from one-shot");
        }

        fs::remove_dir_all(&tmp).ok();
    }

    // A cancelled build must leave the previous dataset exactly as it was — a
    // half-written corpus that a later run consumes is worse than no build at all.
    #[test]
    fn a_cancelled_build_leaves_the_previous_dataset_untouched() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("build_cancel_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let stages = stage_over(&tmp, "Prose", &[("a.txt", "some words to tokenize\n")]);

        let datasets = storage::datasets_dir();
        let tokenizer = storage::tokenizer_dir();
        fs::create_dir_all(&datasets).unwrap();
        fs::create_dir_all(&tokenizer).unwrap();
        fs::write(datasets.join("corpus.bin"), b"previous corpus").unwrap();
        fs::write(datasets.join("segments.json"), b"previous segments").unwrap();
        fs::write(tokenizer.join("merges.bin"), b"previous merges").unwrap();

        let cancel = Arc::new(AtomicBool::new(true));
        let outcome = build_dataset(512, &stages, Some(cancel), |_| {}).unwrap();

        assert!(outcome.is_none(), "a pre-cancelled build must report cancellation");
        assert_eq!(fs::read(datasets.join("corpus.bin")).unwrap(), b"previous corpus");
        assert_eq!(fs::read(datasets.join("segments.json")).unwrap(), b"previous segments");
        assert_eq!(fs::read(tokenizer.join("merges.bin")).unwrap(), b"previous merges");
        assert!(!datasets.join("corpus.bin.partial").exists(), "partial left behind");

        fs::remove_dir_all(&tmp).ok();
    }

    // Runs the real pipeline over the real uploads directory, to measure rather
    // than estimate. Ignored because it depends on this machine's data and takes
    // minutes; run it deliberately, under a memory monitor:
    //
    //   /usr/bin/time -v cargo test -p backend --bins -- --ignored --nocapture \
    //       measure_a_real_build
    //
    // The uploads are symlinked, never copied or written to, and everything the
    // build produces lands in a temp root — the live dataset is not touched.
    #[test]
    #[ignore]
    fn measure_a_real_build() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Point at another tree to measure a corpus this machine does not hold —
        // the flat-memory claim is about scale, and the real uploads are small.
        let real_uploads = std::env::var("MEASURE_UPLOADS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data/raw/uploads"));
        if !real_uploads.is_dir() {
            eprintln!("no data/raw/uploads here — skipping");
            return;
        }

        let tmp = std::env::temp_dir().join(format!("build_real_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let uploads = tmp.join("raw").join("uploads");
        fs::create_dir_all(&uploads).unwrap();

        let mut datasets = Vec::new();
        for entry in fs::read_dir(&real_uploads).unwrap().flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            std::os::unix::fs::symlink(entry.path().canonicalize().unwrap(), uploads.join(&name))
                .unwrap();
            datasets.push(DatasetSelection { name, count: 0 });
        }
        eprintln!("datasets: {:?}", datasets.iter().map(|d| &d.name).collect::<Vec<_>>());

        // SAFETY: single-threaded under ENV_LOCK; DATA_DIR drives storage::root().
        unsafe { std::env::set_var("DATA_DIR", &tmp) };
        let stages = vec![CurriculumStage { name: None, datasets, epochs: 1 }];

        let started = std::time::Instant::now();
        let stats = build_dataset(8192, &stages, None, |event| {
            if let TokenizeEvent::Encode { files_done, files, chars } = event {
                eprintln!("  encode {files_done}/{files} files, {chars} chars");
            }
        })
        .unwrap()
        .expect("not cancelled");

        eprintln!(
            "files={} chars={} tokens={} docs={} vocab={} merges={} in {:.1}s",
            stats.files,
            stats.chars,
            stats.corpus_len,
            stats.documents,
            stats.vocab_size,
            stats.merges,
            started.elapsed().as_secs_f64(),
        );
        let bytes = fs::metadata(storage::datasets_dir().join("corpus.bin")).unwrap().len();
        eprintln!("corpus.bin = {bytes} bytes for {} tokens", stats.corpus_len);
        assert_eq!(bytes as usize, CORPUS_HEADER_BYTES_FOR_TEST + stats.corpus_len * 2);

        fs::remove_dir_all(&tmp).ok();
    }

    // Mirrors the tokenizer's private header size; asserted against here so a
    // change to the format shows up as a failing measurement rather than silently.
    const CORPUS_HEADER_BYTES_FOR_TEST: usize = 8;

    // The point of rendering chat documents inside the build: a run picks up
    // conversations added since the last one, with no export step in between.
    // Before this, `uploads/Chat` was written by a separate endpoint and a run
    // that skipped it trained on whatever was left from last time.
    #[test]
    fn a_run_renders_the_chat_documents_it_trains_on() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("build_chat_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        // A prose dataset so the tokenizer has something to learn from, plus the
        // virtual Chat dataset, which no folder under uploads/ provides.
        let mut stages = stage_over(&tmp, "Prose", &[("a.txt", "the cat sat on the mat\n")]);
        stages[0].datasets.push(DatasetSelection { name: CHAT_DATASET.to_string(), count: 0 });

        let sft = tmp.join("raw").join("sft");
        fs::create_dir_all(&sft).unwrap();
        // Enough conversations that some land in the pretrain share whatever the
        // shares are set to.
        let lines: String = (0..60)
            .map(|i| {
                format!(
                    "{{\"turns\":[{{\"role\":\"user\",\"text\":\"q{i}\"}},\
                     {{\"role\":\"assistant\",\"text\":\"answer number {i}\"}}]}}\n"
                )
            })
            .collect();
        fs::write(sft.join("conversations.jsonl"), &lines).unwrap();

        let expected = count_chat_documents().unwrap();
        assert!(expected > 0, "fixture must put some conversations in the pretrain share");
        // Offered in the picker without any folder existing for it.
        assert!(
            list_datasets().unwrap().iter().any(|d| d.name == CHAT_DATASET && d.files == expected),
            "Chat should be listed as a dataset with its pretrain-share count",
        );

        let stats = build_dataset(512, &stages, None, |_| {}).unwrap().expect("not cancelled");
        assert_eq!(stats.files, 1 + expected, "the run must include the chat documents");

        // The scaffold markers reach the corpus, which is the whole reason the chat
        // data is seeded into pretraining — SFT must not be their first sighting.
        let assistant = tokenize::bpe::encode("<|assistant|>", &[])[0];
        assert!(
            stats.special_ids_seen.contains(&assistant),
            "chat markers missing from the corpus: {:?}",
            stats.special_ids_seen,
        );

        // And a conversation added afterwards is picked up by the next run with no
        // export in between.
        fs::write(
            sft.join("more.jsonl"),
            "{\"turns\":[{\"role\":\"user\",\"text\":\"late arrival\"},\
             {\"role\":\"assistant\",\"text\":\"a reply that came later\"}]}\n",
        )
        .unwrap();
        let grown = count_chat_documents().unwrap();
        let stats = build_dataset(512, &stages, None, |_| {}).unwrap().unwrap();
        assert_eq!(stats.files, 1 + grown, "a later run must see conversations added since");

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn an_empty_selection_is_an_error_rather_than_an_empty_corpus() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("build_empty_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let stages = stage_over(&tmp, "Prose", &[]);

        let err = build_dataset(512, &stages, None, |_| {}).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);

        fs::remove_dir_all(&tmp).ok();
    }

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
        for budget in llm::chat_template::ThinkBudget::ALL {
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
        assert_eq!(conv.budget, Some(llm::chat_template::ThinkBudget::Med));

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

    // `DATA_DIR` is a process-global that `storage::root()` reads, so tests that
    // set it must not run concurrently. The lock lives in `storage` because it has
    // to be shared with every other module's tests — this one and `synth`'s used
    // to be separate locks over the same variable, which is no lock at all.
    // (`.lock()` recovers from a poisoned guard so one failing test doesn't cascade.)
    use storage::ENV_LOCK;

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

        // Two conversations, one per line — chosen so both land in the SFT share,
        // since `build_sft_dataset` now trains on that share alone. Picking them by
        // asking `chat_destination` keeps the fixture valid if the shares change.
        let (q1, a1) = sft_share_conversation(None, 0);
        let (q2, a2) = sft_share_conversation(Some("be nice"), 0);
        let jsonl = format!(
            "{{\"turns\":[{{\"role\":\"user\",\"text\":\"{q1}\"}},{{\"role\":\"assistant\",\"text\":\"{a1}\"}}]}}\n\
             {{\"system\":\"be nice\",\"turns\":[{{\"role\":\"user\",\"text\":\"{q2}\"}},{{\"role\":\"assistant\",\"text\":\"{a2}\"}}]}}\n"
        );
        fs::write(sft.join("data.jsonl"), jsonl).unwrap();

        let stats = build_sft_dataset(&base, llm::chat_template::Reasoning::Keep).unwrap();
        assert_eq!(stats.conversations, 2);
        // No pretrain corpus exists under this temp root, so the tokenizer guard
        // refuses replay and the corpus is conversations only — which is what the
        // token arithmetic below assumes.
        assert_eq!(stats.replay_tokens, 0, "replay must be skipped without a corpus");

        let corpus = tokenize::bpe::load_corpus(
            &storage::datasets_dir().join("sft_corpus.bin").to_string_lossy(),
        )
        .unwrap();
        let mask_bytes = fs::read(storage::datasets_dir().join("sft_mask.bin")).unwrap();
        // One mask byte per token, and the mask is what the training loop indexes
        // with — a drift here misaligns every chunk's mask slice.
        assert_eq!(corpus.len(), mask_bytes.len(), "corpus/mask must align");
        assert_eq!(stats.total_tokens, mask_bytes.len());

        // Trainable tokens = the two assistant replies + their closing markers.
        let expected = llm::chat_template::build_training_example(
            None,
            &[
                llm::chat_template::Turn { role: llm::chat_template::Role::User, text: q1.clone() },
                llm::chat_template::Turn { role: llm::chat_template::Role::Assistant, text: a1.clone() },
            ],
            None,
            &merges,
            llm::chat_template::Reasoning::Keep,
        ).1.iter().filter(|&&b| b != 0).count()
        + llm::chat_template::build_training_example(
            Some("be nice"),
            &[
                llm::chat_template::Turn { role: llm::chat_template::Role::User, text: q2.clone() },
                llm::chat_template::Turn { role: llm::chat_template::Role::Assistant, text: a2.clone() },
            ],
            None,
            &merges,
            llm::chat_template::Reasoning::Keep,
        ).1.iter().filter(|&&b| b != 0).count();
        assert_eq!(stats.trainable_tokens, expected);
        assert!(stats.trainable_tokens < stats.total_tokens);

        // The sidecar the per-epoch shuffle reads: one span per conversation,
        // separated by exactly one <|endoftext|>, together covering the corpus. If
        // the spans and the baked-in separators disagree, a reordered epoch either
        // drops tokens or reads across a boundary — so this is the invariant that
        // matters, not the span count.
        let segments = load_sft_segments(&storage::datasets_dir().join(SFT_SEGMENTS_FILE))
            .unwrap()
            .expect("build must write the conversation spans");
        assert_eq!(segments.conversations.len(), 2);
        let first = segments.conversations[0];
        let second = segments.conversations[1];
        assert_eq!(first.start, 0);
        assert_eq!(second.start, first.len + 1, "exactly one separator between spans");
        assert_eq!(second.start + second.len, stats.total_tokens);

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
        // The same back-compat rule for `strip_thinking`: absent means the runs
        // already on disk, which kept their reasoning.
        assert!(!cfg.strip_thinking);
    }

    // The policy, in one place: a pretrain corpus never carries reasoning whatever
    // the request said, and SFT does exactly what it was asked.
    #[test]
    fn pretrain_always_strips_and_sft_obeys_the_request() {
        use llm::chat_template::Reasoning;
        for requested in [false, true] {
            assert_eq!(
                strip_thinking_for(TrainMode::Pretrain, requested),
                Reasoning::Strip,
                "pretrain kept reasoning when asked {requested}"
            );
        }
        assert_eq!(strip_thinking_for(TrainMode::Sft, false), Reasoning::Keep);
        assert_eq!(strip_thinking_for(TrainMode::Sft, true), Reasoning::Strip);
    }

    // A conversation left with nothing to say once stripped would teach "answer
    // with nothing and stop", so it is dropped rather than trained on.
    #[test]
    fn a_reply_that_is_only_reasoning_does_not_survive_stripping() {
        use llm::chat_template::{Reasoning, Role, Turn};
        let only_thinking = [
            Turn { role: Role::User, text: "q".into() },
            Turn {
                role: Role::Assistant,
                text: "<|thinking|>all of it<|/thinking|>".into(),
            },
        ];
        assert!(!any_reply_survives(&only_thinking, Reasoning::Strip));
        // Nothing is removed under `Keep`, so the same conversation is fine there.
        assert!(any_reply_survives(&only_thinking, Reasoning::Keep));

        let has_answer = [
            Turn { role: Role::User, text: "q".into() },
            Turn { role: Role::Assistant, text: "<|thinking|>hmm<|/thinking|>a".into() },
        ];
        assert!(any_reply_survives(&has_answer, Reasoning::Strip));
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
            let (ids, mask) = llm::chat_template::build_training_example(
                conv.system.as_deref(),
                &turns,
                conv.budget,
                &merges,
                llm::chat_template::Reasoning::Keep,
            );
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

#[cfg(test)]
mod chat_eval_export_tests {
    use super::*;

    // Writes `conversations` into a temp DATA_DIR's raw/sft and returns the root.
    // Caller must hold ENV_LOCK.
    fn sft_root(tag: &str, lines: &[&str]) -> PathBuf {
        let tmp = std::env::temp_dir().join(format!("chat_eval_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let sft = tmp.join("raw").join("sft");
        fs::create_dir_all(&sft).unwrap();
        fs::write(sft.join("a.jsonl"), lines.join("\n")).unwrap();
        // SAFETY: single-threaded under ENV_LOCK; DATA_DIR drives storage::root().
        unsafe { std::env::set_var("DATA_DIR", &tmp) };
        tmp
    }

    fn conversation(n: usize) -> String {
        format!(
            r#"{{"turns":[{{"role":"user","text":"q{n}"}},{{"role":"assistant","text":"a{n}"}}]}}"#
        )
    }

    // The whole point: the held-out share reaches disk where `eval::chat` reads it.
    // It never did before — `ChatDestination::Eval` was computed and dropped, so
    // every benchmark report carried `"chat": null` and the one instrument built to
    // show a fine-tune working never ran.
    #[test]
    fn the_held_out_share_is_written_where_the_benchmark_reads_it() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Enough conversations that the 15% share is very unlikely to be empty.
        let lines: Vec<String> = (0..200).map(conversation).collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let tmp = sft_root("written", &refs);

        let written = export_chat_eval_set().unwrap();
        assert!(written > 0, "no conversations were held out of 200");

        let path = tmp.join("eval").join("chat").join("heldout.jsonl");
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), written);

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn it_writes_only_the_eval_share() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let lines: Vec<String> = (0..200).map(conversation).collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let tmp = sft_root("share", &refs);

        export_chat_eval_set().unwrap();
        let text =
            fs::read_to_string(tmp.join("eval").join("chat").join("heldout.jsonl")).unwrap();

        // Every exported conversation must be one `chat_destination` put in Eval —
        // a leak the other way would put training data in the held-out set, which is
        // the failure a held-out set exists to prevent.
        for line in text.lines() {
            let parsed: SftConversation = serde_json::from_str(line).unwrap();
            assert_eq!(
                chat_destination(&parsed),
                ChatDestination::Eval,
                "exported a conversation that belongs in training: {line}",
            );
        }

        fs::remove_dir_all(&tmp).ok();
    }

    // The split is a content hash, so a re-export must not move anything. If it did,
    // two benchmark runs of the same model would score different text.
    #[test]
    fn the_export_is_idempotent() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let lines: Vec<String> = (0..120).map(conversation).collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let tmp = sft_root("idempotent", &refs);
        let path = tmp.join("eval").join("chat").join("heldout.jsonl");

        assert_eq!(export_chat_eval_set().unwrap(), export_chat_eval_set().unwrap());
        let first = fs::read_to_string(&path).unwrap();
        export_chat_eval_set().unwrap();
        assert_eq!(first, fs::read_to_string(&path).unwrap());

        fs::remove_dir_all(&tmp).ok();
    }

    // Round-trips through the reader that actually consumes it. A shape mismatch
    // here would be invisible until a benchmark silently scored zero conversations.
    #[test]
    fn what_is_written_is_what_the_benchmark_can_read() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let lines: Vec<String> = (0..200).map(conversation).collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let tmp = sft_root("roundtrip", &refs);

        let written = export_chat_eval_set().unwrap();
        let loaded = llm::eval::chat::load(&tmp.join("eval").join("chat"));
        assert_eq!(loaded.len(), written);
        assert_eq!(loaded[0].turns.len(), 2);
        assert_eq!(loaded[0].turns[0].role, "user");

        fs::remove_dir_all(&tmp).ok();
    }

    // A system prompt must survive: it is part of what the fine-tune conditioned on,
    // and dropping it would make the held-out score measure a different task.
    #[test]
    fn a_system_prompt_survives_the_round_trip() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Hunt for a conversation the hash puts in Eval, so the assertion is about
        // the round trip rather than about which bucket this one landed in.
        let mut lines = Vec::new();
        for n in 0..400 {
            lines.push(format!(
                r#"{{"system":"be terse","turns":[{{"role":"user","text":"q{n}"}},{{"role":"assistant","text":"a{n}"}}]}}"#
            ));
        }
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let tmp = sft_root("system", &refs);

        export_chat_eval_set().unwrap();
        let loaded = llm::eval::chat::load(&tmp.join("eval").join("chat"));
        assert!(!loaded.is_empty());
        assert_eq!(loaded[0].system.as_deref(), Some("be terse"));

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_missing_sft_directory_exports_nothing_rather_than_failing() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("chat_eval_empty_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        // SAFETY: single-threaded under ENV_LOCK.
        unsafe { std::env::set_var("DATA_DIR", &tmp) };

        // A fresh install has no conversations at all; that is not an error, and the
        // markup suite still scores its pair stage.
        assert_eq!(export_chat_eval_set().unwrap(), 0);
        assert!(llm::eval::chat::load(&tmp.join("eval").join("chat")).is_empty());

        fs::remove_dir_all(&tmp).ok();
    }

    // A stale export from a previous set must not linger: the directory is rebuilt,
    // not appended to.
    #[test]
    fn a_previous_export_is_replaced_rather_than_added_to() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let lines: Vec<String> = (0..200).map(conversation).collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let tmp = sft_root("stale", &refs);
        let dir = tmp.join("eval").join("chat");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("old.jsonl"), conversation(9999)).unwrap();

        export_chat_eval_set().unwrap();
        assert!(!dir.join("old.jsonl").exists(), "a stale export survived");

        fs::remove_dir_all(&tmp).ok();
    }
}

#[cfg(test)]
mod catalogue_tests {
    use super::*;
    use std::fs;

    // Writes a preference directory under a temp DATA_DIR and returns its root.
    // `generated` lands in the auto-label file, `human` in the feedback file, and
    // the two are treated differently on purpose. Caller must hold ENV_LOCK.
    fn catalogue_root(
        name: &str,
        generated: &[serde_json::Value],
        human: &[serde_json::Value],
    ) -> PathBuf {
        let tmp = std::env::temp_dir().join(format!("catalogue_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let dir = tmp.join("raw").join("preference");
        fs::create_dir_all(&dir).unwrap();
        let body = |lines: &[serde_json::Value]| -> String {
            lines.iter().map(|l| format!("{l}\n")).collect()
        };
        fs::write(dir.join(storage::AUTOLABEL_FILE), body(generated)).unwrap();
        fs::write(dir.join(storage::HUMAN_FEEDBACK_FILE), body(human)).unwrap();
        // SAFETY: single-threaded under ENV_LOCK; DATA_DIR drives storage::root().
        unsafe { std::env::set_var("DATA_DIR", &tmp) };
        tmp
    }

    // A generated line stamped with the round that wrote it. `id_model` rides along
    // as provenance the way the real writer emits it, and is deliberately *not* what
    // the filter reads — keying on it is the bug this scoping replaced.
    fn stamped(round: (Uuid, i32), response: &str) -> serde_json::Value {
        serde_json::json!({
            "prompt": "why is the sky blue?",
            "response": response,
            "label": "up",
            "id_model": Uuid::new_v4(),
            "id_cycle": round.0,
            "round": round.1,
        })
    }

    fn unstamped(response: &str) -> serde_json::Value {
        serde_json::json!({
            "prompt": "why is the sky blue?",
            "response": response,
            "label": "up",
        })
    }

    // The whole point of the provenance stamp. The catalogue file is append-only
    // and spans every round a cycle has ever run; without this filter round 7's
    // self-train is a max-likelihood pass over rounds 1-7's output, and the
    // earliest and worst generations accumulate the most total gradient.
    #[test]
    fn a_round_reads_only_its_own_generations() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let cycle = Uuid::new_v4();
        let mine = (cycle, 3);
        // The *same cycle*, an earlier round. This is the case keying on the
        // generating model could never see: a re-run generates with the same seed, so
        // model id alone puts both of these in the same bucket.
        let theirs = (cycle, 2);
        let tmp = catalogue_root(
            "scoped",
            &[
                stamped(mine, "this round"),
                stamped(theirs, "an earlier round"),
                stamped(mine, "this round again"),
            ],
            &[],
        );

        let lines = read_catalogue(Some(mine)).unwrap();
        assert_eq!(lines.len(), 2);
        assert!(lines.iter().all(|l| l.response.starts_with("this round")));

        fs::remove_dir_all(&tmp).ok();
    }

    // The exemption that matters, and the one that must not be over-broad: human
    // ratings are the seed and the audit, a few dozen hand-clicked judgements
    // against thousands of generated lines, and they belong in every run.
    #[test]
    fn human_ratings_survive_every_filter() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mine = (Uuid::new_v4(), 1);
        let tmp = catalogue_root(
            "human",
            &[stamped(mine, "generated"), stamped((Uuid::new_v4(), 1), "someone else's")],
            &[unstamped("hand rated")],
        );

        let lines = read_catalogue(Some(mine)).unwrap();
        assert_eq!(lines.len(), 2);
        assert!(lines.iter().any(|l| l.response == "hand rated"));

        fs::remove_dir_all(&tmp).ok();
    }

    // The bug this file had on its first cut: exempting *every* unstamped line
    // rather than every human one. An unstamped line in the auto-label file is a
    // generation from before the stamp existed — exactly the accumulated backlog
    // the scoping exists to stop training on. Grandfathering it in makes the
    // filter inert on any install with a pre-existing catalogue, which is every
    // install that has ever run the loop.
    #[test]
    fn unstamped_generations_are_backlog_and_are_dropped() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mine = (Uuid::new_v4(), 1);
        let tmp = catalogue_root(
            "backlog",
            &[
                stamped(mine, "this round"),
                unstamped("written before the stamp existed"),
                unstamped("and another"),
            ],
            &[],
        );

        let lines = read_catalogue(Some(mine)).unwrap();
        assert_eq!(lines.len(), 1, "legacy catalogue must not ride along");
        assert_eq!(lines[0].response, "this round");

        fs::remove_dir_all(&tmp).ok();
    }

    // A run started by hand from the API has no round to scope to, and must see
    // the file exactly as it did before scoping existed.
    #[test]
    fn no_filter_reads_the_whole_catalogue() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = catalogue_root(
            "whole",
            &[stamped((Uuid::new_v4(), 1), "a"), unstamped("b")],
            &[unstamped("c")],
        );

        assert_eq!(read_catalogue(None).unwrap().len(), 3);

        fs::remove_dir_all(&tmp).ok();
    }

    // A round that generated nothing gets nothing back — it does not silently fall
    // through to the whole file. The empty result is what makes the size gates fire
    // and the round refuse, which is the correct outcome.
    #[test]
    fn a_round_with_no_generations_selects_nothing() {
        let _guard = storage::ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = catalogue_root("empty", &[stamped((Uuid::new_v4(), 1), "someone else's")], &[]);

        assert!(read_catalogue(Some((Uuid::new_v4(), 1))).unwrap().is_empty());

        fs::remove_dir_all(&tmp).ok();
    }
}
