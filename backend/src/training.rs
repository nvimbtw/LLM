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
    // How many uploaded books to train on, chosen at random. 0 (or >= the number
    // available) means "use all of them". Lets the user keep a big pile of data
    // in uploads and dial the training set size per run.
    #[serde(default)]
    pub num_books: usize,
    // How many synthetic documents (from the synthetic dir) to mix in, chosen at
    // random. 0 means none — synthetic data is opt-in per run.
    #[serde(default)]
    pub num_synthetic: usize,
    // The model's id, which names its on-disk directory. Set server-side by the
    // start route (reused on retrain, freshly generated for a new model), never
    // by the client — hence skipped on deserialize.
    #[serde(skip_deserializing)]
    pub storage_id: Uuid,
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
// Fisher–Yates shuffle with a time-seeded xorshift PRNG. Avoids pulling in the
// `rand` crate for the one spot we need randomness (picking a book subset) — the
// bar here is "different each run", not cryptographic quality.
fn shuffle<T>(items: &mut [T]) {
    let mut state = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
        | 1; // xorshift must never start at zero
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in (1..items.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        items.swap(i, j);
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

// writes the two files `llm::train` consumes: a flat token-id `corpus.bin` in
// `datasets/` and `merges.bin` in `artifacts/tokenizer/`. Progress is pushed
// through `on_event` (synchronous; runs on the caller's thread).
pub fn build_dataset(
    max_vocab: u32,
    num_books: usize,
    num_synthetic: usize,
    on_event: impl Fn(TokenizeEvent),
) -> io::Result<DatasetStats> {
    // Books from uploads (0 => all) plus a chosen number of synthetic docs.
    let mut paths = select_txt(&storage::uploads_dir(), num_books, true)?;
    paths.extend(select_txt(&storage::synthetic_dir(), num_synthetic, false)?);

    if paths.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no .txt files found in uploads or synthetic",
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
    let model_dir = storage::models_dir().join(cfg.storage_id.to_string());
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
//
// These items are exercised by tests now and wired into the run orchestration +
// route by ISSUE-0037; allow them to be otherwise-unused until then (same
// "defined ahead of its caller" pattern as the path helpers in storage.rs).

// One turn of a conversation, as it appears in the input JSONL.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct SftTurn {
    role: String,
    text: String,
}

// One conversation: an optional system prompt plus alternating turns, ending on
// an assistant reply. One JSON object per line in the input `.jsonl` files.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct SftConversation {
    #[serde(default)]
    system: Option<String>,
    turns: Vec<SftTurn>,
}

// Summary of what the SFT build produced, for the dashboard + meta.json. The
// trainable fraction is a useful sanity check — it should sit well under 50%
// (most tokens are system/user context the model only conditions on).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SftDatasetStats {
    pub conversations: usize,
    pub total_tokens: usize,
    pub trainable_tokens: usize,
}

// Map the JSONL role string onto the chat-template Role. Unknown roles are a
// hard error — bad training data should fail loudly, not train on the wrong mask.
#[allow(dead_code)]
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
#[allow(dead_code)]
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

#[cfg(test)]
mod sft_tests {
    use super::*;

    // Build a dataset from a fixture SFT dir + a fixture merges.bin (empty merges
    // → byte-level tokenization, deterministic), and check the corpus/mask are
    // aligned and the trained-token count matches the assistant spans.
    #[test]
    fn test_build_sft_dataset_aligns_and_counts() {
        // Isolate storage + a base-model dir under a unique temp root.
        let tmp = std::env::temp_dir().join(format!("sft_test_{}", std::process::id()));
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

    #[test]
    fn test_build_sft_dataset_rejects_unknown_role() {
        assert!(parse_role("system").is_err());
        assert!(parse_role("user").is_ok());
        assert!(parse_role("assistant").is_ok());
    }
}
