//! Every conversation in `raw/sft/` must render inside the model's context window.
//!
//! `build_sft_dataset` now skips one that does not, reporting `skipped_too_long`
//! the way KTO and RAFT always have — so this is not a correctness backstop. It is
//! a *tuning* check: the budget in `scripts/prepare_smoltalk.py` is expressed in
//! characters because the tokenizer is Rust, and this is what says whether that
//! character budget is close enough. A few tenths of a percent overrunning is the
//! expected cost of a coarse pre-filter; a large fraction means the import is
//! mis-tuned for this model's window and is quietly throwing away most of itself.
//!
//! ```text
//! MODEL_DIR=$PWD/data/artifacts/models/<id> \
//!   cargo test -p llm --test sft_fits_the_window -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use llm::chat_template::{self, Reasoning, Role, Turn};

// Matches `training::MAX_USER_TURN_BYTES`: a user turn is capped before rendering,
// so measuring the raw text would over-report what training actually sees. The
// *tail* is kept, as it is there — a long turn is pasted context followed by the
// actual request, and the request is what the reply answers.
const MAX_USER_TURN_BYTES: usize = 4000;

fn cap_user_turn(text: &str) -> String {
    if text.len() <= MAX_USER_TURN_BYTES {
        return text.to_string();
    }
    let mut start = text.len() - MAX_USER_TURN_BYTES;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_string()
}

#[derive(serde::Deserialize)]
struct Conversation {
    #[serde(default)]
    system: Option<String>,
    turns: Vec<ConversationTurn>,
    #[serde(default)]
    budget: Option<chat_template::ThinkBudget>,
}

#[derive(serde::Deserialize)]
struct ConversationTurn {
    role: String,
    text: String,
}

#[test]
#[ignore = "needs an installed model and SFT set: set MODEL_DIR"]
fn every_sft_conversation_fits_the_context_window() {
    let model_dir = PathBuf::from(std::env::var("MODEL_DIR").expect("MODEL_DIR"));
    let merges = tokenize::bpe::load_merges(&model_dir.join("merges.bin").to_string_lossy())
        .expect("merges.bin");
    let ctx = ctx_window(&model_dir);

    let sft_dir = model_dir
        .ancestors()
        .nth(3)
        .expect("models dir layout")
        .join("raw/sft");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&sft_dir)
        .unwrap_or_else(|e| panic!("{}: {e}", sft_dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    paths.sort();

    let (mut total, mut over, mut tokens, mut bytes) = (0usize, 0usize, 0usize, 0usize);
    let mut longest = 0usize;
    let mut worst = String::new();

    for path in &paths {
        let text = std::fs::read_to_string(path).expect("read");
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(conversation) = serde_json::from_str::<Conversation>(line) else {
                panic!("unparseable line in {}", path.display());
            };
            let turns: Vec<Turn> = conversation
                .turns
                .iter()
                .map(|t| Turn {
                    role: match t.role.as_str() {
                        "user" => Role::User,
                        "assistant" => Role::Assistant,
                        other => panic!("unknown role {other:?} in {}", path.display()),
                    },
                    text: t.text.clone(),
                })
                .map(|mut t| {
                    if t.role == Role::User {
                        t.text = cap_user_turn(&t.text);
                    }
                    t
                })
                .collect();

            let (ids, _) = chat_template::build_training_example(
                conversation.system.as_deref(),
                &turns,
                conversation.budget,
                &merges,
                Reasoning::Keep,
            );
            total += 1;
            tokens += ids.len();
            bytes += conversation.turns.iter().map(|t| t.text.len()).sum::<usize>();
            if ids.len() > longest {
                longest = ids.len();
                worst = path.file_name().unwrap_or_default().to_string_lossy().into();
            }
            if ids.len() > ctx {
                over += 1;
            }
        }
    }

    assert!(total > 0, "no conversations found under {}", sft_dir.display());
    let wasted = 100.0 * over as f64 / total as f64;
    eprintln!("{total} conversations across {} file(s)", paths.len());
    eprintln!("  ctx {ctx}, longest {longest} tokens (in {worst})");
    eprintln!("  mean {:.0} tokens", tokens as f64 / total as f64);
    eprintln!("  {:.2} bytes/token over the reply text", bytes as f64 / tokens as f64);
    eprintln!("  {over} ({wasted:.2}%) would be skipped as too long");
    assert!(
        wasted < MAX_WASTED_PERCENT,
        "{over} of {total} conversations ({wasted:.1}%) render past the {ctx}-token \
         window; the character budget in scripts/prepare_smoltalk.py is too loose",
    );
}

// How much of the imported set may overrun before the character budget that
// produced it counts as mis-tuned. Generous against the measured 0.2%, because the
// number moves with the corpus mix rather than with anything under this repo's
// control — code-heavy conversations tokenize at far fewer bytes per token than the
// prose the vocabulary was trained on, and they are what overruns.
const MAX_WASTED_PERCENT: f64 = 2.0;

// The window the model was built with, off its own meta.json rather than assumed:
// the budget this verifies is only meaningful against the model that will train on it.
fn ctx_window(model_dir: &std::path::Path) -> usize {
    let raw = std::fs::read_to_string(model_dir.join("meta.json")).expect("meta.json");
    let meta: serde_json::Value = serde_json::from_str(&raw).expect("meta.json is json");
    meta.get("config")
        .and_then(|c| c.get("ctx_window"))
        .and_then(serde_json::Value::as_u64)
        .expect("config.ctx_window") as usize
}
