//! Builds the real SFT dataset from the installed `raw/sft/` and reports it.
//!
//! Phase 1 of a training run, on demand. Worth having as a probe because the set
//! went from ~1,900 conversations to ~454,000 in one step, and everything about
//! that phase — the whole corpus held in memory, the segment sidecar, the prose
//! replay sized against the trainable count — was written when it was the former.
//!
//! ```text
//! DATA_DIR=$PWD/data MODEL_DIR=$PWD/data/artifacts/models/<id> \
//!   cargo test --release -p training --test build_sft_at_scale -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

#[test]
#[ignore = "rewrites datasets/sft_*.bin from the installed SFT set"]
fn build_and_report() {
    let base = PathBuf::from(std::env::var("MODEL_DIR").expect("MODEL_DIR"));
    let started = Instant::now();
    let stats = training::build_sft_dataset(&base, llm::chat_template::Reasoning::Keep)
        .expect("build_sft_dataset failed");

    eprintln!("conversations     {}", stats.conversations);
    eprintln!("skipped too long  {}", stats.skipped_too_long);
    eprintln!("total tokens      {}", stats.total_tokens);
    eprintln!("trainable tokens  {}", stats.trainable_tokens);
    eprintln!("replay tokens     {}", stats.replay_tokens);
    eprintln!(
        "trainable share   {:.1}%",
        100.0 * stats.trainable_tokens as f64 / stats.total_tokens.max(1) as f64
    );
    eprintln!("markers seen      {}", stats.special_ids_seen.len());
    eprintln!("{:.1}s", started.elapsed().as_secs_f64());

    assert!(stats.conversations > 0, "no conversations survived the SFT share");
    // Replay of 0 means the tokenizer guard refused it, which turns this into the
    // unanchored fine-tune that replay exists to prevent.
    assert!(stats.replay_tokens > 0, "prose replay was refused");

    // Replay has to actually reach its budget, not merely be non-zero. It is what
    // holds the fine-tune to the pretraining distribution, and a single pass over
    // the documents caps out at ~7M tokens — 14% of what an SFT set this size asks
    // for, which would leave the run anchored at a seventh of the intended strength
    // with nothing but this number to say so.
    let chat_trainable = stats.trainable_tokens - stats.replay_tokens;
    assert!(
        stats.replay_tokens * 2 > chat_trainable,
        "replay {} against {chat_trainable} trainable chat tokens: the fine-tune is \
         barely anchored to prose",
        stats.replay_tokens,
    );
}
