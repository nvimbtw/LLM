//! Rebuilds `eval/chat/heldout.jsonl` from `raw/sft/` and reports the chat split.
//!
//! The queue calls `export_chat_eval_set` before every benchmark, so this is only
//! a way to do it on demand — after replacing the SFT set, when the assistant
//! anchor has to be re-derived against the new holdout before any score is
//! recorded against it.
//!
//! ```text
//! cargo test -p training --test regenerate_chat_eval -- --ignored --nocapture
//! ```

#[test]
#[ignore = "rewrites eval/chat/heldout.jsonl from the installed SFT set"]
fn regenerate_and_report() {
    let written = training::export_chat_eval_set().expect("export failed");
    eprintln!("held-out conversations written: {written}");

    let path = storage::eval_dir().join("chat").join("heldout.jsonl");
    let text = std::fs::read_to_string(&path).expect("heldout.jsonl");
    let lines = text.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(lines, written, "the file disagrees with the reported count");
    eprintln!("{} bytes at {}", text.len(), path.display());
}
