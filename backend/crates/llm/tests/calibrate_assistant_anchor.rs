//! Re-derives `eval::CHAT_BPB_MID` from the models on disk.
//!
//! Not a test of behaviour — it asserts nothing and is `#[ignore]`d. It exists
//! because the anchor is a *measurement*: it should sit at the middle of the cohort
//! so the logistic is steepest where models actually are, and version 1 of this
//! suite died precisely by keeping an anchor after the thing it was measured
//! against had moved. Re-run it over every model whenever the held-out set or the
//! tokenizer changes, and move the constant to the new cohort mean.
//!
//! ```text
//! for id in $(ls data/artifacts/models); do
//!   MODEL_DIR=$PWD/data/artifacts/models/$id EVAL_DIR=$PWD/data/eval \
//!     cargo test -p llm --test calibrate_assistant_anchor -- --ignored --nocapture --test-threads=1
//! done
//! ```
//!
//! `--test-threads=1` is mandatory: the GPU scratch is file-static and shared (see
//! backend/CLAUDE.md).

#[test]
#[ignore]
fn report_assistant_measurement() {
    let model_dir = std::env::var("MODEL_DIR").expect("MODEL_DIR");
    let eval_dir = std::env::var("EVAL_DIR").expect("EVAL_DIR");

    let mut config = llm::eval::EvalConfig::new(
        std::path::PathBuf::from(&model_dir),
        std::path::Path::new(&eval_dir),
    );
    config.budget = Some(llm::chat_template::ThinkBudget::High);

    let report = llm::eval::run_assistant(&config, Box::new(|_, _| {}), None)
        .expect("suite failed")
        .expect("cancelled");

    let chat = report.chat.as_ref().expect("no held-out chat set");
    eprintln!("--- {model_dir}");
    eprintln!("conversations   {}", chat.conversations);
    eprintln!("tokens_scored   {}", chat.tokens_scored);
    eprintln!("bytes_scored    {}", chat.bytes_scored);
    eprintln!("bytes/token     {:.4}", chat.bytes_scored as f64 / chat.tokens_scored as f64);
    eprintln!("bits_per_token  {:.4} +- {:.4}", chat.bits_per_token, chat.bits_per_token_stderr);
    eprintln!("bits_per_byte   {:.4} +- {:.4}", chat.bits_per_byte, chat.bits_per_byte_stderr);
    eprintln!("score           {:.2} +- {:.2}", report.score, report.breakdown.score_stderr);
}
