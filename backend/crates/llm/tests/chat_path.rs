//! The serving entry point, end to end on a real model.
//!
//!   LLM_TEST_MODEL_DIR=$PWD/data/artifacts/models/<uuid> \
//!     cargo test -p llm --release --test chat_path -- --ignored --nocapture --test-threads=1
//!
//! Ignored because it needs weights on disk. Everything below the HTTP layer is
//! exercised here: `build_prompt`'s template and truncation, the model cache and its
//! `InferSession`, prefill then decode against the KV cache, the shared sampler, the
//! reusable BPE decoder, incremental streaming, stop-marker detection, and
//! cancellation. The WebSocket plumbing above it is `jobs::queue`'s business.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use llm::chat_template::{Role, Turn};
use llm::sample::SamplingParams;

// The serving defaults, with the nucleus off: these tests assert about streaming,
// truncation and the KV cache, none of which should depend on how the candidate
// set was narrowed.
const SAMPLING: SamplingParams = SamplingParams { temperature: 0.8, top_k: 40, top_p: 1.0 };

// Greedy, for the two tests that compare one generation against another and need
// the run-to-run variation gone.
const GREEDY: SamplingParams = SamplingParams { temperature: 0.0, top_k: 1, top_p: 1.0 };

fn model_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("LLM_TEST_MODEL_DIR")
            .expect("set LLM_TEST_MODEL_DIR to a model directory (absolute path)"),
    )
}

fn user(text: &str) -> Turn {
    Turn { role: Role::User, text: text.to_string() }
}

fn assistant(text: &str) -> Turn {
    Turn { role: Role::Assistant, text: text.to_string() }
}

#[test]
#[ignore = "needs a trained model: set LLM_TEST_MODEL_DIR"]
fn a_chat_turn_streams_and_stops() {
    let dir = model_dir();
    let history = vec![user("What is the capital of France?")];

    let mut streamed = String::new();
    let out = llm::infer::generate_chat_response_streaming(
        &dir,
        Some("You are a helpful assistant."),
        &history,
        None,
        48,
        SAMPLING,
        None,
        |tok| streamed.push_str(tok),
    )
    .expect("generation failed");

    eprintln!("  input {} tokens, output {}", out.input_tokens, out.output_tokens);
    eprintln!("  reply: {:?}", out.text);

    assert!(out.input_tokens > 0, "the prompt encoded to nothing");
    assert!(out.output_tokens > 0, "no tokens were generated");
    assert!(out.output_tokens <= 48, "generated past max_tokens");

    // The streamed text is the reply plus, possibly, the stop marker that ends it —
    // `text` has that trimmed. Streaming and the return value must not diverge
    // beyond that: they come from the same decoder and the same accumulator.
    assert!(
        streamed.starts_with(&out.text),
        "streamed text and returned text disagree:\n  streamed {streamed:?}\n  returned {:?}",
        out.text
    );
    // A stop marker is scaffolding and must never reach the caller's text.
    assert!(!out.text.contains("<|endoftext|>"), "a stop marker leaked into the reply");
    assert!(!out.text.contains("<|/assistant|>"), "a stop marker leaked into the reply");
}

// Cancellation is checked before each forward pass, so a flag set up front should
// stop the turn without erroring — a partial reply is a real thing a user watched
// arrive, and the caller decides what to do with it.
#[test]
#[ignore = "needs a trained model: set LLM_TEST_MODEL_DIR"]
fn a_cancelled_turn_returns_what_it_had() {
    let dir = model_dir();
    let history = vec![user("Tell me a long story.")];
    let cancel = AtomicBool::new(true);

    let out = llm::infer::generate_chat_response_streaming(
        &dir,
        None,
        &history,
        None,
        64,
        SAMPLING,
        Some(&cancel),
        |_| {},
    )
    .expect("a cancelled turn should not be an error");

    assert_eq!(out.output_tokens, 0, "cancelled before the first pass, so nothing generated");
    assert!(out.text.is_empty());
}

// Consecutive turns are the case the KV cache and prefix reuse exist for: turn two
// extends turn one's sequence. This checks the second turn still produces a sane
// reply rather than something conditioned on a stale cache.
#[test]
#[ignore = "needs a trained model: set LLM_TEST_MODEL_DIR"]
fn a_second_turn_reuses_the_session_without_going_wrong() {
    let dir = model_dir();
    let mut history = vec![user("Hello there.")];

    let first = llm::infer::generate_chat_response_streaming(
        &dir, None, &history, None, 24, SAMPLING, None, |_| {},
    )
    .expect("first turn failed");

    history.push(assistant(&first.text));
    history.push(user("And what about tomorrow?"));

    let second = llm::infer::generate_chat_response_streaming(
        &dir, None, &history, None, 24, SAMPLING, None, |_| {},
    )
    .expect("second turn failed");

    eprintln!("  turn 1 ({} tokens): {:?}", first.output_tokens, first.text);
    eprintln!("  turn 2 ({} tokens): {:?}", second.output_tokens, second.text);

    assert!(second.output_tokens > 0, "the second turn generated nothing");
    assert!(
        second.input_tokens > first.input_tokens,
        "the second turn's prompt ({}) should be longer than the first's ({})",
        second.input_tokens,
        first.input_tokens
    );
}

// `release()` must leave the path usable: the next call reloads from disk. If it
// left a dangling session or a half-freed scratch, this is where it would show.
#[test]
#[ignore = "needs a trained model: set LLM_TEST_MODEL_DIR"]
fn generation_still_works_after_release() {
    let dir = model_dir();
    let history = vec![user("Hi.")];
    let gen = || {
        llm::infer::generate_chat_response_streaming(
            &dir, None, &history, None, 8, GREEDY, None, |_| {},
        )
        .expect("generation failed")
    };

    let before = gen();
    llm::infer::release();
    let after = gen();

    // Temperature 0 is argmax, so the same prompt through a freshly reloaded model
    // must give the same tokens back.
    assert_eq!(
        before.text, after.text,
        "greedy generation changed across a release/reload"
    );
    let _ = Ordering::SeqCst;
}

// ── the marker grammar, end to end ───────────────────────────────────────────

// Walks a served reply's markers, returning what is wrong with them.
//
// Deliberately a second implementation rather than a call into `llm::markup`: it
// works from the *decoded text* a client receives, so a bug in the automaton cannot
// hide behind the automaton's own idea of what it emitted.
fn markup_fault(text: &str) -> Option<String> {
    let mut stack: Vec<&str> = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find("<|") {
        let Some(end) = rest[open..].find("|>") else { break };
        let marker = &rest[open..open + end + 2];
        rest = &rest[open + end + 2..];
        let inner = &marker[2..marker.len() - 2];

        if inner.starts_with("think:") || inner == "divider" || inner == "endoftext" {
            continue;
        }
        // A role marker inside an assistant turn is the leakage the grammar exists
        // to stop; it can never be balanced, so it is its own fault.
        if matches!(inner, "user" | "/user" | "system" | "/system" | "assistant") {
            return Some(format!("role marker <|{inner}|> inside the reply"));
        }
        match inner.strip_prefix('/') {
            Some(name) => match stack.pop() {
                Some(top) if top == name => {}
                Some(top) => return Some(format!("<|/{name}|> closed <|{top}|>")),
                None => return Some(format!("stray <|/{name}|>")),
            },
            None => stack.push(inner),
        }
    }
    (!stack.is_empty()).then(|| format!("left open: {stack:?}"))
}

// The reason `llm::markup` exists. Before it, the markup suite scored
// `closer_matches_opener` at 52% against a 50% baseline and served replies showed
// exactly that: `<|par|>` closed by `<|/item|>`, `<|/user|><|assistant|>` invented
// mid-turn, blocks left hanging. None of that is now representable.
//
// Sampled rather than greedy, and over prompts that draw out lists and nesting,
// because the failure was never on the most likely token — it was in the tail.
#[test]
#[ignore = "needs a trained model: set LLM_TEST_MODEL_DIR"]
fn a_served_reply_is_always_well_formed_markup() {
    let dir = model_dir();
    let prompts = [
        "Give me a list of three things to check on a leaking sink.",
        "List the steps to make tea.",
        "Explain in bullet points why the sky is blue.",
        "What is your name?",
    ];

    for prompt in prompts {
        for _ in 0..3 {
            let history = vec![user(prompt)];
            let out = llm::infer::generate_chat_response_streaming(
                &dir, None, &history, None, 250, SAMPLING, None, |_| {},
            )
            .expect("generation failed");
            if let Some(fault) = markup_fault(&out.text) {
                panic!("{prompt}: {fault}\n{}", out.text);
            }
        }
    }
}

// The token cap is the one stop the grammar cannot refuse, so the loop reserves room
// for the closers and drains what is still open. Short caps are where that has to
// work: they cut the reply mid-block almost every time.
#[test]
#[ignore = "needs a trained model: set LLM_TEST_MODEL_DIR"]
fn a_reply_cut_off_by_the_token_cap_is_still_well_formed() {
    let dir = model_dir();
    for cap in [4, 8, 12, 20, 48] {
        for _ in 0..5 {
            let history = vec![user("Give me a long list of things to check on a sink.")];
            let out = llm::infer::generate_chat_response_streaming(
                &dir, None, &history, None, cap, SAMPLING, None, |_| {},
            )
            .expect("generation failed");

            if let Some(fault) = markup_fault(&out.text) {
                panic!("cap {cap}: {fault}\n{}", out.text);
            }
            // The drain is generated output too, and must not buy its way past the
            // ceiling the caller set.
            assert!(
                out.output_tokens <= cap,
                "cap {cap} produced {} tokens",
                out.output_tokens,
            );
        }
    }
}
