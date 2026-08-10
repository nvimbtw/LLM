//! Exercises the client against a real engine the supervisor starts.
//!
//! Run single-threaded: these bind real ports and hold the engine's single
//! device lock. `cargo test -p engine -- --test-threads=1`

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use engine::client::{EvalRequest, InferRequest, RunEvent, SftRequest, TrainRequest};
use engine::{Client, Engine, EngineConfig};

fn config(port: u16) -> Option<EngineConfig> {
    let dir = PathBuf::from("../../../engine");
    let python = dir.join(".venv/bin/python");
    if !python.is_file() {
        eprintln!("skipping: no venv at {}", python.display());
        return None;
    }
    Some(EngineConfig {
        python,
        data_dir: PathBuf::from("../../data"),
        dir,
        host: "127.0.0.1".into(),
        port,
        autostart: true,
        startup_timeout: Duration::from_secs(90),
    })
}

fn tiny(name: &str) -> TrainRequest {
    TrainRequest {
        model_name: name.into(),
        dimensions: Some(128),
        heads: Some(4),
        layers: Some(2),
        ctx_window: Some(128),
        max_vocab: Some(2048),
        epochs: Some(1),
        batch_size: Some(8),
        warmup_steps: Some(20),
        ..Default::default()
    }
}

#[tokio::test]
async fn trains_streams_and_evaluates() {
    let Some(cfg) = config(8130) else { return };
    let url = cfg.base_url();
    let _engine = Engine::start(cfg).await.expect("engine starts");
    let client = Client::new(url);

    client.health().await.expect("health");

    let run = client.train(&tiny("client-train")).await.expect("train accepted");
    assert_eq!(run.status, "running", "train should return before it finishes");

    // Progress frames must arrive, and the stream must end on a terminal frame.
    let frames = Arc::new(Mutex::new((0usize, false)));
    let seen = frames.clone();
    let finished = client
        .follow(&run.run_id, move |event| {
            let mut s = seen.lock().unwrap();
            match event {
                RunEvent::Progress(_) => s.0 += 1,
                RunEvent::Done(_) => s.1 = true,
                other => panic!("unexpected terminal frame: {other:?}"),
            }
        })
        .await
        .expect("follow");

    let (progress, done) = *frames.lock().unwrap();
    assert!(progress > 0, "expected progress frames, got none");
    assert!(done, "stream should end on a done frame");
    assert_eq!(finished.status, "completed");
    assert!(finished.is_terminal());

    // The device is free again, so the next run is accepted.
    let evaluated = client
        .evaluate(&EvalRequest { model_name: "client-train".into(), max_tokens: Some(24), ..Default::default() })
        .await
        .expect("eval accepted");
    let evaluated = client.follow(&evaluated.run_id, |_| {}).await.expect("follow eval");
    assert_eq!(evaluated.status, "completed", "eval failed: {:?}", evaluated.error);

    let score = evaluated.result.as_ref().and_then(|r| r.get("score")).cloned();
    assert!(score.is_some(), "eval produced no score");

    let reply = client
        .infer(&InferRequest {
            model_name: "client-train".into(),
            prompt: "The history of".into(),
            max_tokens: Some(8),
            seed: Some(1),
            ..Default::default()
        })
        .await
        .expect("infer");
    assert_eq!(reply.generated_tokens, 8);
}

#[tokio::test]
async fn cancels_a_run_at_a_step_boundary() {
    let Some(cfg) = config(8131) else { return };
    let url = cfg.base_url();
    let _engine = Engine::start(cfg).await.expect("engine starts");
    let client = Client::new(url);

    let mut long = tiny("client-cancel");
    long.epochs = Some(20);
    let run = client.train(&long).await.expect("train accepted");

    // Let it get past dataset building and into the loop.
    tokio::time::sleep(Duration::from_secs(12)).await;
    client.cancel(&run.run_id).await.expect("cancel");

    let finished = client.follow(&run.run_id, |_| {}).await.expect("follow");
    assert!(finished.is_cancelled(), "expected cancelled, got {}", finished.status);

    // Cancelling something already stopped is the caller's desired outcome, not
    // an error — the engine answers 409 and the client treats it as success.
    client.cancel(&run.run_id).await.expect("second cancel should be a no-op");
}

#[tokio::test]
async fn surfaces_the_engines_own_error_message() {
    let Some(cfg) = config(8132) else { return };
    let url = cfg.base_url();
    let _engine = Engine::start(cfg).await.expect("engine starts");
    let client = Client::new(url);

    let err = client
        .infer(&InferRequest {
            model_name: "no-such-model".into(),
            prompt: "hello".into(),
            ..Default::default()
        })
        .await
        .expect_err("should fail");

    let message = err.to_string();
    assert!(message.contains("404"), "wrong status: {message}");
    assert!(message.contains("no-such-model"), "engine's detail was lost: {message}");
}

#[tokio::test]
async fn refuses_a_second_run_while_the_device_is_busy() {
    let Some(cfg) = config(8133) else { return };
    let url = cfg.base_url();
    let _engine = Engine::start(cfg).await.expect("engine starts");
    let client = Client::new(url);

    let mut long = tiny("client-busy");
    long.epochs = Some(20);
    let first = client.train(&long).await.expect("first accepted");

    let err = client.train(&tiny("client-busy-2")).await.expect_err("second refused");
    assert!(err.to_string().contains("409"), "wrong error: {err}");

    client.cancel(&first.run_id).await.ok();
}

#[tokio::test]
async fn fine_tunes_from_a_pretrained_base() {
    let Some(cfg) = config(8134) else { return };
    let url = cfg.base_url();
    let _engine = Engine::start(cfg).await.expect("engine starts");
    let client = Client::new(url);

    let base = client.train(&tiny("client-base")).await.expect("pretrain accepted");
    let base = client.follow(&base.run_id, |_| {}).await.expect("follow pretrain");
    assert_eq!(base.status, "completed", "pretrain failed: {:?}", base.error);

    let sft = client
        .sft(&SftRequest {
            model_name: "client-sft".into(),
            base_model: "client-base".into(),
            epochs: Some(1),
            batch_size: Some(4),
            warmup_steps: Some(10),
            ..Default::default()
        })
        .await
        .expect("sft accepted");
    let sft = client.follow(&sft.run_id, |_| {}).await.expect("follow sft");
    assert_eq!(sft.status, "completed", "sft failed: {:?}", sft.error);

    let trainable = sft.result.as_ref().and_then(|r| r.get("trainable_tokens")).cloned();
    assert!(trainable.is_some(), "sft reported no trainable tokens");
}

#[tokio::test]
async fn runs_each_eval_suite() {
    let Some(cfg) = config(8135) else { return };
    let url = cfg.base_url();
    let _engine = Engine::start(cfg).await.expect("engine starts");
    let client = Client::new(url);

    // The suites are scored separately and must each produce their own report.
    for suite in ["language", "markup", "assistant"] {
        let run = client
            .evaluate(&EvalRequest {
                model_name: "base-sft".into(),
                suite: Some(suite.into()),
                max_tokens: Some(16),
                ..Default::default()
            })
            .await
            .unwrap_or_else(|e| panic!("{suite} not accepted: {e}"));

        let done = client.follow(&run.run_id, |_| {}).await.expect("follow");
        assert_eq!(done.status, "completed", "{suite} failed: {:?}", done.error);

        let result = done.result.expect("no result");
        assert_eq!(result.get("suite").and_then(|s| s.as_str()), Some(suite));
        assert!(result.get("score").is_some(), "{suite} produced no score");
    }
}

#[tokio::test]
async fn rejects_an_unknown_suite() {
    let Some(cfg) = config(8136) else { return };
    let url = cfg.base_url();
    let _engine = Engine::start(cfg).await.expect("engine starts");
    let client = Client::new(url);

    let run = client
        .evaluate(&EvalRequest {
            model_name: "base-sft".into(),
            suite: Some("nonsense".into()),
            ..Default::default()
        })
        .await
        .expect("accepted as a run");

    // The suite name is only validated once the run executes, so the failure
    // arrives as a terminal event rather than as a rejected request.
    let done = client.follow(&run.run_id, |_| {}).await.expect("follow");
    assert_eq!(done.status, "failed");
    assert!(done.error.unwrap_or_default().contains("unknown suite"));
}

#[tokio::test]
async fn chats_streams_measures_and_counts() {
    let Some(cfg) = config(8137) else { return };
    let url = cfg.base_url();
    let _engine = Engine::start(cfg).await.expect("engine starts");
    let client = Client::new(url);

    let turns = vec![engine::Turn {
        role: engine::Role::User,
        text: "Tell me about water.".into(),
    }];

    let mut req = engine::client::ChatRequest::new("base-sft", turns.clone(), 24);
    req.system = Some("You are concise.".into());
    req.budget = Some(engine::ThinkBudget::Low);

    let reply = client.chat(&req).await.expect("chat");
    assert!(reply.prompt_tokens > 0);
    assert!(reply.generated_tokens > 0);

    // Streaming must reconstruct exactly the same text as the batch call. A
    // byte-level token can be half a UTF-8 character, so this is the assertion
    // that the engine decodes pieces in context rather than individually.
    let streamed = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = streamed.clone();
    let summary = client
        .chat_stream(&req, move |piece| sink.lock().unwrap().push_str(piece))
        .await
        .expect("chat stream");
    assert_eq!(*streamed.lock().unwrap(), summary.text, "streamed pieces != final text");

    let count = client.count_tokens("base-sft", "hello world").await.expect("count");
    assert!(count > 0);

    let measured = client
        .measure(&engine::client::MeasureRequest {
            model_name: "base-sft".into(),
            turns,
            system: None,
        })
        .await
        .expect("measure");
    assert!(measured.trainable_tokens <= measured.tokens);

    let markers = client.markers().await.expect("markers");
    assert!(markers.markers.contains(&"<|assistant|>".to_string()));
    assert!(markers.markers.len() <= markers.block_size);
}

#[tokio::test]
async fn trains_over_one_socket() {
    let Some(cfg) = config(8138) else { return };
    let url = cfg.base_url();
    let _engine = Engine::start(cfg).await.expect("engine starts");
    let client = Client::new(url);

    let frames = Arc::new(Mutex::new((0usize, false)));
    let seen = frames.clone();
    let finished = client
        .train_stream(
            &tiny("socket-train"),
            Arc::new(AtomicBool::new(false)),
            move |event| {
                let mut s = seen.lock().unwrap();
                match event {
                    RunEvent::Progress(_) => s.0 += 1,
                    RunEvent::Done(_) => s.1 = true,
                    other => panic!("unexpected terminal frame: {other:?}"),
                }
            },
        )
        .await
        .expect("train over socket");

    let (progress, done) = *frames.lock().unwrap();
    assert!(progress > 0, "expected progress frames, got none");
    assert!(done, "socket should end on a done frame");
    assert_eq!(finished.status, "completed", "train failed: {:?}", finished.error);

    // The run is the registry's like any other, so it is still addressable by id.
    let again = client.run(&finished.run_id).await.expect("run by id");
    assert_eq!(again.status, "completed");
}

#[tokio::test]
async fn cancels_over_the_same_socket() {
    let Some(cfg) = config(8139) else { return };
    let url = cfg.base_url();
    let _engine = Engine::start(cfg).await.expect("engine starts");
    let client = Client::new(url);

    let mut long = tiny("socket-cancel");
    long.epochs = Some(20);

    // Flipped on the first progress frame — the same flag the queue sets, read
    // on the socket's own poll. Keyed to the run reaching the loop rather than
    // to a sleep, so the assertion does not depend on how fast the corpus packs.
    let cancel = Arc::new(AtomicBool::new(false));
    let trigger = cancel.clone();
    let finished = client
        .train_stream(&long, cancel, move |event| {
            if let RunEvent::Progress(p) = event {
                if p.get("step").is_some() {
                    trigger.store(true, Ordering::Relaxed);
                }
            }
        })
        .await
        .expect("train over socket");
    assert!(finished.is_cancelled(), "expected cancelled, got {}", finished.status);
}

#[tokio::test]
async fn a_busy_engine_refuses_the_socket_before_a_run_exists() {
    let Some(cfg) = config(8140) else { return };
    let url = cfg.base_url();
    let _engine = Engine::start(cfg).await.expect("engine starts");
    let client = Client::new(url);

    let mut long = tiny("socket-busy");
    long.epochs = Some(20);
    let first = client.train(&long).await.expect("first accepted");

    let err = client
        .train_stream(&tiny("socket-busy-2"), Arc::new(AtomicBool::new(false)), |_| {})
        .await
        .expect_err("second refused");
    assert!(err.to_string().contains("still active"), "engine's detail was lost: {err}");

    client.cancel(&first.run_id).await.ok();
}
