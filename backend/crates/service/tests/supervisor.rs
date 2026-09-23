//! Exercises the supervisor against the real Python engine.
//!
//! These spawn an actual uvicorn, so they need `llm-engine/.venv` to exist. Without
//! it they skip rather than fail: a checkout that has not run `pip install` yet
//! is not a broken supervisor.
//!
//! Run them single-threaded — `cargo test -p engine -- --test-threads=1`. They
//! bind real ports and one of them uses `pkill`, which is process-wide however
//! specific its pattern.

use std::path::PathBuf;
use std::time::Duration;

use service::{Config, Supervisor};

fn test_config(port: u16) -> Option<Config> {
    // Tests run with the crate root as the working directory.
    let dir = PathBuf::from("../../../llm-engine");
    let python = dir.join(".venv/bin/python");
    if !python.is_file() {
        eprintln!("skipping: no venv at {}", python.display());
        return None;
    }
    Some(Config {
        audience: "llm-engine".to_string(),
        python,
        dir,
        host: "127.0.0.1".to_string(),
        port,
        autostart: true,
        service_secret: None,
        secret_var: "LLM_ENGINE_SERVICE_SECRET".to_string(),
        extra_env: vec![("DATA_DIR".to_string(), "../../data".to_string())],
        startup_timeout: Duration::from_secs(90),
    })
}

async fn responds(base_url: &str) -> bool {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap()
        .get(base_url)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

#[tokio::test]
async fn starts_serves_and_stops() {
    let Some(config) = test_config(8123) else {
        return;
    };
    let base_url = config.base_url();

    let engine = Supervisor::start(config)
        .await
        .expect("engine should start");
    assert!(
        responds(&base_url).await,
        "engine should answer once start() returns"
    );

    engine.shutdown();
    // The kill is asynchronous; give the child a moment to actually go away.
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert!(
        !responds(&base_url).await,
        "engine should be gone after shutdown"
    );
}

#[tokio::test]
async fn missing_interpreter_fails_at_start() {
    let config = Config {
        audience: "llm-engine".to_string(),
        python: PathBuf::from("/nonexistent/python"),
        dir: PathBuf::from("."),
        host: "127.0.0.1".to_string(),
        port: 8124,
        autostart: true,
        service_secret: None,
        secret_var: "LLM_ENGINE_SERVICE_SECRET".to_string(),
        extra_env: Vec::new(),
        startup_timeout: Duration::from_secs(2),
    };

    let err = Supervisor::start(config)
        .await
        .expect_err("should refuse to start");
    assert!(err.contains("does not exist"), "unhelpful error: {err}");
}

#[tokio::test]
async fn autostart_off_requires_a_running_engine() {
    let config = Config {
        audience: "llm-engine".to_string(),
        python: PathBuf::from("/nonexistent/python"),
        dir: PathBuf::from("."),
        host: "127.0.0.1".to_string(),
        port: 8125,
        autostart: false,
        service_secret: None,
        secret_var: "LLM_ENGINE_SERVICE_SECRET".to_string(),
        extra_env: Vec::new(),
        startup_timeout: Duration::from_secs(2),
    };

    // Nothing is listening, so this must fail rather than silently pretend an
    // externally-managed engine is there.
    let err = Supervisor::start(config)
        .await
        .expect_err("should fail with nothing listening");
    // Actionable means naming the switch that caused it and the address it tried.
    assert!(
        err.contains("LLM_ENGINE_AUTOSTART is off"),
        "unhelpful error: {err}"
    );
    assert!(err.contains("127.0.0.1:8125"), "unhelpful error: {err}");
}

#[tokio::test]
async fn restarts_the_engine_when_it_dies() {
    let Some(config) = test_config(8126) else {
        return;
    };
    let base_url = config.base_url();

    let _engine = Supervisor::start(config)
        .await
        .expect("engine should start");
    assert!(responds(&base_url).await);

    // Kill it the way the OS would — out from under the supervisor, with no
    // shutdown signal — so the restart path is what is under test.
    std::process::Command::new("pkill")
        .args(["-f", "uvicorn main:app --host 127.0.0.1 --port 8126"])
        .status()
        .expect("pkill should run");

    // Long enough to cover the death, the 1s backoff, and a cold torch import.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
    let mut came_back = false;
    while tokio::time::Instant::now() < deadline {
        if responds(&base_url).await {
            came_back = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert!(came_back, "supervisor should have restarted the LLM engine");
}
