//! The authenticated hop, against a real engine.
//!
//! Everything else about the hop is covered by the engine's own suite; what
//! only this side can prove is that a token minted here opens a door there.

use std::path::PathBuf;
use std::time::Duration;

use service::{Config, Proxy, Supervisor};

const SECRET: &str = "seam-test-secret";

fn config(port: u16, secret: Option<&str>) -> Option<Config> {
    let dir = PathBuf::from("../../../engine");
    if !dir.join(".venv/bin/python").is_file() {
        eprintln!("skipping: no venv");
        return None;
    }
    let root = std::env::temp_dir().join(format!("slate-seam-{port}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("input/Tiny")).unwrap();
    std::fs::write(root.join("input/Tiny/a.txt"), "the cat sat on the mat.\n".repeat(50)).unwrap();

    Some(Config {
        audience: "engine".into(),
        python: dir.join(".venv/bin/python"),
        dir,
        host: "127.0.0.1".into(),
        port,
        autostart: true,
        service_secret: secret.map(str::to_string),
        secret_var: "ENGINE_SERVICE_SECRET".into(),
        extra_env: vec![("DATA_DIR".into(), root.display().to_string())],
        startup_timeout: Duration::from_secs(90),
    })
}

#[tokio::test]
async fn a_signed_client_reaches_an_enforcing_engine() {
    let Some(cfg) = config(8151, Some(SECRET)) else { return };
    let url = cfg.base_url();
    let _engine = Supervisor::start(cfg).await.expect("engine starts");

    Proxy::new(url, "engine", Some(SECRET.into()))
        .get(&auth::Caller::System, "/")
        .await
        .expect("signed call accepted");
}

#[tokio::test]
async fn an_unsigned_client_is_refused_by_an_enforcing_engine() {
    let Some(cfg) = config(8152, Some(SECRET)) else { return };
    let url = cfg.base_url();
    let _engine = Supervisor::start(cfg).await.expect("engine starts");

    let err = Proxy::new(url, "engine", None)
        .get(&auth::Caller::System, "/")
        .await
        .expect_err("must be refused");
    assert!(err.to_string().contains("401"), "wrong failure: {err}");
}

#[tokio::test]
async fn the_wrong_secret_is_refused() {
    let Some(cfg) = config(8153, Some(SECRET)) else { return };
    let url = cfg.base_url();
    let _engine = Supervisor::start(cfg).await.expect("engine starts");

    let err = Proxy::new(url, "engine", Some("not-the-secret".into()))
        .get(&auth::Caller::System, "/")
        .await
        .expect_err("must be refused");
    assert!(err.to_string().contains("401"), "wrong failure: {err}");
}
