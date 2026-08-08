//! Owns the Python engine process for the lifetime of the server.
//!
//! There is one engine and it holds the GPU, so its lifecycle belongs to
//! whatever else is long-lived — the Actix server — rather than to individual
//! jobs. The supervisor spawns it, waits for it to answer, restarts it if it
//! dies, and kills it on shutdown.

use std::process::Stdio;
use std::time::Duration;

use tokio::process::{Child, Command};
use tokio::sync::watch;

use crate::config::EngineConfig;

// Restart backoff bounds. A crash-looping engine (bad venv, port already bound)
// should not spin the CPU retrying, but a one-off segfault should come back
// quickly enough that a queued job barely notices.
const BACKOFF_START: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(30);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// A handle to the running engine. Dropping it stops the supervisor and kills
/// the child.
#[derive(Debug)]
pub struct Engine {
    config: EngineConfig,
    // Signals the supervisor task to stop. `None` when the engine is externally
    // managed (autostart off), where there is no task and nothing to kill.
    shutdown: Option<watch::Sender<bool>>,
}

impl Engine {
    /// Start the engine and wait until it answers its health check.
    ///
    /// With autostart off this only verifies that something is already
    /// listening, so a developer running `uvicorn --reload` by hand gets the
    /// same up-front failure as a misconfigured spawn rather than a surprise at
    /// the first job.
    pub async fn start(config: EngineConfig) -> Result<Engine, String> {
        let base_url = config.base_url();

        if !config.autostart {
            wait_until_healthy(&base_url, config.startup_timeout).await.map_err(|e| {
                format!("ENGINE_AUTOSTART is off and no engine is answering at {base_url}: {e}")
            })?;
            return Ok(Engine { config, shutdown: None });
        }

        // Paths made absolute here, before anything spawns with a different
        // working directory. See `EngineConfig::resolved`.
        let config = config.resolved()?;

        let mut child = spawn(&config)?;
        // Health-check before returning, so `start` failing means "the engine is
        // not usable" rather than "the process exists". A server that boots
        // green and fails every job is the outcome worth avoiding.
        //
        // The child is watched alongside the port, and that is not belt-and-braces.
        // Health-checking the port alone cannot tell a healthy new engine from a
        // *stale one still holding the port*: the new child dies on bind, the old
        // process answers 200, and the supervisor reports ready while every
        // request goes to an engine with the wrong configuration. That failure is
        // silent, survives restarts, and looks exactly like a config bug.
        if let Err(e) = wait_until_healthy_child(&mut child, &base_url, config.startup_timeout).await
        {
            return Err(format!("engine spawned but never became healthy: {e}"));
        }

        let (tx, rx) = watch::channel(false);
        tokio::spawn(supervise(config.clone(), child, rx));

        Ok(Engine { config, shutdown: Some(tx) })
    }

    pub fn base_url(&self) -> String {
        self.config.base_url()
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Stop supervising and kill the child. Idempotent.
    pub fn shutdown(&self) {
        if let Some(tx) = &self.shutdown {
            let _ = tx.send(true);
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The supervision loop: wait on the child, restart it if it dies unexpectedly,
/// kill it when asked to stop.
async fn supervise(config: EngineConfig, mut child: Child, mut shutdown: watch::Receiver<bool>) {
    let mut backoff = BACKOFF_START;

    loop {
        tokio::select! {
            // Biased so a shutdown racing a crash always wins: the alternative
            // is spawning a replacement microseconds before being told to stop,
            // which orphans it.
            biased;

            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    let _ = child.kill().await;
                    return;
                }
            }

            status = child.wait() => {
                if *shutdown.borrow() {
                    return;
                }
                eprintln!(
                    "engine: process exited ({}); restarting in {:?}",
                    status.map(|s| s.to_string()).unwrap_or_else(|e| e.to_string()),
                    backoff,
                );
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(BACKOFF_MAX);

                match spawn(&config) {
                    Ok(replacement) => {
                        child = replacement;
                        // Only reset the backoff once the replacement is
                        // actually serving. Resetting on spawn alone would let
                        // an engine that dies during startup retry every second
                        // forever, which is the loop this bound exists to stop.
                        if wait_until_healthy(&config.base_url(), config.startup_timeout)
                            .await
                            .is_ok()
                        {
                            backoff = BACKOFF_START;
                        }
                    }
                    Err(e) => {
                        eprintln!("engine: respawn failed: {e}");
                        return;
                    }
                }
            }
        }
    }
}

fn spawn(config: &EngineConfig) -> Result<Child, String> {
    Command::new(&config.python)
        .args([
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            &config.host,
            "--port",
            &config.port.to_string(),
        ])
        .current_dir(&config.dir)
        // The engine resolves every path through this; see EngineConfig::data_dir.
        .env("DATA_DIR", &config.data_dir)
        // Inherited, so the engine's own logs land in the server's output
        // instead of filling an OS pipe buffer and blocking it once nobody
        // reads them.
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        // Kills the child if this handle is ever dropped without `shutdown`
        // being called — a panicking supervisor task must not leave a process
        // holding the GPU. It does NOT cover the parent being SIGKILLed; on that
        // path the engine is orphaned and has to be reaped by hand.
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("failed to spawn {}: {e}", config.python.display()))
}

/// Poll for health, failing early if the child dies first.
///
/// A child that exits during startup is almost always a port collision, so the
/// error says so rather than leaving the caller with a bare timeout.
async fn wait_until_healthy_child(
    child: &mut Child,
    base_url: &str,
    timeout: Duration,
) -> std::result::Result<(), String> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut last = "no attempt made".to_string();

    while tokio::time::Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!(
                "the engine process exited during startup ({status}) — something \
                 else is probably already listening on {base_url}"
            ));
        }
        match health_once(base_url).await {
            Ok(()) => return Ok(()),
            Err(e) => last = e,
        }
        tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
    }

    Err(format!("timed out after {timeout:?} (last: {last})"))
}

async fn health_once(base_url: &str) -> std::result::Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;
    match client.get(base_url).send().await {
        Ok(r) if r.status().is_success() => Ok(()),
        Ok(r) => Err(format!("HTTP {}", r.status())),
        Err(e) => Err(e.to_string()),
    }
}

/// Poll `GET /` until it answers or the timeout expires.
async fn wait_until_healthy(base_url: &str, timeout: Duration) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;

    let deadline = tokio::time::Instant::now() + timeout;
    let mut last = "no attempt made".to_string();

    while tokio::time::Instant::now() < deadline {
        match client.get(base_url).send().await {
            Ok(r) if r.status().is_success() => return Ok(()),
            Ok(r) => last = format!("HTTP {}", r.status()),
            Err(e) => last = e.to_string(),
        }
        tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
    }

    Err(format!("timed out after {timeout:?} (last: {last})"))
}
