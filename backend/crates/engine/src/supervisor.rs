//! Owns the Python engine process for the lifetime of the server.
//!
//! There is one engine and it holds the GPU, so its lifecycle belongs to
//! whatever else is long-lived — the Actix server — rather than to any one
//! request.

use std::process::Stdio;
use std::time::Duration;

use tokio::process::{Child, Command};
use tokio::sync::watch;

use crate::config::EngineConfig;

// A crash-looping engine must not spin the CPU retrying, but a one-off segfault
// should come back before a queued run notices.
const BACKOFF_START: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(30);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug)]
pub struct Engine {
    config: EngineConfig,
    /// `None` when the engine is externally managed: no task, nothing to kill.
    shutdown: Option<watch::Sender<bool>>,
}

impl Engine {
    /// Start the engine and wait until it answers.
    ///
    /// Returning only once it is *serving* is the point: a server that boots
    /// green and fails every request is the outcome worth avoiding. With
    /// autostart off this verifies that something is already listening, so a
    /// hand-run `uvicorn --reload` fails the same way a misconfigured spawn does.
    pub async fn start(config: EngineConfig) -> Result<Engine, String> {
        let base_url = config.base_url();
        let secret = config.service_secret.clone();

        if !config.autostart {
            wait_until_healthy(&base_url, config.startup_timeout, secret.as_deref(), None)
                .await
                .map_err(|e| {
                    format!("ENGINE_AUTOSTART is off and no engine answers at {base_url}: {e}")
                })?;
            return Ok(Engine { config, shutdown: None });
        }

        // Before anything spawns with a different working directory.
        let config = config.resolved()?;
        let mut child = spawn(&config)?;

        wait_until_healthy(
            &base_url,
            config.startup_timeout,
            secret.as_deref(),
            Some(&mut child),
        )
        .await
        .map_err(|e| format!("engine spawned but never became healthy: {e}"))?;

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

async fn supervise(config: EngineConfig, mut child: Child, mut shutdown: watch::Receiver<bool>) {
    let mut backoff = BACKOFF_START;

    loop {
        tokio::select! {
            // Biased so a shutdown racing a crash always wins; the alternative
            // spawns a replacement microseconds before being told to stop.
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
                        // Reset only once the replacement is serving. Resetting
                        // on spawn alone lets an engine that dies during startup
                        // retry every second forever.
                        let healthy = wait_until_healthy(
                            &config.base_url(),
                            config.startup_timeout,
                            config.service_secret.as_deref(),
                            None,
                        )
                        .await;
                        if healthy.is_ok() {
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
        .env("DATA_DIR", &config.data_dir)
        .envs(config.service_secret.iter().map(|s| ("ENGINE_SERVICE_SECRET", s)))
        // Inherited so the engine's logs land in the server's output rather than
        // filling a pipe buffer and blocking on it once nobody reads them.
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        // Covers a panicking supervisor task, which must not leave a process
        // holding the GPU. It does *not* cover the parent being SIGKILLed —
        // that path orphans the engine and it has to be reaped by hand.
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("failed to spawn {}: {e}", config.python.display()))
}

/// Poll `GET /` until it answers, `child` dies, or the timeout expires.
///
/// Watching the child is not belt-and-braces: the port alone cannot tell a
/// healthy new engine from a *stale one still holding it* — the new child dies
/// on bind, the old process answers 200, and every request then goes to an
/// engine with the wrong configuration.
///
/// Signed like every other call, because an engine with a secret answers 401 to
/// an unsigned probe and would look like it never started.
async fn wait_until_healthy(
    base_url: &str,
    timeout: Duration,
    secret: Option<&str>,
    mut child: Option<&mut Child>,
) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;

    let deadline = tokio::time::Instant::now() + timeout;
    let mut last = "no attempt made".to_string();

    while tokio::time::Instant::now() < deadline {
        if let Some(Ok(Some(status))) = child.as_deref_mut().map(Child::try_wait) {
            return Err(format!(
                "the engine exited during startup ({status}) — something else is \
                 probably already listening on {base_url}"
            ));
        }

        match sign(client.get(base_url), secret).send().await {
            Ok(r) if r.status().is_success() => return Ok(()),
            Ok(r) => last = format!("HTTP {}", r.status()),
            Err(e) => last = e.to_string(),
        }
        tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
    }

    Err(format!("timed out after {timeout:?} (last: {last})"))
}

fn sign(request: reqwest::RequestBuilder, secret: Option<&str>) -> reqwest::RequestBuilder {
    let Some(secret) = secret else { return request };
    match auth::token::issue_service_token(crate::AUDIENCE, &auth::Caller::System, &[], secret) {
        Ok(token) => request.bearer_auth(token),
        Err(_) => request,
    }
}
