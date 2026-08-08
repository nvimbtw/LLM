//! Where the Python engine lives and how to start it.
//!
//! Every field is configurable because none of them are properties of *this*
//! project — they are properties of the machine it runs on. Hardcoding
//! `.venv/bin/python` and port 8001 would bake one developer's box into the
//! server; port 8000 on that same box is already taken by an unrelated Apache
//! instance, which is exactly the kind of thing config exists for.

use std::path::PathBuf;
use std::time::Duration;

/// Read from the environment, with defaults that suit a checkout laid out the
/// way this repository is.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// The Python interpreter to run the engine with. The engine is started as
    /// `<python> -m uvicorn`, so this is the only interpreter-shaped knob —
    /// there is no separate path to a `uvicorn` binary to keep in sync with it.
    pub python: PathBuf,
    /// Working directory for the engine process; `main:app` is resolved
    /// relative to it.
    pub dir: PathBuf,
    pub host: String,
    pub port: u16,
    /// Whether to spawn the process at all. Off means "an engine is already
    /// running and I will manage it myself" — the normal case when developing
    /// against `uvicorn --reload` in another terminal.
    pub autostart: bool,
    /// The data root the engine works in, handed to it as `DATA_DIR`.
    ///
    /// Relative paths resolve against `dir`, so the default is the engine's own
    /// `data/`. The engine owns that tree outright — corpora arrive through the
    /// fetch scripts, and this side only names them in a curriculum.
    pub data_dir: PathBuf,
    /// How long to wait for a freshly spawned engine to answer its health check
    /// before giving up. Importing torch is slow enough (seconds, and more on a
    /// cold page cache) that a short timeout would flap.
    pub startup_timeout: Duration,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let dir = PathBuf::from("../engine");
        EngineConfig {
            python: dir.join(".venv/bin/python"),
            data_dir: dir.join("data"),
            dir,
            host: "127.0.0.1".to_string(),
            port: 8001,
            autostart: true,
            startup_timeout: Duration::from_secs(60),
        }
    }
}

impl EngineConfig {
    /// Build from the environment, falling back to the defaults above.
    ///
    /// `ENGINE_PYTHON` is resolved against `ENGINE_DIR` when it is relative, so
    /// the common case of moving the checkout needs one variable set, not two.
    pub fn from_env() -> EngineConfig {
        let defaults = EngineConfig::default();

        let dir = std::env::var("ENGINE_DIR")
            .map(PathBuf::from)
            .unwrap_or(defaults.dir);

        let python = match std::env::var("ENGINE_PYTHON") {
            Ok(p) => {
                let p = PathBuf::from(p);
                if p.is_relative() && p.components().count() > 1 {
                    dir.join(p)
                } else {
                    p
                }
            }
            Err(_) => dir.join(".venv/bin/python"),
        };

        // Computed before `dir` moves into the struct. Defaults to the engine's
        // own `data/`, which it owns outright; DATA_DIR overrides it as given.
        let data_dir = std::env::var("DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dir.join("data"));

        EngineConfig {
            python,
            dir,
            data_dir,
            host: env_or("ENGINE_HOST", defaults.host),
            port: parse_env("ENGINE_PORT").unwrap_or(defaults.port),
            // Anything but an explicit falsey value keeps the default on: the
            // failure mode of spawning an engine nobody asked for is a wasted
            // process, and of *not* spawning one is a server that looks healthy
            // and fails every job.
            autostart: std::env::var("ENGINE_AUTOSTART")
                .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no"))
                .unwrap_or(defaults.autostart),
            startup_timeout: parse_env("ENGINE_STARTUP_TIMEOUT_SECS")
                .map(Duration::from_secs)
                .unwrap_or(defaults.startup_timeout),
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    /// Check the paths exist and return a copy with both made absolute.
    ///
    /// The absolute part is not cosmetic. `Command` resolves the *program* path
    /// after chdir-ing into `current_dir`, so spawning a relative
    /// `../engine/.venv/bin/python` with the working directory set to
    /// `../engine` looks for it under `../engine/../engine/...` and fails with a
    /// bare ENOENT. Resolving both up front is the only thing that makes a
    /// relative `ENGINE_PYTHON` mean what it reads as.
    ///
    /// Existence is an error rather than a warning: an engine that never starts
    /// turns every training and inference job into a confusing connection
    /// refused, minutes later and far from the cause. A missing interpreter is
    /// knowable at boot, so it is reported at boot.
    pub fn resolved(&self) -> Result<EngineConfig, String> {
        if !self.dir.is_dir() {
            return Err(format!(
                "engine directory {} does not exist (set ENGINE_DIR)",
                self.dir.display()
            ));
        }
        if !self.python.is_file() {
            return Err(format!(
                "python interpreter {} does not exist (set ENGINE_PYTHON)",
                self.python.display()
            ));
        }

        Ok(EngineConfig {
            python: absolute(&self.python)?,
            dir: absolute(&self.dir)?,
            // Absolute for the same reason as the others: the engine runs with a
            // different working directory, so a relative data root would resolve
            // against `dir` and silently create a second, empty tree.
            data_dir: absolute(&if self.data_dir.is_absolute() {
                self.data_dir.clone()
            } else {
                self.dir.join(&self.data_dir)
            })?,
            ..self.clone()
        })
    }
}

/// Make `path` absolute **without following symlinks**.
///
/// `canonicalize` is the wrong tool here and silently breaks the venv: a
/// virtualenv's `bin/python` is a symlink to the system interpreter, and Python
/// derives `sys.prefix` from the path it was invoked by. Resolving the link
/// hands over `/usr/bin/python3.14`, which has no idea it is meant to be in a
/// venv, so `-m uvicorn` fails with ModuleNotFoundError against a venv where
/// uvicorn is plainly installed.
fn absolute(path: &std::path::Path) -> Result<PathBuf, String> {
    std::path::absolute(path).map_err(|e| format!("cannot resolve {}: {e}", path.display()))
}

fn env_or(key: &str, fallback: String) -> String {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty()).unwrap_or(fallback)
}

fn parse_env<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok()?.trim().parse().ok()
}
