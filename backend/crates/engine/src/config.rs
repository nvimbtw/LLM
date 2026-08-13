//! Where the Python engine lives and how to start it.
//!
//! Every field comes from the environment because none of them are properties
//! of this project — they are properties of the machine it runs on. The table
//! in `backend/CLAUDE.md` lists them.

use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Started as `<python> -m uvicorn`, so this is the only interpreter-shaped
    /// knob — there is no uvicorn path to keep in sync with it.
    pub python: PathBuf,
    pub dir: PathBuf,
    pub host: String,
    pub port: u16,
    /// Off means "an engine is already running and I will manage it myself" —
    /// the normal setup when developing against `uvicorn --reload`.
    pub autostart: bool,
    /// Handed to the engine as `DATA_DIR`. Relative paths resolve against `dir`.
    pub data_dir: PathBuf,
    /// Unset leaves both sides unauthenticated, which is the development pair.
    pub service_secret: Option<String>,
    /// Importing torch is slow enough on a cold page cache that a short one
    /// would flap.
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
            service_secret: None,
            startup_timeout: Duration::from_secs(60),
        }
    }
}

impl EngineConfig {
    /// `ENGINE_PYTHON` resolves against `ENGINE_DIR` when relative, so moving
    /// the checkout needs one variable set rather than two.
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

        // Computed before `dir` moves into the struct.
        let data_dir = std::env::var("DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dir.join("data"));

        EngineConfig {
            python,
            dir,
            data_dir,
            host: env_or("ENGINE_HOST", defaults.host),
            port: parse_env("ENGINE_PORT").unwrap_or(defaults.port),
            service_secret: std::env::var("ENGINE_SERVICE_SECRET")
                .ok()
                .filter(|s| !s.is_empty()),
            // Anything but an explicit falsey value keeps it on: a spare
            // process is cheaper than a server that boots green and fails
            // every request.
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

    /// Check the paths exist and return a copy with them made absolute.
    ///
    /// Absolute is not cosmetic: `Command` resolves the program path *after*
    /// chdir-ing into `current_dir`, so a relative `../engine/.venv/bin/python`
    /// is looked for under `../engine/../engine/...` and fails with a bare
    /// ENOENT.
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
            // A relative data root would resolve against the engine's own
            // working directory and silently create a second, empty tree.
            data_dir: absolute(&if self.data_dir.is_absolute() {
                self.data_dir.clone()
            } else {
                self.dir.join(&self.data_dir)
            })?,
            ..self.clone()
        })
    }
}

/// Absolute **without following symlinks**. `canonicalize` silently breaks the
/// venv: `bin/python` is a symlink to the system interpreter, and Python derives
/// `sys.prefix` from the path it was invoked by — so resolving it hands over an
/// interpreter that fails `-m uvicorn` against a venv where uvicorn is installed.
fn absolute(path: &std::path::Path) -> Result<PathBuf, String> {
    std::path::absolute(path).map_err(|e| format!("cannot resolve {}: {e}", path.display()))
}

fn env_or(key: &str, fallback: String) -> String {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty()).unwrap_or(fallback)
}

fn parse_env<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok()?.trim().parse().ok()
}
