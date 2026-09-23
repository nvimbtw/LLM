//! Where a Python service lives and how to start it.
//!
//! Every field comes from the environment because none of them are properties of
//! this project — they are properties of the machine it runs on. The variables
//! are the service's prefix plus a fixed suffix (`LLM_ENGINE_PORT`,
//! `DATA_ENGINE_PORT`), so adding a service adds a documented block to `.env`
//! rather than a new naming scheme.

use std::path::PathBuf;
use std::time::Duration;

/// What a service is called and where it lives when nothing says otherwise.
#[derive(Clone, Debug)]
pub struct Defaults {
    /// Addressed in the `aud` claim. A token minted for one service is refused
    /// by every other.
    pub audience: &'static str,
    /// Read for every variable: `<prefix>_PORT`, `<prefix>_DIR`, and so on.
    pub env_prefix: &'static str,
    pub dir: &'static str,
    pub port: u16,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub audience: String,
    /// Started as `<python> -m uvicorn`, so this is the only interpreter-shaped
    /// knob — there is no uvicorn path to keep in sync with it.
    pub python: PathBuf,
    pub dir: PathBuf,
    pub host: String,
    pub port: u16,
    /// Off means "a service is already running and I will manage it myself" —
    /// the normal setup when developing against `uvicorn --reload`.
    pub autostart: bool,
    /// Unset leaves both sides unauthenticated, which is the development pair.
    pub service_secret: Option<String>,
    /// The variable the child reads its secret from. Named here rather than
    /// derived at the spawn, so the child and the token agree by construction.
    pub secret_var: String,
    /// Anything else the child needs in its environment. The LLM engine's data root
    /// is the only current case, and it belongs to the LLM engine, not here.
    pub extra_env: Vec<(String, String)>,
    /// Importing torch is slow enough on a cold page cache that a short one
    /// would flap.
    pub startup_timeout: Duration,
}

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

impl Config {
    /// `<prefix>_PYTHON` resolves against `<prefix>_DIR` when relative, so moving
    /// the checkout needs one variable set rather than two.
    pub fn from_env(defaults: &Defaults) -> Config {
        let prefix = defaults.env_prefix;

        let dir = std::env::var(format!("{prefix}_DIR"))
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(defaults.dir));

        let python = match std::env::var(format!("{prefix}_PYTHON")) {
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

        let secret_var = format!("{prefix}_SERVICE_SECRET");
        Config {
            audience: defaults.audience.to_string(),
            python,
            dir,
            host: env_or(&format!("{prefix}_HOST"), DEFAULT_HOST.to_string()),
            port: parse_env(&format!("{prefix}_PORT")).unwrap_or(defaults.port),
            service_secret: std::env::var(&secret_var).ok().filter(|s| !s.is_empty()),
            secret_var,
            extra_env: Vec::new(),
            // Anything but an explicit falsey value keeps it on: a spare process
            // is cheaper than a server that boots green and fails every request.
            autostart: std::env::var(format!("{prefix}_AUTOSTART"))
                .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no"))
                .unwrap_or(true),
            startup_timeout: parse_env(&format!("{prefix}_STARTUP_TIMEOUT_SECS"))
                .map(Duration::from_secs)
                .unwrap_or(DEFAULT_STARTUP_TIMEOUT),
        }
    }

    /// Add a variable to the child's environment, resolving relative paths
    /// against the service's own directory.
    ///
    /// Relative is the trap: a path resolved against the *server's* working
    /// directory silently creates a second, empty tree beside the real one.
    pub fn with_env_path(mut self, key: &str, path: PathBuf) -> Config {
        let resolved = if path.is_absolute() {
            path
        } else {
            self.dir.join(path)
        };
        self.extra_env
            .push((key.to_string(), resolved.display().to_string()));
        self
    }

    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    /// Check the paths exist and return a copy with them made absolute.
    ///
    /// Absolute is not cosmetic: `Command` resolves the program path *after*
    /// chdir-ing into `current_dir`, so a relative `../llm-engine/.venv/bin/python`
    /// is looked for under `../llm-engine/../llm-engine/...` and fails with a bare ENOENT.
    pub fn resolved(&self) -> Result<Config, String> {
        let prefix = self.audience.to_uppercase().replace('-', "_");
        if !self.dir.is_dir() {
            return Err(format!(
                "{} directory {} does not exist (set {prefix}_DIR)",
                self.audience,
                self.dir.display()
            ));
        }
        if !self.python.is_file() {
            return Err(format!(
                "python interpreter {} does not exist (set {prefix}_PYTHON)",
                self.python.display()
            ));
        }

        let extra_env = self
            .extra_env
            .iter()
            .map(|(k, v)| {
                Ok((
                    k.clone(),
                    absolute(&PathBuf::from(v))?.display().to_string(),
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(Config {
            python: absolute(&self.python)?,
            dir: absolute(&self.dir)?,
            extra_env,
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
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or(fallback)
}

fn parse_env<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok()?.trim().parse().ok()
}
