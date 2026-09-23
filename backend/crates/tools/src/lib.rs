//! What a model can reach for, and what comes back.
//!
//! A tool is named inside the call the model emits —
//! `<|tool-call|>{"tool": "web-search", "argument": "…"}<|/tool-call|>` is the
//! tool named `web-search`. Nothing here parses markers or JSON: by the time a
//! call arrives the LLM engine has already made it a `(name, argument)` pair, and a
//! body it could not read never became a call at all.
//!
//! Adding a tool is one `impl Tool` and one line in `Registry::with_config`,
//! plus the two reserved slots the LLM engine needs to spell it
//! (`llm/chat/tools.py`).

mod bash;
pub mod config;
pub mod error;
pub mod markup;
mod search;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

pub use bash::{observation, Bash, Output};
pub use config::Config;
pub use error::{Result, ToolError};
pub use search::WebSearch;

#[async_trait]
pub trait Tool: Send + Sync {
    /// The marker stem the model calls it by.
    fn name(&self) -> &'static str;

    /// Answer `argument`, rendered in the marker vocabulary.
    ///
    /// The returned string is the *inside* of a `<|tool-result|>` block — the
    /// engine adds the delimiters and repeats the call above it. Everything a
    /// tool puts here has been through `markup::neutralize`, because it is text
    /// from outside going into a prompt.
    ///
    /// `session` is an opaque key for "the conversation this call belongs to",
    /// and most tools ignore it: a search is the same search whoever asked. A
    /// sandboxed shell is not — a reply that writes a file in one round has to
    /// find it there in the next — so the key is passed to every tool rather
    /// than bolted onto the one that needs it. It is a key and not a chat: what
    /// conversation it stands for, and who owns it, stays in `api`.
    async fn run(&self, argument: &str, session: &str) -> Result<String>;
}

/// Where a tool that fetches the same thing twice can go instead of fetching
/// it twice — a persistent store this crate does not own or know the shape
/// of. `WebSearch` is the one caller today: `key` is a normalized query, the
/// stored value its rendered result.
///
/// A trait rather than a concrete type because the store lives behind the
/// data engine, and `tools` stays the leaf it is documented as — it knows
/// about the outside world and the marker vocabulary, nothing about this
/// project's other services. The concrete implementation is wired in from
/// `api`, which already depends on both.
#[async_trait]
pub trait SearchCache: Send + Sync {
    /// The freshest entry under `key` and how long ago it was written, or
    /// `None` for a miss.
    ///
    /// Age rather than a timestamp: the only question a caller ever asks is
    /// "is this fresh enough," and answering in age keeps every
    /// implementation's clock — and clock skew between this process and
    /// wherever the store lives — out of the question entirely.
    async fn get(&self, key: &str) -> Option<(String, std::time::Duration)>;

    /// Record `body` under `key`.
    ///
    /// Best-effort by convention: a caller that cannot reach the store should
    /// swallow the failure rather than fail the tool call over it — the
    /// search itself already succeeded, and a person must not see a warning
    /// because a cache write did not land.
    async fn put(&self, key: &str, body: &str);
}

#[derive(Clone)]
pub struct Registry {
    tools: HashMap<&'static str, Arc<dyn Tool>>,
    origins: Vec<String>,
}

impl Registry {
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Registry {
        Registry {
            origins: tools.iter().map(|tool| tool.name().to_string()).collect(),
            tools: tools.into_iter().map(|tool| (tool.name(), tool)).collect(),
        }
    }

    /// The tools this machine is configured for.
    ///
    /// A tool whose credentials are missing is left out entirely rather than
    /// registered and failing: chat has to work on a checkout with no API keys,
    /// and `answer` turns the resulting `Unknown` into a caveat the model reads.
    ///
    /// `cache` is `web-search`'s, if the caller has one wired up — `None` runs
    /// every search live, exactly as before the cache existed.
    pub fn from_env(cache: Option<Arc<dyn SearchCache>>) -> Registry {
        Registry::with_config(&Config::from_env(), cache)
    }

    pub fn with_config(config: &Config, cache: Option<Arc<dyn SearchCache>>) -> Registry {
        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
        let mut origins = Vec::new();
        if let Some(search) = WebSearch::from_config(config, cache) {
            origins.push(format!("{} ({})", search.name(), search.origin()));
            tools.push(Arc::new(search));
        }
        if let Some(bash) = Bash::from_config(config) {
            origins.push(format!("{} ({})", bash.name(), bash.origin()));
            tools.push(Arc::new(bash));
        }
        Registry {
            tools: tools.iter().map(|t| (t.name(), t.clone())).collect(),
            origins,
        }
    }

    pub fn names(&self) -> Vec<&'static str> {
        let mut names: Vec<_> = self.tools.keys().copied().collect();
        names.sort_unstable();
        names
    }

    /// Each tool and where its answers come from, for the line at startup. A
    /// registry that merely names `web-search` does not say whether that is a
    /// third party or a box in the next room, which is the thing worth seeing.
    pub fn origins(&self) -> &[String] {
        &self.origins
    }

    pub async fn run(&self, name: &str, argument: &str, session: &str) -> Result<String> {
        match self.tools.get(name) {
            Some(tool) => tool.run(argument, session).await,
            None => Err(ToolError::Unknown(name.to_string())),
        }
    }

    /// `run`, with every failure turned into something the model can read.
    ///
    /// A turn that has already been generated is waiting on this, and a reply
    /// the person never sees because a search 429'd is a worse outcome than one
    /// that says the search failed. The caller gets a body either way; the error
    /// is returned alongside it so it can still be logged.
    pub async fn answer(
        &self,
        name: &str,
        argument: &str,
        session: &str,
    ) -> (String, Option<ToolError>) {
        match self.run(name, argument, session).await {
            Ok(body) => (body, None),
            Err(e) => (markup::warning(&e.to_string()), Some(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Stub;

    #[async_trait]
    impl Tool for Stub {
        fn name(&self) -> &'static str {
            "stub"
        }
        async fn run(&self, argument: &str, _session: &str) -> Result<String> {
            Err(ToolError::Status {
                code: 429,
                detail: argument.to_string(),
            })
        }
    }

    #[test]
    fn a_registry_with_no_key_registers_no_search() {
        let empty = Registry::with_config(&Config::default(), None);
        assert!(empty.names().is_empty());
    }

    #[test]
    fn a_key_is_all_web_search_needs_to_appear() {
        let configured = Registry::with_config(
            &Config {
                search_api_key: Some("k".to_string()),
                ..Config::default()
            },
            None,
        );
        assert_eq!(configured.names(), vec!["web-search"]);
    }

    #[test]
    fn each_tool_appears_on_its_own_configuration_and_not_the_others() {
        let sandbox_only = Registry::with_config(
            &Config {
                bash_url: Some("http://localhost:8899".to_string()),
                ..Config::default()
            },
            None,
        );
        assert_eq!(sandbox_only.names(), vec!["bash"]);

        let both = Registry::with_config(
            &Config {
                search_api_key: Some("k".to_string()),
                bash_url: Some("http://localhost:8899".to_string()),
                ..Config::default()
            },
            None,
        );
        assert_eq!(both.names(), vec!["bash", "web-search"]);
        assert_eq!(both.origins().len(), 2);
    }

    #[tokio::test]
    async fn a_tool_that_is_not_there_is_a_caveat_rather_than_a_lost_reply() {
        let registry = Registry::new(vec![]);
        let (body, failure) = registry.answer("web-search", "anything", "chat").await;
        assert!(body.starts_with("<|warn|>"));
        assert!(matches!(failure, Some(ToolError::Unknown(_))));
    }

    #[tokio::test]
    async fn a_refusal_reaches_the_model_as_markup_it_can_read() {
        let registry = Registry::new(vec![Arc::new(Stub)]);
        let (body, failure) = registry.answer("stub", "why", "chat").await;
        assert!(body.contains("429"));
        assert!(matches!(failure, Some(ToolError::Status { code: 429, .. })));
    }
}
