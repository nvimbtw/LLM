//! What a model can reach for, and what comes back.
//!
//! A tool is named by the marker the model emits: `<|web-search|>query<|/web-search|>`
//! is the tool named `web-search`, and the engine reports the call under that
//! name. Nothing here parses markers — by the time a call arrives it is already
//! a `(name, argument)` pair.
//!
//! Adding a tool is one `impl Tool` and one line in `Registry::from_env`, plus
//! the two reserved slots the engine needs to spell it (`llm/chat/tools.py`).

pub mod config;
pub mod error;
pub mod markup;
mod web_search;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

pub use config::Config;
pub use error::{Result, ToolError};

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
    async fn run(&self, argument: &str) -> Result<String>;
}

#[derive(Clone)]
pub struct Registry {
    tools: HashMap<&'static str, Arc<dyn Tool>>,
}

impl Registry {
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Registry {
        Registry {
            tools: tools.into_iter().map(|tool| (tool.name(), tool)).collect(),
        }
    }

    /// The tools this machine is configured for.
    ///
    /// A tool whose credentials are missing is left out entirely rather than
    /// registered and failing: chat has to work on a checkout with no API keys,
    /// and `answer` turns the resulting `Unknown` into a caveat the model reads.
    pub fn from_env() -> Registry {
        Registry::with_config(&Config::from_env())
    }

    pub fn with_config(config: &Config) -> Registry {
        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
        if let Some(search) = web_search::BraveSearch::new(config) {
            tools.push(Arc::new(search));
        }
        Registry::new(tools)
    }

    pub fn names(&self) -> Vec<&'static str> {
        let mut names: Vec<_> = self.tools.keys().copied().collect();
        names.sort_unstable();
        names
    }

    pub async fn run(&self, name: &str, argument: &str) -> Result<String> {
        match self.tools.get(name) {
            Some(tool) => tool.run(argument).await,
            None => Err(ToolError::Unknown(name.to_string())),
        }
    }

    /// `run`, with every failure turned into something the model can read.
    ///
    /// A turn that has already been generated is waiting on this, and a reply
    /// the person never sees because a search 429'd is a worse outcome than one
    /// that says the search failed. The caller gets a body either way; the error
    /// is returned alongside it so it can still be logged.
    pub async fn answer(&self, name: &str, argument: &str) -> (String, Option<ToolError>) {
        match self.run(name, argument).await {
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
        async fn run(&self, argument: &str) -> Result<String> {
            Err(ToolError::Status { code: 429, detail: argument.to_string() })
        }
    }

    #[test]
    fn a_registry_with_no_key_registers_no_search() {
        let empty = Registry::with_config(&Config::default());
        assert!(empty.names().is_empty());
    }

    #[test]
    fn a_key_is_all_web_search_needs_to_appear() {
        let configured = Registry::with_config(&Config {
            search_api_key: Some("k".to_string()),
            ..Config::default()
        });
        assert_eq!(configured.names(), vec!["web-search"]);
    }

    #[tokio::test]
    async fn a_tool_that_is_not_there_is_a_caveat_rather_than_a_lost_reply() {
        let registry = Registry::new(vec![]);
        let (body, failure) = registry.answer("web-search", "anything").await;
        assert!(body.starts_with("<|warn|>"));
        assert!(matches!(failure, Some(ToolError::Unknown(_))));
    }

    #[tokio::test]
    async fn a_refusal_reaches_the_model_as_markup_it_can_read() {
        let registry = Registry::new(vec![Arc::new(Stub)]);
        let (body, failure) = registry.answer("stub", "why").await;
        assert!(body.contains("429"));
        assert!(matches!(failure, Some(ToolError::Status { code: 429, .. })));
    }
}
