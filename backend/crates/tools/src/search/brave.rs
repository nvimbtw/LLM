//! The Brave Search API. Hosted, keyed, metered.

use async_trait::async_trait;
use serde::Deserialize;

use super::{client, first_line, transport, Hit, Provider};
use crate::config::Config;
use crate::error::{Result, ToolError};

const ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";
const KEY_HEADER: &str = "X-Subscription-Token";

pub struct Brave {
    http: reqwest::Client,
    key: String,
}

impl Brave {
    pub fn new(key: &str, config: &Config) -> Option<Brave> {
        Some(Brave { http: client(config.timeout)?, key: key.to_string() })
    }
}

#[async_trait]
impl Provider for Brave {
    fn origin(&self) -> String {
        "Brave Search".to_string()
    }

    async fn find(&self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let response = self
            .http
            .get(ENDPOINT)
            .header(KEY_HEADER, &self.key)
            .query(&[("q", query), ("count", &limit.to_string())])
            .send()
            .await
            .map_err(transport)?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            return Err(ToolError::Status { code: status.as_u16(), detail: first_line(&detail) });
        }

        let found: Response =
            response.json().await.map_err(|e| ToolError::Decode(e.to_string()))?;
        Ok(found
            .web
            .results
            .into_iter()
            .map(|r| Hit {
                title: r.title.unwrap_or_default(),
                snippet: r.description.unwrap_or_default(),
            })
            .collect())
    }
}

/// Only what the model is shown. Anything else Brave returns is a field this
/// build would have to decide how to render, and there is no marker for it.
#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    web: Web,
}

#[derive(Deserialize, Default)]
struct Web {
    #[serde(default)]
    results: Vec<Row>,
}

/// `Option` rather than `#[serde(default)] String` for the reason spelled out in
/// `searxng::Row`: a field that arrives as an explicit `null` is a decode error
/// against a plain `String`, and the default only covers an absent key.
#[derive(Deserialize)]
struct Row {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    description: Option<String>,
}
