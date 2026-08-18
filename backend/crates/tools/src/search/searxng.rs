//! A self-hosted SearXNG instance. No key, no quota, no third party.
//!
//! SearXNG is a metasearch front end: it forwards a query to engines it is
//! configured for and merges what comes back. `deploy/searxng/` holds a compose
//! file and the settings this expects.
//!
//! **JSON output is off by default and the instance must be configured for it**
//! (`search.formats`), which is the one thing that goes wrong when standing one
//! up — see `refusal` below for how that failure is reported.

use async_trait::async_trait;
use serde::Deserialize;

use super::{client, first_line, transport, Hit, Provider};
use crate::config::Config;
use crate::error::{Result, ToolError};

const SEARCH_PATH: &str = "search";

pub struct SearxNG {
    http: reqwest::Client,
    base_url: String,
}

impl SearxNG {
    pub fn new(base_url: &str, config: &Config) -> Option<SearxNG> {
        Some(SearxNG {
            http: client(config.timeout)?,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }
}

#[async_trait]
impl Provider for SearxNG {
    fn origin(&self) -> String {
        format!("SearXNG at {}", self.base_url)
    }

    async fn find(&self, query: &str, _limit: usize) -> Result<Vec<Hit>> {
        // No count parameter: SearXNG returns a whole merged page and the
        // caller takes what it needs. Asking for a page size it does not have
        // would be a silently ignored argument.
        let response = self
            .http
            .get(format!("{}/{SEARCH_PATH}", self.base_url))
            .query(&[("q", query), ("format", "json")])
            .send()
            .await
            .map_err(transport)?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(refusal(status.as_u16(), &body));
        }

        let found: Response =
            response.json().await.map_err(|e| ToolError::Decode(e.to_string()))?;
        Ok(hits(found))
    }
}

/// A refusal, with the cause named where it is knowable.
///
/// A bare 403 from SearXNG is almost always one of two configuration mistakes
/// rather than anything about the query, and both are invisible from the status
/// line alone. Saying so here saves the reader the same half hour twice, and
/// the text reaches the model as a `<|warn|>` — so it must read as a sentence.
fn refusal(code: u16, body: &str) -> ToolError {
    if code == 403 {
        return ToolError::Status {
            code,
            detail: "the instance refused the request — check that `json` is in \
                     `search.formats` and that `server.limiter` is off"
                .to_string(),
        };
    }
    ToolError::Status { code, detail: first_line(body) }
}

fn hits(found: Response) -> Vec<Hit> {
    // Answers first. A direct answer is the one thing on the page that is
    // already the shape a small model needs, and burying it under three link
    // summaries wastes the position it has the best chance of reading.
    found
        .answers
        .into_iter()
        .map(|a| Hit { title: String::new(), snippet: a.text() })
        .chain(found.results.into_iter().map(|r| Hit { title: r.title, snippet: r.content }))
        .collect()
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    results: Vec<Row>,
    #[serde(default)]
    answers: Vec<Answer>,
}

#[derive(Deserialize)]
struct Row {
    #[serde(default)]
    title: String,
    #[serde(default)]
    content: String,
}

/// SearXNG changed `answers` from a list of strings to a list of objects. Both
/// spellings are accepted rather than pinning a version, because the instance is
/// the operator's to upgrade and this side should not be what breaks when they
/// do.
#[derive(Deserialize)]
#[serde(untagged)]
enum Answer {
    Text(String),
    Object {
        #[serde(default)]
        answer: String,
    },
}

impl Answer {
    fn text(self) -> String {
        match self {
            Answer::Text(text) => text,
            Answer::Object { answer } => answer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> Vec<Hit> {
        hits(serde_json::from_str(body).expect("valid json"))
    }

    #[test]
    fn results_carry_their_title_and_content() {
        let found = parse(
            r#"{"results":[{"title":"Elbrus","content":"5,642 m","url":"http://x","engine":"ddg"}]}"#,
        );
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].title, "Elbrus");
        assert_eq!(found[0].snippet, "5,642 m");
    }

    #[test]
    fn a_direct_answer_is_put_before_the_links_it_beats() {
        let found = parse(
            r#"{"answers":[{"answer":"5,642 metres"}],
                "results":[{"title":"Elbrus","content":"a volcano"}]}"#,
        );
        assert_eq!(found[0].snippet, "5,642 metres");
        assert_eq!(found[1].title, "Elbrus");
    }

    #[test]
    fn answers_are_read_in_either_spelling_the_project_has_used() {
        assert_eq!(parse(r#"{"answers":["older form"]}"#)[0].snippet, "older form");
        assert_eq!(parse(r#"{"answers":[{"answer":"newer"}]}"#)[0].snippet, "newer");
    }

    #[test]
    fn a_response_with_neither_key_is_empty_rather_than_a_decode_error() {
        assert!(parse(r#"{"query":"x","number_of_results":0}"#).is_empty());
    }

    #[test]
    fn a_403_says_what_is_actually_wrong_with_the_instance() {
        let ToolError::Status { detail, .. } = refusal(403, "Forbidden") else {
            panic!("a status, not something else")
        };
        assert!(detail.contains("search.formats"));
        assert!(detail.contains("limiter"));
    }

    #[test]
    fn any_other_status_reports_what_the_instance_said() {
        let ToolError::Status { code, detail } = refusal(502, "upstream engine died\nstack…")
        else {
            panic!("a status, not something else")
        };
        assert_eq!(code, 502);
        assert_eq!(detail, "upstream engine died");
    }

    #[test]
    fn a_trailing_slash_on_the_configured_url_does_not_double_up() {
        let instance = SearxNG::new("http://localhost:8080/", &Config::default())
            .expect("a client builds");
        assert_eq!(instance.base_url, "http://localhost:8080");
    }
}
