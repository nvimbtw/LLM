//! The `tools::SearchCache` `web-search` checks before paying for a live
//! search, backed by the data engine's `/search-cache` route.
//!
//! Lives here rather than in `tools`, which stays the leaf
//! `docs/backend/README.md` documents it as — knowing nothing about this
//! project's other services. `api` already depends on both `tools` and
//! `data-engine`, so the wiring belongs where both are already in scope.

use std::time::Duration;

use async_trait::async_trait;
use auth::Caller;
use data_engine::Proxy;
use serde_json::json;
use tools::SearchCache;

/// A search result is a fact about the world, not about whoever happened to
/// ask first — every call is made as the backend acting for itself
/// (`Caller::System`), not on behalf of the person whose turn triggered it.
/// The data engine already has a name for exactly this: `"system"`, the
/// `act` a route sees when nothing more specific minted the token.
pub struct DataEngineCache {
    data: Proxy,
}

impl DataEngineCache {
    pub fn new(data: Proxy) -> DataEngineCache {
        DataEngineCache { data }
    }
}

#[async_trait]
impl SearchCache for DataEngineCache {
    async fn get(&self, key: &str) -> Option<(String, Duration)> {
        let body = self
            .data
            .post(&Caller::System, "/search-cache/lookup", json!({ "key": key }))
            .await
            .ok()?;

        let content = body.get("content")?.as_str()?.to_string();
        let created_at = body.get("created_at")?.as_str()?;
        let age = age_of(created_at)?;
        Some((content, age))
    }

    async fn put(&self, key: &str, body: &str) {
        // Best-effort, per the trait's own contract: the search already
        // succeeded, and a person must not see a warning because this write
        // did not land.
        let _ = self
            .data
            .post(&Caller::System, "/search-cache", json!({ "key": key, "content": body }))
            .await;
    }
}

/// How long ago an RFC 3339 timestamp the data engine sent was.
///
/// `None` for anything that fails to parse or lands in the future — a clock
/// far enough out of step to make either happen means the timestamp cannot be
/// trusted, and the caller treats that exactly like a miss rather than
/// trusting a negative age into "always fresh."
fn age_of(timestamp: &str) -> Option<Duration> {
    let then = chrono::DateTime::parse_from_rfc3339(timestamp).ok()?.to_utc();
    (chrono::Utc::now() - then).to_std().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_timestamp_is_a_small_age() {
        let now = chrono::Utc::now().to_rfc3339();
        let age = age_of(&now).expect("parses");
        assert!(age < Duration::from_secs(5), "expected a small age, got {age:?}");
    }

    #[test]
    fn a_week_old_timestamp_is_about_a_week() {
        let then = chrono::Utc::now() - chrono::Duration::days(7);
        let age = age_of(&then.to_rfc3339()).expect("parses");
        let week = Duration::from_secs(7 * 86_400);
        assert!(age.as_secs().abs_diff(week.as_secs()) < 5);
    }

    #[test]
    fn garbage_is_a_miss_not_a_panic() {
        assert!(age_of("not a timestamp").is_none());
    }

    #[test]
    fn a_future_timestamp_is_not_trusted_as_negative_age() {
        let ahead = (chrono::Utc::now() + chrono::Duration::days(1)).to_rfc3339();
        assert!(age_of(&ahead).is_none());
    }
}
