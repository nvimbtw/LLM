//! Producing candidate replies that become preference data in `raw/preference/`.
//!
//! Two strategies for one job, which is why they share a crate. `autolabel`
//! generates several candidates per prompt and lets `llm::autolabel::Judge` pick
//! a best and a worst by rule; `gather` generates one reply per prompt and hands
//! it to a person. The rule-based judge scores surface properties and is
//! deliberately independent of taste; a person rating a reply *is* the taste.
//! They produce different data on purpose and both land in the same place.
//!
//! Below `db` in the dependency order: neither module touches sqlx, so the
//! thresholds in `gather` are pure functions the callers apply to counts they
//! fetched themselves.

pub mod autolabel;
pub mod gather;
