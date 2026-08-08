//! The chat wire types.
//!
//! These lived in `llm::chat_template` when rendering happened in this
//! workspace. Rendering is the engine's job now, so what remains on this side is
//! the vocabulary the two ends agree on — a conversation, a role, a reasoning
//! budget, and the sampling knobs.
//!
//! They live here rather than in a crate of their own because this is the crate
//! that serializes them: every one of these crosses the HTTP boundary, and a
//! type whose only purpose is to be sent somewhere belongs with the thing that
//! sends it.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub role: Role,
    pub text: String,
}

/// How much reasoning the assistant should show before answering.
///
/// Three levels rather than a token count, because the label has to be something
/// a prompt can ask a teacher for, a person can pick off a menu, and a model can
/// condition on — and a model asked for "173 thinking tokens" cannot count its
/// own tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkBudget {
    Low,
    Med,
    High,
}

impl ThinkBudget {
    pub const ALL: [ThinkBudget; 3] = [ThinkBudget::Low, ThinkBudget::Med, ThinkBudget::High];

    pub fn as_str(self) -> &'static str {
        match self {
            ThinkBudget::Low => "low",
            ThinkBudget::Med => "med",
            ThinkBudget::High => "high",
        }
    }

    /// Parses the wire/on-disk spelling. `None` for anything else, including
    /// "auto" — the absence of a budget is a real state, which the caller
    /// expresses as `Option::None` rather than as a fourth variant.
    pub fn parse(s: &str) -> Option<ThinkBudget> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Some(ThinkBudget::Low),
            "med" | "medium" => Some(ThinkBudget::Med),
            "high" => Some(ThinkBudget::High),
            _ => None,
        }
    }
}

// The engine's own defaults, restated here only because callers construct
// `SamplingParams` directly. Each filter is disabled by its own neutral value —
// temperature 0.0 collapses to argmax, a top_k past the vocabulary keeps
// everything, a top_p of 1.0 cuts nothing.
pub const DEFAULT_TEMPERATURE: f32 = 0.8;
pub const DEFAULT_TOP_K: u32 = 40;
pub const DEFAULT_TOP_P: f32 = 0.95;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: DEFAULT_TEMPERATURE,
            top_k: DEFAULT_TOP_K,
            top_p: DEFAULT_TOP_P,
        }
    }
}
