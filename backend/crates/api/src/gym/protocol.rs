//! Interpreted gym records and stateless generation requests.

use crate::streaming::{envelope, string_at};
use serde_json::{json, Value};

const DEFAULT_ROUNDS: usize = 8;
const DEFAULT_REPLY_TOKENS: u32 = 256;

pub(super) struct Episode {
    pub id: String,
    pub prompt: String,
    system: String,
    status: String,
    task_type: String,
    index: Value,
    seed: Value,
    no_tools: bool,
}

impl Episode {
    /// Interpret the fields Rust owns, retaining legacy defaults for optional or malformed service fields.
    pub fn from_value(value: &Value) -> Self {
        Self {
            id: string_at(value, "id"),
            prompt: string_at(value, "prompt"),
            system: string_at(value, "system"),
            status: string_at(value, "status"),
            task_type: string_at(value, "task_type"),
            index: value["index"].clone(),
            seed: value["seed"].clone(),
            no_tools: value["tools"].as_array().is_some_and(Vec::is_empty),
        }
    }

    /// Report whether the stored episode is eligible to run again; completed and unknown statuses are skipped.
    pub fn runnable(&self) -> bool {
        matches!(self.status.as_str(), "pending" | "running")
    }

    /// Choose one terminal round for a tool-free episode or a positive configured ceiling for a shell episode.
    pub fn rounds(&self) -> usize {
        if self.no_tools {
            return 1;
        }
        std::env::var("MAX_GYM_ROUNDS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|rounds| *rounds > 0)
            .unwrap_or(DEFAULT_ROUNDS)
    }

    /// Build the episode-start frame with its attribution and task description.
    pub fn start_frame(&self) -> String {
        envelope(
            "episode_start",
            &json!({"episode_id":self.id,"index":self.index,
            "task_type":self.task_type,"seed":self.seed,"prompt":self.prompt,"system":self.system}),
        )
        .to_string()
    }

    /// Build a stateless round using the run's model and task system; client sampling remains opaque.
    /// A terminal round masks tool calls even for engines that would otherwise continue indefinitely.
    pub fn generation(&self, model: &str, turns: &[Value], ask: &Value, terminal: bool) -> Value {
        let mut body = json!({"model_name":model,"system":self.system,"turns":turns,"max_tokens":DEFAULT_REPLY_TOKENS});
        if terminal {
            body["max_tool_calls"] = json!(0)
        }
        for key in [
            "temperature",
            "top_k",
            "top_p",
            "max_tokens",
            "budget",
            "seed",
            "context_multiplier",
        ] {
            if let Some(value) = ask.get(key) {
                body[key] = value.clone()
            }
        }
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The run fixes the model/system while sampling passes through and terminal generation forbids tools.
    #[test]
    fn generation_keeps_authoritative_task_settings() {
        let episode = Episode::from_value(&json!({"id":"e","system":"task","tools":[]}));
        let body = episode.generation("run-model", &[], &json!({"model_name":"other","system":"other","temperature":0.7,"context_multiplier":4}), true);
        assert_eq!(body["model_name"], "run-model");
        assert_eq!(body["system"], "task");
        assert_eq!(body["max_tool_calls"], 0);
        assert_eq!(body["context_multiplier"], 4);
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(episode.rounds(), 1);
    }

    /// Only pending and running episodes execute, including records without a tool policy from older services.
    #[test]
    fn runnable_statuses_match_the_stored_run_contract() {
        for status in ["pending", "running", "done", "abandoned", "unknown"] {
            assert_eq!(
                Episode::from_value(&json!({"status":status})).runnable(),
                matches!(status, "pending" | "running")
            );
        }
    }
}
