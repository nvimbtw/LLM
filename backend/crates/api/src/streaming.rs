//! Shared wire vocabulary for chat, gym and job streams.

use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ToolCall {
    pub tool: String,
    pub argument: String,
}

impl ToolCall {
    /// Interpret a reported call, retaining the legacy empty-string fallback for absent or non-string fields.
    pub fn from_value(value: &Value) -> Self {
        Self {
            tool: string_at(value, "tool"),
            argument: string_at(value, "argument"),
        }
    }

    /// Build the client-visible result for this call and its rendered observation body.
    pub fn result(&self, body: String) -> Value {
        json!({ "tool": self.tool, "argument": self.argument, "body": body })
    }
}

/// Flatten an opaque object onto a client frame; existing payload fields retain their precedence.
pub fn envelope(kind: &str, data: &Value) -> Value {
    let mut payload = json!({ "type": kind });
    if let (Some(target), Some(source)) = (payload.as_object_mut(), data.as_object()) {
        target.extend(source.clone());
    }
    payload
}

/// Read a string from an interpreted legacy payload, treating absent or non-string values as empty.
pub fn string_at(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unknown service fields survive flattening and keep the historical precedence of payload keys.
    #[test]
    fn opaque_fields_and_payload_precedence_survive() {
        assert_eq!(
            envelope("progress", &json!({"type":"extension", "nested":{"x":1}})),
            json!({"type":"extension", "nested":{"x":1}})
        );
        assert_eq!(envelope("done", &Value::Null), json!({"type":"done"}));
    }

    /// Missing and wrongly typed legacy call fields remain safe empty-string inputs to the registry.
    #[test]
    fn invalid_call_fields_keep_the_legacy_fallback() {
        let call = ToolCall::from_value(&json!({"tool":7}));
        assert_eq!(
            call,
            ToolCall {
                tool: String::new(),
                argument: String::new()
            }
        );
    }
}
