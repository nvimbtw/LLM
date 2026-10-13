//! Pure interpretation of chat request and inference payloads.

use db::conversation::{Message, NewMessage};
use serde_json::{json, Value};
use uuid::Uuid;

use super::{failure, Round};

/// Require a model name and parse the optional request UUID, generating an identity only when absent.
pub(super) fn request_id(ask: &Value) -> Result<Uuid, service::Error> {
    ask["model_name"]
        .as_str()
        .ok_or_else(|| failure(400, "model_name is required"))?;
    match ask.get("request_id") {
        Some(value) => value
            .as_str()
            .and_then(|value| Uuid::parse_str(value).ok())
            .ok_or_else(|| failure(400, "invalid request_id")),
        None => Ok(Uuid::new_v4()),
    }
}

/// Parse attachment UUIDs and enforce the five-file turn limit without reading or binding storage.
pub(super) fn attachment_ids(ask: &Value) -> Result<Vec<Uuid>, service::Error> {
    let files: Vec<Uuid> =
        serde_json::from_value(ask.get("attachment_ids").cloned().unwrap_or(json!([])))
            .map_err(|_| failure(400, "invalid attachments"))?;
    if files.len() > 5 {
        return Err(failure(400, "at most five attachments are allowed"));
    }
    Ok(files)
}

/// Apply a known message floor to ordered history; an absent, malformed or unknown floor leaves history intact.
pub(super) fn apply_history_floor(history: &mut Vec<Message>, ask: &Value) {
    let Some(floor) = ask["history_floor"]
        .as_str()
        .and_then(|value| Uuid::parse_str(value).ok())
    else {
        return;
    };
    if let Some(index) = history.iter().position(|message| message.id == floor) {
        history.drain(..index);
    }
}

/// Validate one or two prepared user/tool messages, assigning the shared request identity and canonical counts.
pub(super) fn new_inputs(
    turns: &[Value],
    request: Uuid,
) -> Result<Vec<NewMessage>, service::Error> {
    if turns.is_empty() || turns.len() > 2 {
        return Err(failure(502, "invalid prepared turn count"));
    }
    turns
        .iter()
        .map(|turn| {
            let role = turn["role"]
                .as_str()
                .filter(|role| matches!(*role, "user" | "tool"))
                .ok_or_else(|| failure(502, "invalid prepared role"))?;
            let content = turn["text"]
                .as_str()
                .ok_or_else(|| failure(502, "missing prepared text"))?;
            Ok(NewMessage {
                role: role.into(),
                content: content.into(),
                tokens: token_count(&turn["tokens"])?,
                model_id: None,
                model_name: None,
                tool: None,
                argument: None,
                request_id: request,
            })
        })
        .collect()
}

/// Build stateless inference history, retaining stored tool metadata and canonical prepared message text.
pub(super) fn inference_turns(history: &[Message], inputs: &[NewMessage]) -> Vec<Value> {
    history
        .iter()
        .map(|message| {
            json!({"role":message.role, "text":message.content,
        "tool":message.tool, "argument":message.argument})
        })
        .chain(
            inputs
                .iter()
                .map(|input| json!({"role":input.role, "text":input.content})),
        )
        .collect()
}

/// Replace initial-message fields with authorized turns while preserving opaque generation settings.
pub(super) fn inference_config(mut ask: Value, turns: Vec<Value>) -> Result<Value, service::Error> {
    let object = ask
        .as_object_mut()
        .ok_or_else(|| failure(400, "expected an object"))?;
    object.insert("turns".into(), json!(turns));
    for key in [
        "text",
        "tool_result",
        "interject",
        "attachment_ids",
        "history_floor",
        "request_id",
    ] {
        object.remove(key);
    }
    Ok(ask)
}

/// Interpret a completed reply as an assistant message with the round's model snapshot and request identity.
pub(super) fn assistant_input(round: &Round, reply: &Value) -> Result<NewMessage, service::Error> {
    Ok(NewMessage {
        role: "assistant".into(),
        content: reply["text"]
            .as_str()
            .ok_or_else(|| failure(502, "missing inference text"))?
            .into(),
        tokens: token_count(&reply["generated_tokens"])?,
        model_id: round.model,
        model_name: round.model_name.clone(),
        tool: None,
        argument: None,
        request_id: round.request,
    })
}

/// Accept only nonnegative integral counts representable in PostgreSQL's signed message-token column.
fn token_count(value: &Value) -> Result<i32, service::Error> {
    value
        .as_u64()
        .and_then(|count| i32::try_from(count).ok())
        .ok_or_else(|| failure(502, "invalid token count from inference"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Invalid request identities fail rather than silently replacing the idempotency key.
    #[test]
    fn request_identity_requires_a_model_and_valid_uuid() {
        let id = Uuid::new_v4();
        assert_eq!(
            request_id(&json!({"model_name":"tiny", "request_id":id})).unwrap(),
            id
        );
        assert!(request_id(&json!({"model_name":"tiny"})).is_ok());
        for ask in [
            json!({}),
            json!({"model_name":1}),
            json!({"model_name":"tiny", "request_id":null}),
            json!({"model_name":"tiny", "request_id":"bad"}),
        ] {
            assert!(matches!(
                request_id(&ask),
                Err(service::Error::Status { code: 400, .. })
            ));
        }
    }

    /// Attachment parsing preserves empty/default behavior and rejects invalid IDs or over-budget batches.
    #[test]
    fn attachments_are_typed_and_bounded() {
        assert!(attachment_ids(&json!({})).unwrap().is_empty());
        assert_eq!(
            attachment_ids(&json!({"attachment_ids":vec![Uuid::nil();5]}))
                .unwrap()
                .len(),
            5
        );
        for files in [Value::Null, json!(["bad"]), json!(vec![Uuid::nil(); 6])] {
            assert!(matches!(
                attachment_ids(&json!({"attachment_ids":files})),
                Err(service::Error::Status { code: 400, .. })
            ));
        }
    }

    /// Prepared messages reject unexpected roles, missing text and counts outside the storage contract.
    #[test]
    fn prepared_inputs_validate_the_service_contract() {
        let request = Uuid::new_v4();
        let inputs = new_inputs(
            &[
                json!({"role":"tool","text":"body","tokens":0}),
                json!({"role":"user","text":"wait","tokens":3}),
            ],
            request,
        )
        .unwrap();
        assert_eq!(inputs[0].request_id, request);
        assert_eq!(inputs[1].role, "user");
        assert!(new_inputs(&[], request).is_err());
        assert!(new_inputs(&vec![json!({}); 3], request).is_err());
        for turn in [
            json!({"role":"assistant","text":"body","tokens":1}),
            json!({"role":"user","tokens":1}),
            json!({"role":"user","text":"body","tokens":-1}),
            json!({"role":"user","text":"body","tokens":2147483648_u64}),
        ] {
            assert!(matches!(
                new_inputs(&[turn], request),
                Err(service::Error::Status { code: 502, .. })
            ));
        }
    }

    /// Counts preserve the signed database boundary and reject fractions, negative values and missing fields.
    #[test]
    fn token_counts_match_the_storage_range() {
        assert_eq!(token_count(&json!(i32::MAX)).unwrap(), i32::MAX);
        assert_eq!(token_count(&json!(0)).unwrap(), 0);
        for count in [
            Value::Null,
            json!(-1),
            json!(0.5),
            json!(2147483648_u64),
            json!("2"),
        ] {
            assert!(token_count(&count).is_err());
        }
    }

    /// Generation keeps extension fields but never forwards message ownership or idempotency fields.
    #[test]
    fn inference_config_strips_only_initial_message_fields() {
        for ask in [Value::Null, json!("wrong"), json!([])] {
            assert!(matches!(
                inference_config(ask, vec![]),
                Err(service::Error::Status { code: 400, .. })
            ));
        }
        let ask = json!({"model_name":"tiny","text":"question","tool_result":{},"interject":"wait",
            "attachment_ids":[],"history_floor":"floor","request_id":"id","extension":{"x":1}});
        let config = inference_config(ask, vec![json!({"role":"user","text":"question"})]).unwrap();
        assert_eq!(config["extension"], json!({"x":1}));
        assert_eq!(config["turns"][0]["text"], "question");
        for key in [
            "text",
            "tool_result",
            "interject",
            "attachment_ids",
            "history_floor",
            "request_id",
        ] {
            assert!(config.get(key).is_none());
        }
    }
}
