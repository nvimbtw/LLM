//! Stored project-context assembly and bounded word-query policy for all chat transports.

use db::conversation::Message;
use serde_json::{json, Value};
use sqlx::PgPool;
use uuid::Uuid;

const QUERY_TEXT_CHARACTERS: usize = 8192;
const QUERY_TERMS: usize = 24;
const QUERY_WORD_CHARACTERS: usize = 64;

/// Remove untrusted project identifiers and memory context before any engine request is made.
pub(super) fn strip(ask: &mut Value) -> Result<(), service::Error> {
    let object = ask
        .as_object_mut()
        .ok_or_else(|| super::failure(400, "expected an object"))?;
    for field in ["project", "project_id", "memories"] {
        object.remove(field);
    }
    Ok(())
}

/// Select the latest submitted steering or user text, falling back to visible authorized user history.
pub(super) fn user_text<'a>(ask: &'a Value, history: &'a [Message]) -> Option<&'a str> {
    ask["interject"]
        .as_str()
        .or_else(|| ask["text"].as_str())
        .or_else(|| {
            history
                .iter()
                .rev()
                .find(|message| message.role == "user")
                .map(|message| message.content.as_str())
        })
}

/// Form at most 24 unique lowercase words joined by OR, scanning at most 8192 source characters.
/// Alphanumeric-only terms prevent query syntax injection; common question words and oversized terms are omitted.
fn word_query(text: Option<&str>) -> Option<String> {
    const COMMON: [&str; 29] = [
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "how", "i",
        "in", "is", "it", "of", "on", "or", "please", "that", "the", "this", "to", "was", "what",
        "with", "you",
    ];
    let bounded = text?
        .chars()
        .take(QUERY_TEXT_CHARACTERS)
        .collect::<String>();
    let mut terms = Vec::new();
    for word in bounded.split(|character: char| !character.is_alphanumeric()) {
        let length = word.chars().count();
        if !(2..=QUERY_WORD_CHARACTERS).contains(&length) {
            continue;
        }
        let word = word.to_lowercase();
        if !COMMON.contains(&word.as_str()) && !terms.contains(&word) {
            terms.push(word);
            if terms.len() == QUERY_TERMS {
                break;
            }
        }
    }
    (!terms.is_empty()).then(|| terms.join(" | "))
}

/// Replace project fields with an owner-scoped database snapshot and bounded matching source snippets.
/// Nonproject chats remain unchanged after stripping; storage failures prevent sending a partial context.
pub(super) async fn apply(
    pool: &PgPool,
    owner: Uuid,
    thread: Uuid,
    text: Option<&str>,
    config: &mut Value,
) -> Result<(), service::Error> {
    strip(config)?;
    let words = word_query(text);
    let Some(context) = db::project::context(pool, owner, thread, words.as_deref())
        .await
        .map_err(super::database)?
    else {
        return Ok(());
    };
    let project = context.project;
    config["project"] = json!({"project_id":project.id,"name":project.name,
        "description":project.description,"notes":project.notes});
    config["memories"] = json!(context
        .memories
        .iter()
        .map(|source| {
            json!({"memory":source.memory,"thread":source.thread,"message":source.message,
            "role":source.role,"age_seconds":source.age_seconds})
        })
        .collect::<Vec<_>>());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Natural questions share individual words rather than requiring an exact question match.
    #[test]
    fn query_terms_are_disjunctive_unique_and_safe() {
        assert_eq!(
            word_query(Some("What are the ORBIT battery and orbit plans?")),
            Some("orbit | battery | plans".into())
        );
        assert_eq!(word_query(Some("the and you?!")), None);
        assert_eq!(word_query(None), None);
        assert_eq!(
            word_query(Some("x'); DROP | table:*")),
            Some("drop | table".into())
        );
    }

    /// Term count, term length and source scanning all have independent resource ceilings.
    #[test]
    fn query_work_is_bounded() {
        let text = (0..100)
            .map(|index| format!("term{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(
            word_query(Some(&text)).unwrap().split(" | ").count(),
            QUERY_TERMS
        );
        assert_eq!(word_query(Some(&"x".repeat(65))), None);
        assert_eq!(
            word_query(Some(&format!(
                "{} orbit",
                " ".repeat(QUERY_TEXT_CHARACTERS)
            ))),
            None
        );
    }

    /// Client-supplied identity/context fields are discarded while ordinary engine extension settings survive.
    #[test]
    fn supplied_context_is_never_trusted() {
        let mut ask = json!({"project":{"notes":"fake"},"project_id":"fake","memories":[{}],"temperature":0.3});
        strip(&mut ask).unwrap();
        assert_eq!(ask, json!({"temperature":0.3}));
        assert!(strip(&mut Value::Null).is_err());
    }
}
