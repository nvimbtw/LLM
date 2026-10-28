//! Conversation use cases: authorization, canonical preparation and persistence acknowledgements.

mod input;
mod project;

use db::conversation::{self as queries, Message};
use llm_engine::Proxy;
use serde_json::{json, Value};
use sqlx::PgPool;
use uuid::Uuid;

/// Construct a public conversation failure with the given HTTP status and safe detail.
pub fn failure(code: u16, detail: &str) -> service::Error {
    service::Error::Status {
        code,
        detail: detail.into(),
    }
}

/// Log a storage cause and translate it to the established ownership, conflict or validation response.
pub fn database(error: sqlx::Error) -> service::Error {
    eprintln!("conversation storage: {error}");
    match &error {
        sqlx::Error::RowNotFound => failure(404, "conversation not found"),
        sqlx::Error::Database(error) if error.is_unique_violation() => {
            failure(409, "this request was already stored")
        }
        sqlx::Error::Database(error) if error.is_foreign_key_violation() => {
            failure(404, "conversation or project not found")
        }
        sqlx::Error::Protocol(_) => failure(400, "an attachment is unknown or already sent"),
        _ => failure(500, "conversation storage failed"),
    }
}

/// Project a stored chat to its public metadata without exposing its owner.
pub fn chat(row: &queries::Chat) -> Value {
    json!({"chat_id":row.id,"title":row.title,"project_id":row.project_id,"tokens":row.tokens,"created_at":row.created_at})
}

/// Project attachment metadata without copying content bytes or internal ownership fields.
pub fn attachment(row: &db::attachment::Metadata) -> Value {
    json!({"attachment_id":row.id,"filename":row.filename,"media_type":row.media_type,"size":row.size,"created_at":row.created_at})
}

/// Load authorized attachment metadata for a stored message and return its public representation.
pub async fn message(pool: &PgPool, owner: Uuid, row: &Message) -> Result<Value, service::Error> {
    let files = db::attachment::for_message(pool, owner, row.id)
        .await
        .map_err(database)?;
    Ok(message_with_files(
        row,
        files.iter().map(attachment).collect(),
    ))
}

/// Assemble a public message from its storage row and already-authorized attachment metadata.
fn message_with_files(row: &Message, files: Vec<Value>) -> Value {
    json!({"message_id":row.id,"chat_id":row.chat_id,"role":row.role,"content":row.content,
        "tokens":row.tokens,"model_id":row.model_id,"model_name":row.model_name,"feedback":row.feedback,
        "attachments":files,"created_at":row.created_at})
}

/// Load ordered owner-scoped messages and join their attachment metadata without fetching file bytes.
pub async fn history(pool: &PgPool, owner: Uuid, id: Uuid) -> Result<Vec<Value>, service::Error> {
    let rows = queries::messages(pool, owner, id).await.map_err(database)?;
    let mut files: std::collections::HashMap<Uuid, Vec<Value>> = std::collections::HashMap::new();
    for file in db::attachment::for_chat(pool, owner, id)
        .await
        .map_err(database)?
    {
        if let Some(message) = file.message_id {
            files.entry(message).or_default().push(attachment(&file));
        }
    }
    Ok(rows
        .iter()
        .map(|row| message_with_files(row, files.remove(&row.id).unwrap_or_default()))
        .collect())
}

/// Return an owner-scoped chat or the same 404 used for an unknown conversation.
pub async fn owned(pool: &PgPool, owner: Uuid, id: Uuid) -> Result<queries::Chat, service::Error> {
    queries::get(pool, owner, id)
        .await
        .map_err(database)?
        .ok_or_else(|| failure(404, "conversation not found"))
}

pub struct Round {
    pub config: Value,
    pub model: Option<Uuid>,
    pub model_name: Option<String>,
    pub request: Uuid,
}

/// Authorize and prepare one round, then atomically persist its accepted input messages and bind uploads.
/// The verified caller visibility reaches engine admission before tokenizer access or stored input; generation receives only authorized stateless history.
pub async fn prepare(
    pool: &PgPool,
    engine: &Proxy,
    owner: Uuid,
    visibility: &str,
    id: Uuid,
    mut ask: Value,
) -> Result<Round, service::Error> {
    owned(pool, owner, id).await?;
    project::strip(&mut ask)?;
    let request = input::request_id(&ask)?;
    let mut history = queries::messages(pool, owner, id).await.map_err(database)?;
    input::apply_history_floor(&mut history, &ask);
    let text = project::user_text(&ask, &history).map(str::to_owned);
    project::apply(pool, owner, id, text.as_deref(), &mut ask).await?;
    let (status, prepared) = engine
        .scoped_json(
            &auth::Caller::User(owner),
            visibility,
            llm_engine::Method::POST,
            "/chat/prepare",
            Some(ask.clone()),
            8 * 1024 * 1024,
        )
        .await?;
    if !(200..300).contains(&status) {
        return Err(failure(
            status,
            prepared["detail"]
                .as_str()
                .unwrap_or("model preparation refused"),
        ));
    }
    let additions = prepared["turns"]
        .as_array()
        .ok_or_else(|| failure(502, "invalid prepared turns"))?;
    let files = input::attachment_ids(&ask)?;
    let inputs = input::new_inputs(additions, request)?;
    let turns = input::inference_turns(&history, &inputs);
    let config = input::inference_config(ask, turns)?;
    queries::append_many(pool, owner, id, &inputs, &files)
        .await
        .map_err(database)?;
    Ok(Round {
        config,
        model: prepared["model_id"]
            .as_str()
            .and_then(|id| Uuid::parse_str(id).ok()),
        model_name: prepared["model_name"].as_str().map(str::to_owned),
        request,
    })
}

/// Add stored project context to a context-floor request using only its authorized visible message history.
pub async fn context_request(
    pool: &PgPool,
    owner: Uuid,
    id: Uuid,
    history: &[Message],
    mut config: Value,
) -> Result<Value, service::Error> {
    let text = project::user_text(&Value::Null, history).map(str::to_owned);
    project::apply(pool, owner, id, text.as_deref(), &mut config).await?;
    Ok(config)
}

/// Persist a completed assistant reply before returning its public message and opaque inference usage.
/// Invalid text/counts or storage failures leave acknowledgement to the caller's error policy.
pub async fn finish(
    pool: &PgPool,
    owner: Uuid,
    id: Uuid,
    round: &Round,
    reply: &Value,
) -> Result<Value, service::Error> {
    let input = input::assistant_input(round, reply)?;
    let stored = queries::append(pool, owner, id, &input, &[])
        .await
        .map_err(database)?;
    let mut result = message(pool, owner, &stored).await?;
    result["usage"] = reply.clone();
    Ok(result)
}

/// A whole turn (including tool rounds) owns its conversation until completion.
/// Weak entries avoid retaining a mutex for every conversation ever opened.
pub async fn lock(id: Uuid) -> tokio::sync::OwnedMutexGuard<()> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock, Weak};
    type Locks = HashMap<Uuid, Weak<tokio::sync::Mutex<()>>>;
    static LOCKS: OnceLock<Mutex<Locks>> = OnceLock::new();
    let lock = {
        let mut locks = LOCKS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        locks.retain(|_, lock| lock.strong_count() > 0);
        let lock = locks
            .get(&id)
            .and_then(Weak::upgrade)
            .unwrap_or_else(|| Arc::new(tokio::sync::Mutex::new(())));
        locks.insert(id, Arc::downgrade(&lock));
        lock
    };
    lock.lock_owned().await
}
