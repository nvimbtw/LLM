use db::conversation::{self as queries, Message, NewMessage};
use llm_engine::Proxy;
use serde_json::{json, Value};
use sqlx::PgPool;
use uuid::Uuid;

pub fn failure(code: u16, detail: &str) -> service::Error {
    service::Error::Status {
        code,
        detail: detail.into(),
    }
}

pub fn database(error: sqlx::Error) -> service::Error {
    eprintln!("conversation storage: {error}");
    match &error {
        sqlx::Error::RowNotFound => failure(404, "conversation not found"),
        sqlx::Error::Database(error) if error.is_unique_violation() => {
            failure(409, "this request was already stored")
        }
        sqlx::Error::Protocol(_) => failure(400, "an attachment is unknown or already sent"),
        _ => failure(500, "conversation storage failed"),
    }
}

pub fn chat(row: &queries::Chat) -> Value {
    json!({"chat_id":row.id,"title":row.title,"tokens":row.tokens,"created_at":row.created_at})
}

pub fn attachment(row: &db::attachment::Metadata) -> Value {
    json!({"attachment_id":row.id,"filename":row.filename,"media_type":row.media_type,"size":row.size,"created_at":row.created_at})
}

pub async fn message(pool: &PgPool, owner: Uuid, row: &Message) -> Result<Value, service::Error> {
    let files = db::attachment::for_message(pool, owner, row.id)
        .await
        .map_err(database)?;
    Ok(message_with_files(
        row,
        files.iter().map(attachment).collect(),
    ))
}

fn message_with_files(row: &Message, files: Vec<Value>) -> Value {
    json!({"message_id":row.id,"chat_id":row.chat_id,"role":row.role,"content":row.content,
        "tokens":row.tokens,"model_id":row.model_id,"model_name":row.model_name,"feedback":row.feedback,
        "attachments":files,"created_at":row.created_at})
}

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

pub async fn prepare(
    pool: &PgPool,
    engine: &Proxy,
    owner: Uuid,
    id: Uuid,
    mut ask: Value,
) -> Result<Round, service::Error> {
    owned(pool, owner, id).await?;
    let caller = auth::Caller::User(owner);
    ask["model_name"]
        .as_str()
        .ok_or_else(|| failure(400, "model_name is required"))?;
    let request = match ask.get("request_id") {
        Some(value) => value
            .as_str()
            .and_then(|v| Uuid::parse_str(v).ok())
            .ok_or_else(|| failure(400, "invalid request_id"))?,
        None => Uuid::new_v4(),
    };
    let prepared = engine.post(&caller, "/chat/prepare", ask.clone()).await?;
    let additions = prepared["turns"]
        .as_array()
        .ok_or_else(|| failure(502, "invalid prepared turns"))?;
    let mut history = queries::messages(pool, owner, id).await.map_err(database)?;
    if let Some(floor) = ask["history_floor"]
        .as_str()
        .and_then(|v| Uuid::parse_str(v).ok())
    {
        if let Some(index) = history.iter().position(|m| m.id == floor) {
            history.drain(..index);
        }
    }
    let files: Vec<Uuid> =
        serde_json::from_value(ask.get("attachment_ids").cloned().unwrap_or(json!([])))
            .map_err(|_| failure(400, "invalid attachments"))?;
    if files.len() > 5 {
        return Err(failure(400, "at most five attachments are allowed"));
    }
    let mut turns: Vec<Value> = history
        .iter()
        .map(|m| json!({"role":m.role,"text":m.content,"tool":m.tool,"argument":m.argument}))
        .collect();
    let inputs = new_inputs(additions, request)?;
    turns.extend(
        inputs
            .iter()
            .map(|input| json!({"role":input.role,"text":input.content})),
    );
    queries::append_many(pool, owner, id, &inputs, &files)
        .await
        .map_err(database)?;
    ask["turns"] = json!(turns);
    let object = ask
        .as_object_mut()
        .ok_or_else(|| failure(400, "expected an object"))?;
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
    Ok(Round {
        config: ask,
        model: prepared["model_id"]
            .as_str()
            .and_then(|id| Uuid::parse_str(id).ok()),
        model_name: prepared["model_name"].as_str().map(str::to_owned),
        request,
    })
}

fn new_inputs(turns: &[Value], request: Uuid) -> Result<Vec<NewMessage>, service::Error> {
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

pub async fn finish(
    pool: &PgPool,
    owner: Uuid,
    id: Uuid,
    round: &Round,
    reply: &Value,
) -> Result<Value, service::Error> {
    let input = NewMessage {
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
    };
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

fn token_count(value: &Value) -> Result<i32, service::Error> {
    value
        .as_u64()
        .and_then(|count| i32::try_from(count).ok())
        .ok_or_else(|| failure(502, "invalid token count from inference"))
}
