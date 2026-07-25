use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

// One row of the messages table, as returned by the two inserts below.
//
// Only `id_message` is actually read (the assistant-reply frame sends it back to the
// client); the rest of the struct exists so it mirrors the table and both `RETURNING`
// clauses can map straight onto it. Reads go through `MessageWithModel` instead, which
// carries the joined model name. Hence the blanket allow rather than one per field.
#[allow(dead_code)]
pub struct Message {
    pub id_message: Uuid,
    pub id_chat: Uuid,
    // Exactly one of these identifies the author: id_user for a person's
    // message, id_model for a model-generated reply. Both are nullable in the
    // schema, so both are Option here.
    pub id_user: Option<Uuid>,
    pub id_model: Option<Uuid>,
    pub content: String,
    // Token length under the generating model's tokenizer; summed per chat (via
    // the aggregate in db::chat::list_for_user) for the sidebar's usage display.
    pub tokens: i32,
    pub created_at: DateTime<Utc>,
}

pub async fn create_user_message(
    pool: &PgPool,
    id_chat: Uuid,
    id_user: Uuid,
    content: &str,
    tokens: i32,
) -> Result<Message, sqlx::Error> {
    sqlx::query_as!(
        Message,
        r#"INSERT INTO messages (id_chat, id_user, content, tokens) VALUES ($1, $2, $3, $4)
           RETURNING id_message, id_chat, id_user, id_model, content, tokens, created_at"#,
        id_chat,
        id_user,
        content,
        tokens,
    )
    .fetch_one(pool)
    .await
}

// An assistant reply, attributed to the model that generated it. `id_model` is
// Option so a reply can still be saved if the run wasn't tied to a registered
// model, but in normal operation it carries the model's id.
pub async fn create_assistant_message(
    pool: &PgPool,
    id_chat: Uuid,
    id_model: Option<Uuid>,
    content: &str,
    tokens: i32,
) -> Result<Message, sqlx::Error> {
    sqlx::query_as!(
        Message,
        r#"INSERT INTO messages (id_chat, id_model, content, tokens) VALUES ($1, $2, $3, $4)
           RETURNING id_message, id_chat, id_user, id_model, content, tokens, created_at"#,
        id_chat,
        id_model,
        content,
        tokens,
    )
    .fetch_one(pool)
    .await
}

// A message together with the display name of the model that produced it. Separate
// from `Message` because `model_name` comes from a join rather than the messages row.
//
// The name is resolved at read time, not snapshotted onto the message — deliberately
// the opposite choice from `benchmark_runs.model_name`. A benchmark score belongs to
// the model *as it was* when measured, so relabelling past runs would be a lie; a
// reply, by contrast, was written by a model that still exists in the registry, so
// renaming that model should relabel its past replies rather than leaving them
// attributed to a name that no longer exists.
pub struct MessageWithModel {
    pub id_message: Uuid,
    // id_user set => a person sent it; otherwise it's an assistant reply.
    pub id_user: Option<Uuid>,
    pub content: String,
    pub created_at: DateTime<Utc>,
    // None on a user's own message, and on a reply whose model row is gone.
    pub model_name: Option<String>,
}

pub async fn list_for_chat(
    pool: &PgPool,
    id_chat: Uuid,
) -> Result<Vec<MessageWithModel>, sqlx::Error> {
    // Oldest first, so the frontend renders the conversation top-to-bottom.
    sqlx::query_as!(
        MessageWithModel,
        // `model_name?` forces Option: the join is a LEFT JOIN, so the name is null
        // for every user message, but sqlx would otherwise infer NOT NULL from the
        // models.name column and hand back a String that panics on the first user row.
        r#"SELECT m.id_message, m.id_user, m.content, m.created_at,
                  model.name as "model_name?"
           FROM messages m
           LEFT JOIN models model ON model.id_model = m.id_model
           WHERE m.id_chat = $1
           ORDER BY m.created_at ASC"#,
        id_chat,
    )
    .fetch_all(pool)
    .await
}
