use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

pub struct Message {
    pub id_message: Uuid,
    // id_chat and id_model are part of the messages-table row that `query_as!`
    // maps (both appear in every RETURNING), but nothing reads them in Rust yet.
    // Kept so the struct mirrors the table rather than trimming the queries.
    #[allow(dead_code)]
    pub id_chat: Uuid,
    // Exactly one of these identifies the author: id_user for a person's
    // message, id_model for a model-generated reply. Both are nullable in the
    // schema, so both are Option here.
    pub id_user: Option<Uuid>,
    #[allow(dead_code)]
    pub id_model: Option<Uuid>,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

pub async fn create_user_message(
    pool: &PgPool,
    id_chat: Uuid,
    id_user: Uuid,
    content: &str,
) -> Result<Message, sqlx::Error> {
    sqlx::query_as!(
        Message,
        r#"INSERT INTO messages (id_chat, id_user, content) VALUES ($1, $2, $3)
           RETURNING id_message, id_chat, id_user, id_model, content, created_at"#,
        id_chat,
        id_user,
        content,
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
) -> Result<Message, sqlx::Error> {
    sqlx::query_as!(
        Message,
        r#"INSERT INTO messages (id_chat, id_model, content) VALUES ($1, $2, $3)
           RETURNING id_message, id_chat, id_user, id_model, content, created_at"#,
        id_chat,
        id_model,
        content,
    )
    .fetch_one(pool)
    .await
}

pub async fn list_for_chat(pool: &PgPool, id_chat: Uuid) -> Result<Vec<Message>, sqlx::Error> {
    // Oldest first, so the frontend renders the conversation top-to-bottom.
    sqlx::query_as!(
        Message,
        r#"SELECT id_message, id_chat, id_user, id_model, content, created_at
           FROM messages WHERE id_chat = $1 ORDER BY created_at ASC"#,
        id_chat,
    )
    .fetch_all(pool)
    .await
}
