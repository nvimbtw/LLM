use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

pub struct Message {
    pub id_message: Uuid,
    pub id_chat: Uuid,
    // Exactly one of these identifies the author: id_user for a person's
    // message, id_model for a model-generated reply. Both are nullable in the
    // schema, so both are Option here.
    pub id_user: Option<Uuid>,
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

// An assistant reply. id_model is left NULL for now — once a real model exists,
// its id should be passed here so replies are attributable to a specific model.
pub async fn create_assistant_message(
    pool: &PgPool,
    id_chat: Uuid,
    content: &str,
) -> Result<Message, sqlx::Error> {
    sqlx::query_as!(
        Message,
        r#"INSERT INTO messages (id_chat, content) VALUES ($1, $2)
           RETURNING id_message, id_chat, id_user, id_model, content, created_at"#,
        id_chat,
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
