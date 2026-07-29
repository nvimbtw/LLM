use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

pub struct Chat {
    pub id_chat: Uuid,
    pub id_user: Uuid,
    pub title: String,
    pub created_at: DateTime<Utc>,
}

// A chat plus its cumulative token usage (sum over its messages), for the
// sidebar list. Separate from `Chat` because it carries a joined aggregate that
// the base table doesn't, and it drops id_user (the list is already scoped to
// the caller).
pub struct ChatListItem {
    pub id_chat: Uuid,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub tokens: i64,
}

pub async fn create(pool: &PgPool, id_user: Uuid, title: &str) -> Result<Chat, sqlx::Error> {
    sqlx::query_as!(
        Chat,
        r#"INSERT INTO chats (id_user, title) VALUES ($1, $2)
           RETURNING id_chat, id_user, title, created_at"#,
        id_user,
        title,
    )
    .fetch_one(pool)
    .await
}

pub async fn find_by_id(pool: &PgPool, id_chat: Uuid) -> Result<Option<Chat>, sqlx::Error> {
    sqlx::query_as!(
        Chat,
        r#"SELECT id_chat, id_user, title, created_at FROM chats WHERE id_chat = $1"#,
        id_chat,
    )
    .fetch_optional(pool)
    .await
}

pub async fn list_for_user(pool: &PgPool, id_user: Uuid) -> Result<Vec<ChatListItem>, sqlx::Error> {
    // Most-recent first, for the sidebar/recency UI on the frontend. LEFT JOIN so
    // brand-new chats with no messages still appear, with a 0 token sum.
    sqlx::query_as!(
        ChatListItem,
        r#"SELECT c.id_chat, c.title, c.created_at,
                  COALESCE(SUM(m.tokens), 0)::int8 AS "tokens!"
           FROM chats c
           LEFT JOIN messages m ON m.id_chat = c.id_chat
           WHERE c.id_user = $1
           GROUP BY c.id_chat
           ORDER BY c.created_at DESC"#,
        id_user,
    )
    .fetch_all(pool)
    .await
}

pub async fn update_title(
    pool: &PgPool,
    id_chat: Uuid,
    title: &str,
) -> Result<Option<Chat>, sqlx::Error> {
    sqlx::query_as!(
        Chat,
        r#"UPDATE chats SET title = $2 WHERE id_chat = $1
           RETURNING id_chat, id_user, title, created_at"#,
        id_chat,
        title,
    )
    .fetch_optional(pool)
    .await
}

// Deletes a chat together with its messages. messages.id_chat references
// chats(id_chat) with no ON DELETE CASCADE, so the child rows must go first;
// a transaction keeps that all-or-nothing. Returns whether a chat row was
// actually removed, so the route can distinguish a real delete from a no-op.
pub async fn delete_with_messages(pool: &PgPool, id_chat: Uuid) -> Result<bool, sqlx::Error> {
    let mut tx = pool.begin().await?;

    sqlx::query!("DELETE FROM messages WHERE id_chat = $1", id_chat)
        .execute(&mut *tx)
        .await?;

    let result = sqlx::query!("DELETE FROM chats WHERE id_chat = $1", id_chat)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;

    Ok(result.rows_affected() > 0)
}
