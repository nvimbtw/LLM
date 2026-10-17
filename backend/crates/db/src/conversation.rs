use chrono::{DateTime, Utc};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

#[derive(FromRow)]
pub struct Chat {
    pub id: Uuid,
    pub title: String,
    pub project_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub tokens: i64,
}

#[derive(FromRow)]
pub struct Message {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub role: String,
    pub content: String,
    pub tokens: i32,
    pub model_id: Option<Uuid>,
    pub model_name: Option<String>,
    pub tool: Option<String>,
    pub argument: Option<String>,
    pub feedback: Option<String>,
    pub created_at: DateTime<Utc>,
}

pub struct NewMessage {
    pub role: String,
    pub content: String,
    pub tokens: i32,
    pub model_id: Option<Uuid>,
    pub model_name: Option<String>,
    pub tool: Option<String>,
    pub argument: Option<String>,
    pub request_id: Uuid,
}

/// Return all owned chat metadata, including optional project membership and stored token totals.
pub async fn list(pool: &PgPool, owner: Uuid) -> Result<Vec<Chat>, sqlx::Error> {
    sqlx::query_as!(Chat, r#"SELECT c.id,c.title,c.project_id,c.created_at,COALESCE(sum(m.tokens),0)::bigint AS "tokens!" FROM chats c LEFT JOIN messages m ON m.chat_id=c.id WHERE c.owner_id=$1 GROUP BY c.id ORDER BY c.created_at DESC"#, owner).fetch_all(pool).await
}

/// Return a bounded recent slice, calculating token totals only for its selected chats.
pub async fn list_recent(pool: &PgPool, owner: Uuid, limit: i64) -> Result<Vec<Chat>, sqlx::Error> {
    sqlx::query_as!(Chat, r#"SELECT c.id,c.title,c.project_id,c.created_at,COALESCE(t.tokens,0)::bigint AS "tokens!" FROM (SELECT id,title,project_id,created_at FROM chats WHERE owner_id=$1 ORDER BY created_at DESC LIMIT $2) c LEFT JOIN LATERAL (SELECT sum(tokens) AS tokens FROM messages WHERE chat_id=c.id) t ON true ORDER BY c.created_at DESC"#, owner, limit).fetch_all(pool).await
}

/// Return an owned chat; foreign or unknown IDs yield none.
pub async fn get(pool: &PgPool, owner: Uuid, id: Uuid) -> Result<Option<Chat>, sqlx::Error> {
    sqlx::query_as!(Chat, r#"SELECT c.id,c.title,c.project_id,c.created_at,COALESCE(sum(m.tokens),0)::bigint AS "tokens!" FROM chats c LEFT JOIN messages m ON m.chat_id=c.id WHERE c.owner_id=$1 AND c.id=$2 GROUP BY c.id"#, owner, id).fetch_optional(pool).await
}

/// Create an owned chat without project membership, returning its committed metadata.
pub async fn create(pool: &PgPool, owner: Uuid, title: &str) -> Result<Chat, sqlx::Error> {
    create_in_project(pool, owner, title, None).await
}

/// Atomically create an owned chat with an optional owned project; invalid membership inserts nothing.
pub async fn create_in_project(
    pool: &PgPool,
    owner: Uuid,
    title: &str,
    project: Option<Uuid>,
) -> Result<Chat, sqlx::Error> {
    sqlx::query_as!(
        Chat,
        r#"INSERT INTO chats(owner_id, title, project_id)
           SELECT $1, $2, $3::uuid
           WHERE $3::uuid IS NULL OR EXISTS (SELECT 1 FROM projects WHERE id = $3 AND owner_id = $1)
           RETURNING id, title, project_id, created_at, 0::bigint AS "tokens!""#,
        owner,
        title,
        project
    )
    .fetch_one(pool)
    .await
}

/// Move an owned chat to an owned project, or detach it, returning none without any mutation on invalid IDs.
pub async fn assign_project(
    pool: &PgPool,
    owner: Uuid,
    id: Uuid,
    project: Option<Uuid>,
) -> Result<Option<Chat>, sqlx::Error> {
    sqlx::query_as!(
        Chat,
        r#"WITH updated AS (
               UPDATE chats SET project_id = $3::uuid WHERE owner_id = $1 AND id = $2
                 AND ($3::uuid IS NULL OR EXISTS (SELECT 1 FROM projects WHERE id = $3 AND owner_id = $1))
               RETURNING id, title, project_id, created_at
           )
           SELECT c.id, c.title, c.project_id, c.created_at,
                  COALESCE((SELECT sum(tokens) FROM messages WHERE chat_id = c.id), 0)::bigint AS "tokens!"
           FROM updated c"#,
        owner,
        id,
        project
    )
    .fetch_optional(pool)
    .await
}

/// Rename an owned chat, reporting whether a row changed without exposing another owner's metadata.
pub async fn rename(
    pool: &PgPool,
    owner: Uuid,
    id: Uuid,
    title: &str,
) -> Result<bool, sqlx::Error> {
    Ok(sqlx::query!(
        r#"UPDATE chats SET title=$3 WHERE owner_id=$1 AND id=$2"#,
        owner,
        id,
        title
    )
    .execute(pool)
    .await?
    .rows_affected()
        == 1)
}

/// Delete an owned chat and its cascading messages and attachments; return false for foreign IDs.
pub async fn remove(pool: &PgPool, owner: Uuid, id: Uuid) -> Result<bool, sqlx::Error> {
    Ok(sqlx::query!(
        r#"DELETE FROM chats WHERE owner_id=$1 AND id=$2"#,
        owner,
        id
    )
    .execute(pool)
    .await?
    .rows_affected()
        == 1)
}

/// Read ordered message history only from the authenticated owner's chat.
pub async fn messages(pool: &PgPool, owner: Uuid, id: Uuid) -> Result<Vec<Message>, sqlx::Error> {
    sqlx::query_as!(Message, r#"SELECT m.id,m.chat_id,m.role::text AS "role!",m.content,m.tokens,m.model_id,m.model_name,m.tool,m.argument,m.feedback::text,m.created_at FROM messages m JOIN chats c ON c.id=m.chat_id WHERE c.owner_id=$1 AND c.id=$2 ORDER BY m.position"#, owner, id).fetch_all(pool).await
}

/// The chat row serializes positions and attachment binding in the same transaction.
pub async fn append(
    pool: &PgPool,
    owner: Uuid,
    chat: Uuid,
    input: &NewMessage,
    files: &[Uuid],
) -> Result<Message, sqlx::Error> {
    append_many(pool, owner, chat, std::slice::from_ref(input), files)
        .await?
        .pop()
        .ok_or(sqlx::Error::RowNotFound)
}

/// Tool result and interjection commit together, preserving prompt order.
pub async fn append_many(
    pool: &PgPool,
    owner: Uuid,
    chat: Uuid,
    inputs: &[NewMessage],
    files: &[Uuid],
) -> Result<Vec<Message>, sqlx::Error> {
    let mut tx = pool.begin().await?;
    sqlx::query!(
        r#"SELECT id FROM chats WHERE id=$1 AND owner_id=$2 FOR UPDATE"#,
        chat,
        owner
    )
    .fetch_one(&mut *tx)
    .await?;
    let mut messages = Vec::with_capacity(inputs.len());
    for (index, input) in inputs.iter().enumerate() {
        let message = sqlx::query_as!(Message, r#"INSERT INTO messages(chat_id,position,role,content,tokens,model_id,model_name,tool,argument,request_id) SELECT $1,COALESCE(max(position),0)+1,$2::text::conversation_role,$3,$4,$5,$6,$7,$8,$9 FROM messages WHERE chat_id=$1 RETURNING id,chat_id,role::text AS "role!",content,tokens,model_id,model_name,tool,argument,feedback::text,created_at"#, chat, &input.role, &input.content, input.tokens, input.model_id, input.model_name.as_deref(), input.tool.as_deref(), input.argument.as_deref(), input.request_id).fetch_one(&mut *tx).await?;
        if index == 0 && !files.is_empty() {
            let bound = sqlx::query!(r#"UPDATE attachments SET message_id=$1 WHERE id=ANY($2) AND owner_id=$3 AND message_id IS NULL"#, message.id, files, owner).execute(&mut *tx).await?.rows_affected();
            if bound != files.len() as u64 {
                return Err(sqlx::Error::Protocol(
                    "an attachment is unknown or already sent".into(),
                ));
            }
        }
        messages.push(message);
    }
    tx.commit().await?;
    Ok(messages)
}

/// Set or clear feedback on an owned message, returning whether an authorized row changed.
pub async fn feedback(
    pool: &PgPool,
    owner: Uuid,
    id: Uuid,
    value: Option<&str>,
) -> Result<bool, sqlx::Error> {
    Ok(sqlx::query!(r#"UPDATE messages m SET feedback=$3::text::conversation_feedback FROM chats c WHERE m.chat_id=c.id AND c.owner_id=$1 AND m.id=$2"#, owner, id, value).execute(pool).await?.rows_affected() == 1)
}

/// Aggregate the owner's stored usage over the requested number of UTC calendar days.
pub async fn usage(
    pool: &PgPool,
    owner: Uuid,
    days: i32,
) -> Result<serde_json::Value, sqlx::Error> {
    sqlx::query_scalar!(r#"SELECT jsonb_build_object('days',$2::integer,'series',COALESCE((SELECT jsonb_agg(s ORDER BY s.date) FROM (SELECT (m.created_at AT TIME ZONE 'UTC')::date AS date,COALESCE(sum(m.tokens) FILTER (WHERE m.role='user'),0) AS asked,COALESCE(sum(m.tokens) FILTER (WHERE m.role='assistant'),0) AS generated,count(*) FILTER (WHERE m.role='assistant') AS replies FROM messages m JOIN chats c ON c.id=m.chat_id WHERE c.owner_id=$1 AND m.created_at>=now()-make_interval(days=>$2) GROUP BY 1) s),'[]'::jsonb),'by_model',COALESCE((SELECT jsonb_agg(s ORDER BY s.tokens DESC) FROM (SELECT m.model_id,m.model_name AS name,sum(m.tokens) AS tokens,count(*) AS replies FROM messages m JOIN chats c ON c.id=m.chat_id WHERE c.owner_id=$1 AND m.role='assistant' AND m.created_at>=now()-make_interval(days=>$2) GROUP BY m.model_id,m.model_name) s),'[]'::jsonb)) AS "usage!""#, owner, days).fetch_one(pool).await
}
