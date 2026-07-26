use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

// A user's verdict on one assistant reply. Two-valued, and always carried as an
// `Option` — `None` is "not rated", which is what almost every reply is.
#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize, sqlx::Type)]
#[sqlx(type_name = "message_feedback", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum Feedback {
    Up,
    Down,
}

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
    // The thumbs verdict, so a reload shows the rating the user already gave.
    // Always None on a user's own message — only replies are rateable.
    pub feedback: Option<Feedback>,
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
                  model.name as "model_name?",
                  m.feedback as "feedback: Feedback"
           FROM messages m
           LEFT JOIN models model ON model.id_model = m.id_model
           WHERE m.id_chat = $1
           ORDER BY m.created_at ASC"#,
        id_chat,
    )
    .fetch_all(pool)
    .await
}

// Which chat a message belongs to, and whether a person wrote it. The feedback
// route needs both before it will write: the chat to check ownership against, and
// the author to reject a thumbs on the user's own message.
pub struct MessageOrigin {
    pub id_chat: Uuid,
    // Some => a person wrote it; None => it's an assistant reply.
    pub id_user: Option<Uuid>,
}

pub async fn find_origin(
    pool: &PgPool,
    id_message: Uuid,
) -> Result<Option<MessageOrigin>, sqlx::Error> {
    sqlx::query_as!(
        MessageOrigin,
        r#"SELECT id_chat, id_user FROM messages WHERE id_message = $1"#,
        id_message,
    )
    .fetch_optional(pool)
    .await
}

// Sets or clears the thumbs on one reply. `None` un-rates it — the user changing
// their mind has to be able to get back to "no opinion", or the export would keep
// counting a verdict they withdrew.
pub async fn set_feedback(
    pool: &PgPool,
    id_message: Uuid,
    feedback: Option<Feedback>,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!(
        r#"UPDATE messages SET feedback = $2 WHERE id_message = $1"#,
        id_message,
        feedback as Option<Feedback>,
    )
    .execute(pool)
    .await?;

    Ok(result.rows_affected() > 0)
}

// Distinct user turns, newest first, for the auto-labeller to generate against.
//
// Real prompts rather than a synthetic list: the point of preference data is to
// improve the model on the distribution it is actually used on, and a fixed prompt
// set would tune it for questions nobody asks.
//
// Deduplicated on the trimmed text because the same message gets re-sent while
// someone is testing, and generating six candidates for twenty copies of "hi" would
// spend the whole run teaching the model about one prompt.
pub async fn distinct_user_prompts(
    pool: &PgPool,
    limit: i64,
) -> Result<Vec<String>, sqlx::Error> {
    let rows = sqlx::query!(
        r#"SELECT DISTINCT ON (btrim(content)) btrim(content) as "content!"
           FROM messages
           WHERE id_user IS NOT NULL AND btrim(content) <> ''
           ORDER BY btrim(content), created_at DESC
           LIMIT $1"#,
        limit,
    )
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(|r| r.content).collect())
}

// One rated reply together with the user turn it answered — the shape the
// benchmark's preference stage scores.
pub struct RatedReply {
    pub prompt: Option<String>,
    pub response: String,
    pub feedback: Option<Feedback>,
}

// Every rated assistant reply across every chat, oldest first.
//
// The prompt comes from a LATERAL subquery rather than a join on some stored
// parent id, because a message has no parent: the conversation is an ordered list,
// so "the prompt this answered" is by definition the most recent user turn before
// it in the same chat. Replies with no preceding user turn (the model spoke first)
// come back with `prompt: None` and the caller drops them — a response scored
// without its prompt measures the wrong conditional.
pub async fn list_rated(pool: &PgPool) -> Result<Vec<RatedReply>, sqlx::Error> {
    sqlx::query_as!(
        RatedReply,
        r#"SELECT prompt.content as "prompt?",
                  m.content as response,
                  m.feedback as "feedback: Feedback"
           FROM messages m
           LEFT JOIN LATERAL (
               SELECT p.content
               FROM messages p
               WHERE p.id_chat = m.id_chat
                 AND p.id_user IS NOT NULL
                 AND p.created_at < m.created_at
               ORDER BY p.created_at DESC
               LIMIT 1
           ) prompt ON true
           WHERE m.feedback IS NOT NULL AND m.id_user IS NULL
           ORDER BY m.created_at ASC"#,
    )
    .fetch_all(pool)
    .await
}
