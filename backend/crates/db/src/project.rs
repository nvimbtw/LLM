//! Owner-scoped private project storage and indexed source-message retrieval.

use crate::conversation::Chat;
use chrono::{DateTime, Utc};
use sqlx::{FromRow, PgConnection, PgPool};
use uuid::Uuid;

#[derive(FromRow)]
pub struct Project {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub notes: String,
    pub thread_count: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(FromRow)]
pub struct Memory {
    pub memory: String,
    pub thread: Uuid,
    pub message: Uuid,
    pub role: String,
    pub age_seconds: i64,
}

pub struct Context {
    pub project: Project,
    pub memories: Vec<Memory>,
}

pub struct Detail {
    pub project: Project,
    pub threads: Vec<Chat>,
}

/// Load one owned project and its current chat count; unknown and foreign IDs both return none.
pub async fn get(pool: &PgPool, owner: Uuid, id: Uuid) -> Result<Option<Project>, sqlx::Error> {
    let mut connection = pool.acquire().await?;
    load_project(&mut connection, owner, id).await
}

/// Read owned project metadata and a bounded thread page from one repeatable-read snapshot.
/// Concurrent membership changes cannot make the count and returned page describe different database states.
pub async fn detail(
    pool: &PgPool,
    owner: Uuid,
    id: Uuid,
    limit: i64,
    offset: i64,
) -> Result<Option<Detail>, sqlx::Error> {
    let mut tx = pool.begin().await?;
    sqlx::query!("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        .execute(&mut *tx)
        .await?;
    let detail = match load_project(&mut tx, owner, id).await? {
        Some(project) => Some(Detail {
            threads: load_threads(&mut tx, owner, id, limit, offset).await?,
            project,
        }),
        None => None,
    };
    tx.commit().await?;
    Ok(detail)
}

/// Load owned project metadata on the caller's connection, retaining its transaction snapshot when present.
async fn load_project(
    connection: &mut PgConnection,
    owner: Uuid,
    id: Uuid,
) -> Result<Option<Project>, sqlx::Error> {
    sqlx::query_as!(
        Project,
        r#"SELECT p.id, p.name, p.description, p.notes, p.created_at, p.updated_at,
                  count(c.id) AS "thread_count!"
           FROM projects p LEFT JOIN chats c ON c.project_id = p.id AND c.owner_id = p.owner_id
           WHERE p.owner_id = $1 AND p.id = $2 GROUP BY p.id"#,
        owner,
        id
    )
    .fetch_optional(connection)
    .await
}

/// Return a bounded page and total from one owner-scoped snapshot, newest projects first.
pub async fn list(
    pool: &PgPool,
    owner: Uuid,
    limit: i64,
    offset: i64,
) -> Result<(Vec<Project>, i64), sqlx::Error> {
    let mut tx = pool.begin().await?;
    sqlx::query!("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        .execute(&mut *tx)
        .await?;
    let total = sqlx::query_scalar!(
        r#"SELECT count(*) AS "total!" FROM projects WHERE owner_id = $1"#,
        owner
    )
    .fetch_one(&mut *tx)
    .await?;
    let rows = sqlx::query_as!(
        Project,
        r#"SELECT p.id, p.name, p.description, p.notes, p.created_at, p.updated_at,
                  count(c.id) AS "thread_count!"
           FROM projects p LEFT JOIN chats c ON c.project_id = p.id AND c.owner_id = p.owner_id
           WHERE p.owner_id = $1 GROUP BY p.id
           ORDER BY p.created_at DESC, p.id LIMIT $2 OFFSET $3"#,
        owner,
        limit,
        offset
    )
    .fetch_all(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok((rows, total))
}

/// Insert validated project fields for one owner and return the committed metadata.
pub async fn create(
    pool: &PgPool,
    owner: Uuid,
    name: &str,
    description: &str,
    notes: &str,
) -> Result<Project, sqlx::Error> {
    sqlx::query_as!(
        Project,
        r#"INSERT INTO projects(owner_id, name, description, notes) VALUES ($1, $2, $3, $4)
           RETURNING id, name, description, notes, created_at, updated_at, 0::bigint AS "thread_count!""#,
        owner,
        name,
        description,
        notes
    )
    .fetch_one(pool)
    .await
}

/// Update only supplied fields of an owned project, returning false for unknown or foreign IDs.
pub async fn update(
    pool: &PgPool,
    owner: Uuid,
    id: Uuid,
    name: Option<&str>,
    description: Option<&str>,
    notes: Option<&str>,
) -> Result<bool, sqlx::Error> {
    Ok(sqlx::query!(
        r#"UPDATE projects SET name = COALESCE($3, name), description = COALESCE($4, description),
                  notes = COALESCE($5, notes), updated_at = now()
           WHERE owner_id = $1 AND id = $2"#,
        owner,
        id,
        name,
        description,
        notes
    )
    .execute(pool)
    .await?
    .rows_affected()
        == 1)
}

/// Delete an owned project while the foreign key clears associated chats instead of deleting them.
pub async fn remove(pool: &PgPool, owner: Uuid, id: Uuid) -> Result<bool, sqlx::Error> {
    Ok(sqlx::query!(
        "DELETE FROM projects WHERE owner_id = $1 AND id = $2",
        owner,
        id
    )
    .execute(pool)
    .await?
    .rows_affected()
        == 1)
}

/// Read a bounded owned-project chat page on the detail operation's snapshot, including stored token totals.
async fn load_threads(
    connection: &mut PgConnection,
    owner: Uuid,
    project: Uuid,
    limit: i64,
    offset: i64,
) -> Result<Vec<Chat>, sqlx::Error> {
    sqlx::query_as!(
        Chat,
        r#"SELECT c.id, c.title, c.project_id, c.created_at, COALESCE(sum(m.tokens), 0)::bigint AS "tokens!"
           FROM chats c LEFT JOIN messages m ON m.chat_id = c.id
           WHERE c.owner_id = $1 AND c.project_id = $2 GROUP BY c.id
           ORDER BY c.created_at DESC, c.id LIMIT $3 OFFSET $4"#,
        owner,
        project,
        limit,
        offset
    )
    .fetch_all(connection)
    .await
}

/// Read current owned project membership and matching messages from one consistent database snapshot.
/// A detached, deleted, unknown or foreign chat supplies no project context.
pub async fn context(
    pool: &PgPool,
    owner: Uuid,
    thread: Uuid,
    words: Option<&str>,
) -> Result<Option<Context>, sqlx::Error> {
    let mut tx = pool.begin().await?;
    sqlx::query!("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        .execute(&mut *tx)
        .await?;
    sqlx::query!("SET LOCAL statement_timeout = '2000ms'")
        .execute(&mut *tx)
        .await?;
    let project = sqlx::query_as!(
        Project,
        r#"SELECT p.id, p.name, p.description, p.notes, p.created_at, p.updated_at,
                  (SELECT count(*) FROM chats WHERE project_id = p.id AND owner_id = $1) AS "thread_count!"
           FROM chats c JOIN projects p ON p.id = c.project_id AND p.owner_id = c.owner_id
           WHERE c.owner_id = $1 AND c.id = $2"#,
        owner,
        thread
    )
    .fetch_optional(&mut *tx)
    .await?;
    let context = if let Some(project) = project {
        let memories = match words {
            Some(words) => matching_messages(&mut tx, owner, project.id, thread, words).await?,
            None => Vec::new(),
        };
        Some(Context { project, memories })
    } else {
        None
    };
    tx.commit().await?;
    Ok(context)
}

/// Find eight ranked source messages using a validated OR word query inside the caller's snapshot.
/// Results contain at most 2000 characters each and nonnegative source ages in seconds.
async fn matching_messages(
    connection: &mut PgConnection,
    owner: Uuid,
    project: Uuid,
    thread: Uuid,
    words: &str,
) -> Result<Vec<Memory>, sqlx::Error> {
    sqlx::query_as!(
        Memory,
        r#"WITH ranked AS (
               SELECT project_memory_text(m.role, m.content) AS content,
                      m.chat_id AS thread, m.id AS message, m.role::text AS role, m.created_at
               FROM messages m JOIN chats c ON c.id = m.chat_id
               JOIN projects p ON p.id = c.project_id AND p.owner_id = c.owner_id
               WHERE c.owner_id = $1 AND c.project_id = $2 AND c.id <> $3
                 AND m.role IN ('user', 'assistant') AND m.memory_words @@ to_tsquery('simple', $4)
               ORDER BY ts_rank_cd(m.memory_words, to_tsquery('simple', $4)) DESC, m.created_at DESC, m.id
               LIMIT 8
           )
           SELECT CASE WHEN char_length(content) <= 2000 THEN content
                  ELSE left(ts_headline('simple', content, to_tsquery('simple', $4),
                      'MaxWords=60, MinWords=8, MaxFragments=1, StartSel="", StopSel=""'), 2000)
                  END AS "memory!", thread, message, role AS "role!",
                  GREATEST(0, floor(extract(epoch FROM now() - created_at)))::bigint AS "age_seconds!"
           FROM ranked"#,
        owner,
        project,
        thread,
        words
    )
    .fetch_all(connection)
    .await
}
