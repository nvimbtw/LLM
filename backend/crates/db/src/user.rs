use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

pub use auth::AccessLevel;

pub struct User {
    pub id_user: Uuid,
    pub username: String,
    pub password: String,
    pub access_level: AccessLevel,
}

pub async fn create(
    pool: &PgPool,
    username: &str,
    password_hash: &str,
    access_level: Option<AccessLevel>,
) -> Result<User, sqlx::Error> {
    let access_level = access_level.unwrap_or(AccessLevel::User);

    sqlx::query_as!(
        User,
        r#"INSERT INTO users (username, password, access_level) VALUES ($1, $2, $3)
           RETURNING id_user, username, password, access_level as "access_level: AccessLevel""#,
        username,
        password_hash,
        access_level as AccessLevel,
    )
    .fetch_one(pool)
    .await
}

pub async fn find_by_username(pool: &PgPool, username: &str) -> Result<Option<User>, sqlx::Error> {
    sqlx::query_as!(
        User,
        r#"SELECT id_user, username, password, access_level as "access_level: AccessLevel"
           FROM users WHERE username = $1"#,
        username,
    )
    .fetch_optional(pool)
    .await
}

pub async fn find_by_id(pool: &PgPool, id: Uuid) -> Result<Option<User>, sqlx::Error> {
    sqlx::query_as!(
        User,
        r#"SELECT id_user, username, password, access_level as "access_level: AccessLevel"
           FROM users WHERE id_user = $1"#,
        id,
    )
    .fetch_optional(pool)
    .await
}

/// COALESCE keeps the existing value, so an omitted field is left alone.
pub async fn update(
    pool: &PgPool,
    id: Uuid,
    username: Option<&str>,
    password_hash: Option<&str>,
) -> Result<Option<User>, sqlx::Error> {
    sqlx::query_as!(
        User,
        r#"UPDATE users
           SET username = COALESCE($2, username),
               password = COALESCE($3, password)
           WHERE id_user = $1
           RETURNING id_user, username, password, access_level as "access_level: AccessLevel""#,
        id,
        username,
        password_hash,
    )
    .fetch_optional(pool)
    .await
}

/// Settings are opaque here — the frontend's `Settings` shape can grow without
/// a migration on this side. `None` means no user with that id.
pub async fn get_settings(pool: &PgPool, id: Uuid) -> Result<Option<Value>, sqlx::Error> {
    let row = sqlx::query!(
        r#"SELECT settings as "settings: Value" FROM users WHERE id_user = $1"#,
        id,
    )
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| r.settings))
}

/// Replaces the whole blob rather than merging — the caller already holds the
/// full object (`SettingsProvider` builds it from its current state plus one
/// changed key), so a partial-update path here would be a second way to do the
/// same thing.
pub async fn update_settings(
    pool: &PgPool,
    id: Uuid,
    settings: &Value,
) -> Result<Option<Value>, sqlx::Error> {
    let row = sqlx::query!(
        r#"UPDATE users SET settings = $2 WHERE id_user = $1
           RETURNING settings as "settings: Value""#,
        id,
        settings,
    )
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| r.settings))
}

/// What the composer is set to send with (model, thinking budget), opaque here
/// for the same reason `get_settings` is. `None` means no user with that id.
pub async fn get_composer_choice(pool: &PgPool, id: Uuid) -> Result<Option<Value>, sqlx::Error> {
    let row = sqlx::query!(
        r#"SELECT composer_choice as "composer_choice: Value" FROM users WHERE id_user = $1"#,
        id,
    )
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| r.composer_choice))
}

/// Replaces the whole blob, same reasoning as `update_settings`.
pub async fn update_composer_choice(
    pool: &PgPool,
    id: Uuid,
    composer_choice: &Value,
) -> Result<Option<Value>, sqlx::Error> {
    let row = sqlx::query!(
        r#"UPDATE users SET composer_choice = $2 WHERE id_user = $1
           RETURNING composer_choice as "composer_choice: Value""#,
        id,
        composer_choice,
    )
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| r.composer_choice))
}

/// True when a row was actually removed, so a no-op delete is not a success.
pub async fn delete(pool: &PgPool, id: Uuid) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!("DELETE FROM users WHERE id_user = $1", id)
        .execute(pool)
        .await?;

    Ok(result.rows_affected() > 0)
}
