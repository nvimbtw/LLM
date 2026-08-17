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

/// True when a row was actually removed, so a no-op delete is not a success.
pub async fn delete(pool: &PgPool, id: Uuid) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!("DELETE FROM users WHERE id_user = $1", id)
        .execute(pool)
        .await?;

    Ok(result.rows_affected() > 0)
}
