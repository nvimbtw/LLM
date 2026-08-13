//! Presets are personal, so every query here is scoped to `id_user`: ownership
//! is enforced in SQL rather than only in the handler above it.

use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

pub struct Preset {
    pub id_preset: Uuid,
    pub name: String,
    pub config: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub async fn list_for_user(pool: &PgPool, id_user: Uuid) -> Result<Vec<Preset>, sqlx::Error> {
    sqlx::query_as!(
        Preset,
        r#"SELECT id_preset, name,
                  config as "config: Value", created_at, updated_at
           FROM training_presets
           WHERE id_user = $1
           ORDER BY updated_at DESC"#,
        id_user,
    )
    .fetch_all(pool)
    .await
}

pub async fn create(
    pool: &PgPool,
    id_user: Uuid,
    name: &str,
    config: &Value,
) -> Result<Preset, sqlx::Error> {
    sqlx::query_as!(
        Preset,
        r#"INSERT INTO training_presets (id_user, name, config)
           VALUES ($1, $2, $3)
           RETURNING id_preset, name,
                     config as "config: Value", created_at, updated_at"#,
        id_user,
        name,
        config,
    )
    .fetch_one(pool)
    .await
}

/// COALESCE keeps the existing value, so an omitted field is left alone.
/// None when no preset with that id belongs to this user.
pub async fn update(
    pool: &PgPool,
    id_preset: Uuid,
    id_user: Uuid,
    name: Option<&str>,
    config: Option<&Value>,
) -> Result<Option<Preset>, sqlx::Error> {
    sqlx::query_as!(
        Preset,
        r#"UPDATE training_presets
           SET name = COALESCE($3, name),
               config = COALESCE($4, config),
               updated_at = now()
           WHERE id_preset = $1 AND id_user = $2
           RETURNING id_preset, name,
                     config as "config: Value", created_at, updated_at"#,
        id_preset,
        id_user,
        name,
        config,
    )
    .fetch_optional(pool)
    .await
}

/// True when a row was actually removed.
pub async fn delete(pool: &PgPool, id_preset: Uuid, id_user: Uuid) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!(
        "DELETE FROM training_presets WHERE id_preset = $1 AND id_user = $2",
        id_preset,
        id_user,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}
