use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

// A named training-parameter set as stored in the registry. `config` holds a
// TrainConfig-shaped JSON blob (inspectable, not queried column-wise), the same
// approach as `models.parameters`. Presets are personal: every query below is
// scoped to `id_user` so ownership is enforced in SQL, not just in the handler.
pub struct Preset {
    pub id_preset: Uuid,
    pub name: String,
    pub config: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// Newest-updated first, so the picker surfaces recently-touched presets. Scoped
// to the owner.
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

// Inserts a new preset for the owner. May surface a UNIQUE(id_user, name)
// violation, which the route layer maps to 409.
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

// Updates a preset in place, owner-scoped. `name`/`config` are optional so a PUT
// can change either independently; COALESCE keeps the existing value when an arg
// is NULL. `updated_at` always advances so the recency ordering stays meaningful.
// Returns None when no preset with that id belongs to this user. A rename that
// collides with another of the user's presets surfaces a UNIQUE violation (→ 409).
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

// Deletes a preset, owner-scoped. Returns true if a row was removed, false if no
// such preset belongs to this user (→ 404).
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
