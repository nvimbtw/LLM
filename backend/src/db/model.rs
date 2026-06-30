use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

// A trained model as stored in the registry. `parameters` holds the full
// training config + dataset stats as JSON (inspectable, not queried column-wise);
// `context` mirrors the config's ctx_window for quick display. Weights themselves
// live on disk under storage::models_dir()/<name>, not here.
pub struct Model {
    pub id_model: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    pub context: Option<i32>,
    pub family: Option<String>,
    pub created_at: DateTime<Utc>,
}

// Input for persisting a freshly trained model. Built by the training worker on
// a successful run and inserted via `upsert`.
pub struct NewModel {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub context: i32,
    pub family: Option<String>,
}

// Inserts a model, or updates it in place if the name already exists. Retraining
// reuses the model_name (and overwrites its on-disk directory), so a second run
// of the same name should refresh the registry row rather than collide on the
// UNIQUE(name) constraint. A user-set `description` is deliberately preserved on
// conflict — only the training-derived fields are refreshed.
pub async fn upsert(pool: &PgPool, model: &NewModel) -> Result<Model, sqlx::Error> {
    sqlx::query_as!(
        Model,
        r#"INSERT INTO models (name, description, parameters, context, family)
           VALUES ($1, $2, $3, $4, $5)
           ON CONFLICT (name) DO UPDATE
           SET parameters = EXCLUDED.parameters,
               context = EXCLUDED.context,
               family = EXCLUDED.family
           RETURNING id_model, name, description,
                     parameters as "parameters: Value", context, family, created_at"#,
        model.name,
        model.description,
        model.parameters,
        model.context,
        model.family,
    )
    .fetch_one(pool)
    .await
}

pub async fn list(pool: &PgPool) -> Result<Vec<Model>, sqlx::Error> {
    // Newest first, for the registry/sidebar recency ordering.
    sqlx::query_as!(
        Model,
        r#"SELECT id_model, name, description,
                  parameters as "parameters: Value", context, family, created_at
           FROM models ORDER BY created_at DESC"#,
    )
    .fetch_all(pool)
    .await
}

pub async fn find_by_id(pool: &PgPool, id: Uuid) -> Result<Option<Model>, sqlx::Error> {
    sqlx::query_as!(
        Model,
        r#"SELECT id_model, name, description,
                  parameters as "parameters: Value", context, family, created_at
           FROM models WHERE id_model = $1"#,
        id,
    )
    .fetch_optional(pool)
    .await
}

// `name`/`description` are optional so a PUT can update either independently;
// COALESCE keeps the existing column value when an arg is NULL.
pub async fn update(
    pool: &PgPool,
    id: Uuid,
    name: Option<&str>,
    description: Option<&str>,
    family: Option<&str>,
) -> Result<Option<Model>, sqlx::Error> {
    sqlx::query_as!(
        Model,
        r#"UPDATE models
           SET name = COALESCE($2, name),
               description = COALESCE($3, description),
               family = COALESCE($4, family)
           WHERE id_model = $1
           RETURNING id_model, name, description,
                     parameters as "parameters: Value", context, family, created_at"#,
        id,
        name,
        description,
        family,
    )
    .fetch_optional(pool)
    .await
}

// Deletes a model and detaches any messages that reference it. `messages.id_model`
// is a nullable FK with no ON DELETE rule, so a referenced model can't be deleted
// outright — we null the references first, in one transaction. Returns the deleted
// model's name (so the caller can remove its on-disk directory) or None if no such
// model existed.
pub async fn delete(pool: &PgPool, id: Uuid) -> Result<Option<String>, sqlx::Error> {
    let mut tx = pool.begin().await?;

    sqlx::query!("UPDATE messages SET id_model = NULL WHERE id_model = $1", id)
        .execute(&mut *tx)
        .await?;

    let deleted = sqlx::query!(
        "DELETE FROM models WHERE id_model = $1 RETURNING name",
        id,
    )
    .fetch_optional(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(deleted.map(|row| row.name))
}
