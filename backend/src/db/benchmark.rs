use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

// One evaluation of one model. A row is inserted as `Running` the moment the run
// starts (so a crashed server leaves visible evidence rather than a silent gap)
// and updated once on completion.
#[derive(Clone, Copy, PartialEq, serde::Serialize, sqlx::Type)]
#[sqlx(type_name = "benchmark_status", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum BenchmarkStatus {
    Running,
    Done,
    Failed,
}

pub struct BenchmarkRun {
    pub id_run: Uuid,
    pub id_model: Uuid,
    pub model_name: String,
    pub status: BenchmarkStatus,
    pub score: Option<f64>,
    pub report: Option<Value>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

// Opens a run. The id is chosen by the caller so the spawned worker thread can
// report progress against it before this insert has necessarily landed.
pub async fn create(
    pool: &PgPool,
    id_run: Uuid,
    id_model: Uuid,
    model_name: &str,
) -> Result<BenchmarkRun, sqlx::Error> {
    sqlx::query_as!(
        BenchmarkRun,
        r#"INSERT INTO benchmark_runs (id_run, id_model, model_name, status)
           VALUES ($1, $2, $3, 'running')
           RETURNING id_run, id_model, model_name,
                     status as "status: BenchmarkStatus", score,
                     report as "report: Value", error, created_at, completed_at"#,
        id_run,
        id_model,
        model_name,
    )
    .fetch_one(pool)
    .await
}

// Closes a run with its report. `score` is duplicated out of the report so the
// history list and trend chart don't have to parse the JSON.
pub async fn complete(
    pool: &PgPool,
    id_run: Uuid,
    score: Option<f64>,
    report: &Value,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"UPDATE benchmark_runs
           SET status = 'done', score = $2, report = $3, completed_at = now()
           WHERE id_run = $1"#,
        id_run,
        score,
        report,
    )
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn fail(pool: &PgPool, id_run: Uuid, error: &str) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"UPDATE benchmark_runs
           SET status = 'failed', error = $2, completed_at = now()
           WHERE id_run = $1"#,
        id_run,
        error,
    )
    .execute(pool)
    .await?;
    Ok(())
}

// Every run, newest first. `report` is deliberately excluded — the list drives a
// history table and a trend chart, and shipping a full report (which embeds
// generated text samples) per row would dominate the response.
pub struct BenchmarkSummary {
    pub id_run: Uuid,
    pub id_model: Uuid,
    pub model_name: String,
    pub status: BenchmarkStatus,
    pub score: Option<f64>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

pub async fn list(pool: &PgPool, limit: i64) -> Result<Vec<BenchmarkSummary>, sqlx::Error> {
    sqlx::query_as!(
        BenchmarkSummary,
        r#"SELECT id_run, id_model, model_name,
                  status as "status: BenchmarkStatus", score, error,
                  created_at, completed_at
           FROM benchmark_runs
           ORDER BY created_at DESC
           LIMIT $1"#,
        limit,
    )
    .fetch_all(pool)
    .await
}

pub async fn find_by_id(pool: &PgPool, id_run: Uuid) -> Result<Option<BenchmarkRun>, sqlx::Error> {
    sqlx::query_as!(
        BenchmarkRun,
        r#"SELECT id_run, id_model, model_name,
                  status as "status: BenchmarkStatus", score,
                  report as "report: Value", error, created_at, completed_at
           FROM benchmark_runs WHERE id_run = $1"#,
        id_run,
    )
    .fetch_optional(pool)
    .await
}

pub async fn delete(pool: &PgPool, id_run: Uuid) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!("DELETE FROM benchmark_runs WHERE id_run = $1", id_run)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// Marks any run still flagged `running` as failed. Called once at startup: a run
// lives on a worker thread with no durable state, so a process that died mid-run
// leaves a row that would otherwise show a spinner forever.
pub async fn fail_orphaned(pool: &PgPool) -> Result<u64, sqlx::Error> {
    let result = sqlx::query!(
        r#"UPDATE benchmark_runs
           SET status = 'failed', error = 'interrupted by a server restart',
               completed_at = now()
           WHERE status = 'running'"#,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}
