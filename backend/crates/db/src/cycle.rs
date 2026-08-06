// Post-training cycles and their rounds. Pure SQL — the state machine lives in
// `cycle.rs`, the HTTP shape in `routes/cycles.rs`.

use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Clone, Copy, PartialEq, serde::Serialize, sqlx::Type)]
#[sqlx(type_name = "cycle_status", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum CycleStatus {
    Running,
    Paused,
    Halted,
    Done,
}

impl CycleStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, CycleStatus::Halted | CycleStatus::Done)
    }
}

#[derive(Clone, Copy, PartialEq, serde::Serialize, sqlx::Type)]
#[sqlx(type_name = "cycle_stage", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum CycleStage {
    Generating,
    Selftrain,
    Preference,
    Benchmarking,
    Deciding,
}

pub struct Cycle {
    pub id_cycle: Uuid,
    pub label: String,
    pub status: CycleStatus,
    pub stage: CycleStage,
    pub round: i32,
    pub max_rounds: i32,
    pub id_model_start: Option<Uuid>,
    pub id_model_current: Option<Uuid>,
    pub id_model_best: Option<Uuid>,
    pub best_score: Option<f64>,
    pub id_job_current: Option<Uuid>,
    pub config: Value,
    pub halt_reason: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
}

// The column list is repeated in each query rather than shared through a constant:
// `query_as!` checks the SQL at compile time, which needs a literal.

pub async fn create(
    pool: &PgPool,
    label: &str,
    id_model_start: Uuid,
    max_rounds: i32,
    config: &Value,
    id_user: Option<Uuid>,
) -> Result<Cycle, sqlx::Error> {
    sqlx::query_as!(
        Cycle,
        r#"INSERT INTO cycles (label, id_model_start, id_model_current, max_rounds, config, id_user)
           VALUES ($1, $2, $2, $3, $4, $5)
           RETURNING id_cycle, label, status as "status: CycleStatus",
                     stage as "stage: CycleStage", round, max_rounds, id_model_start,
                     id_model_current, id_model_best, best_score, id_job_current,
                     config as "config: Value", halt_reason, created_at, updated_at, finished_at"#,
        label,
        id_model_start,
        max_rounds,
        config,
        id_user,
    )
    .fetch_one(pool)
    .await
}

// Cycles the driver still has work to do for. Paused ones are included so a resume
// is a status write rather than a re-registration.
pub async fn list_active(pool: &PgPool) -> Result<Vec<Cycle>, sqlx::Error> {
    sqlx::query_as!(
        Cycle,
        r#"SELECT id_cycle, label, status as "status: CycleStatus",
                  stage as "stage: CycleStage", round, max_rounds, id_model_start,
                  id_model_current, id_model_best, best_score, id_job_current,
                  config as "config: Value", halt_reason, created_at, updated_at, finished_at
           FROM cycles
           WHERE status IN ('running', 'paused')
           ORDER BY created_at"#,
    )
    .fetch_all(pool)
    .await
}

pub async fn list_all(pool: &PgPool, limit: i64) -> Result<Vec<Cycle>, sqlx::Error> {
    sqlx::query_as!(
        Cycle,
        r#"SELECT id_cycle, label, status as "status: CycleStatus",
                  stage as "stage: CycleStage", round, max_rounds, id_model_start,
                  id_model_current, id_model_best, best_score, id_job_current,
                  config as "config: Value", halt_reason, created_at, updated_at, finished_at
           FROM cycles
           ORDER BY created_at DESC
           LIMIT $1"#,
        limit,
    )
    .fetch_all(pool)
    .await
}

pub async fn find_by_id(pool: &PgPool, id_cycle: Uuid) -> Result<Option<Cycle>, sqlx::Error> {
    sqlx::query_as!(
        Cycle,
        r#"SELECT id_cycle, label, status as "status: CycleStatus",
                  stage as "stage: CycleStage", round, max_rounds, id_model_start,
                  id_model_current, id_model_best, best_score, id_job_current,
                  config as "config: Value", halt_reason, created_at, updated_at, finished_at
           FROM cycles WHERE id_cycle = $1"#,
        id_cycle,
    )
    .fetch_optional(pool)
    .await
}

pub async fn set_stage(
    pool: &PgPool,
    id_cycle: Uuid,
    stage: CycleStage,
    id_job: Option<Uuid>,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"UPDATE cycles SET stage = $2, id_job_current = $3, updated_at = now()
           WHERE id_cycle = $1"#,
        id_cycle,
        stage as CycleStage,
        id_job,
    )
    .execute(pool)
    .await?;
    Ok(())
}

// Starts the next round: bump the counter, back to the first stage, no current job.
pub async fn advance_round(
    pool: &PgPool,
    id_cycle: Uuid,
    id_model_current: Uuid,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"UPDATE cycles
           SET round = round + 1, stage = 'generating', id_job_current = NULL,
               id_model_current = $2, updated_at = now()
           WHERE id_cycle = $1"#,
        id_cycle,
        id_model_current,
    )
    .execute(pool)
    .await?;
    Ok(())
}

// Records a new best. The cycle's actual output: when a later round fails to
// improve, this is the model to keep, not whatever the last round produced.
pub async fn set_best(
    pool: &PgPool,
    id_cycle: Uuid,
    id_model: Uuid,
    score: f64,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"UPDATE cycles SET id_model_best = $2, best_score = $3, updated_at = now()
           WHERE id_cycle = $1"#,
        id_cycle,
        id_model,
        score,
    )
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn set_status(
    pool: &PgPool,
    id_cycle: Uuid,
    status: CycleStatus,
    halt_reason: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"UPDATE cycles
           SET status = $2,
               halt_reason = COALESCE($3, halt_reason),
               -- Cast explicit: without it the same placeholder is deduced as
               -- cycle_status in the SET and as text in the IN, which is ambiguous.
               finished_at = CASE
                   WHEN $2::cycle_status IN ('halted', 'done') THEN now()
                   ELSE NULL
               END,
               updated_at = now()
           WHERE id_cycle = $1"#,
        id_cycle,
        status as CycleStatus,
        halt_reason,
    )
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn delete(pool: &PgPool, id_cycle: Uuid) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!("DELETE FROM cycles WHERE id_cycle = $1", id_cycle)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// ── rounds ──────────────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
pub struct Round {
    pub id_round: Uuid,
    pub round: i32,
    pub approved_before: Option<i64>,
    pub rejected_before: Option<i64>,
    pub approved_after: Option<i64>,
    pub rejected_after: Option<i64>,
    // The model that generated this round's catalogue, and so the model whose
    // generations both of its training stages are scoped to.
    pub id_model_in: Option<Uuid>,
    pub id_model_out: Option<Uuid>,
    pub score: Option<f64>,
    pub verdict: Option<String>,
    pub note: Option<String>,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
}

// Opens a round with the catalogue size going in. `ON CONFLICT DO NOTHING` against
// the (cycle, round) unique key: the driver is edge-triggered off job events, and a
// duplicate event must not open a second row for the same round.
pub async fn open_round(
    pool: &PgPool,
    id_cycle: Uuid,
    round: i32,
    id_model_in: Uuid,
    approved: i64,
    rejected: i64,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"INSERT INTO cycle_rounds (id_cycle, round, id_model_in, approved_before, rejected_before)
           VALUES ($1, $2, $3, $4, $5)
           ON CONFLICT (id_cycle, round) DO NOTHING"#,
        id_cycle,
        round,
        id_model_in,
        approved,
        rejected,
    )
    .execute(pool)
    .await?;
    Ok(())
}

// The model that generated a round's catalogue.
//
// Read back rather than carried on the `Cycle` row because `id_model_current`
// moves during the round: the self-train stage replaces it with its own output,
// so by the preference stage it no longer names the model whose generations both
// stages are meant to train on.
//
// `None` covers two cases the caller must treat alike — no such round, and a round
// opened before the column existed. Both mean "there is no recorded slice", and
// the honest answer for both is to fall back to the whole catalogue.
pub async fn round_model_in(
    pool: &PgPool,
    id_cycle: Uuid,
    round: i32,
) -> Result<Option<Uuid>, sqlx::Error> {
    let row = sqlx::query!(
        "SELECT id_model_in FROM cycle_rounds WHERE id_cycle = $1 AND round = $2",
        id_cycle,
        round,
    )
    .fetch_optional(pool)
    .await?;
    Ok(row.and_then(|r| r.id_model_in))
}

// The catalogue size coming out of the generation stage. With the "before" pair
// this is what makes yield a delta rather than a running total.
pub async fn record_catalogue(
    pool: &PgPool,
    id_cycle: Uuid,
    round: i32,
    approved: i64,
    rejected: i64,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"UPDATE cycle_rounds SET approved_after = $3, rejected_after = $4
           WHERE id_cycle = $1 AND round = $2"#,
        id_cycle,
        round,
        approved,
        rejected,
    )
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn close_round(
    pool: &PgPool,
    id_cycle: Uuid,
    round: i32,
    id_model_out: Option<Uuid>,
    score: Option<f64>,
    verdict: &str,
    note: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"UPDATE cycle_rounds
           SET id_model_out = $3, score = $4, verdict = $5, note = $6, finished_at = now()
           WHERE id_cycle = $1 AND round = $2"#,
        id_cycle,
        round,
        id_model_out,
        score,
        verdict,
        note,
    )
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn list_rounds(pool: &PgPool, id_cycle: Uuid) -> Result<Vec<Round>, sqlx::Error> {
    sqlx::query_as!(
        Round,
        r#"SELECT id_round, round, approved_before, rejected_before, approved_after,
                  rejected_after, id_model_in, id_model_out, score, verdict, note,
                  started_at, finished_at
           FROM cycle_rounds WHERE id_cycle = $1 ORDER BY round"#,
        id_cycle,
    )
    .fetch_all(pool)
    .await
}

// The jobs a cycle has spawned, for the timeline on its page.
pub struct CycleJob {
    pub id_job: Uuid,
    pub kind: super::job::JobKind,
    pub status: super::job::JobStatus,
    pub label: String,
    pub cycle_role: Option<String>,
    pub created_at: DateTime<Utc>,
}

pub async fn list_jobs(pool: &PgPool, id_cycle: Uuid) -> Result<Vec<CycleJob>, sqlx::Error> {
    sqlx::query_as!(
        CycleJob,
        r#"SELECT id_job, kind as "kind: super::job::JobKind",
                  status as "status: super::job::JobStatus", label, cycle_role, created_at
           FROM jobs WHERE id_cycle = $1 ORDER BY created_at"#,
        id_cycle,
    )
    .fetch_all(pool)
    .await
}
