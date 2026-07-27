// Queue rows. Pure SQL — the scheduling policy lives in queue.rs, the HTTP shape
// in routes/jobs.rs. This file only knows how to read and write the table.

use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

// Which kind of work a row represents. The runner dispatches on this, and it also
// picks the throbber's mode on the frontend.
#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize, sqlx::Type)]
#[sqlx(type_name = "job_type", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum JobKind {
    Train,
    Benchmark,
    Chat,
    // Generate candidate replies and label them with the rule-based judge, writing
    // preference training data. GPU work like the rest, so it queues like the rest.
    Autolabel,
    // Generate short replies for a *person* to rate, so a KTO run has something to
    // train on. The autolabel twin with the judge replaced by a human.
    Gather,
    // Ask a cheap hosted model for synthetic SFT conversations, validate them, and
    // append the survivors to the SFT corpus. The only kind that touches no local
    // compute at all — see `lane`.
    Synth,
}

// Which runner executes a kind of work.
//
// The GPU line exists because there is one device: a second concurrent CUDA job
// would fight the first for memory and both would be slower than running them in
// order. That argument does not extend to waiting on somebody else's HTTP endpoint,
// which contends for nothing — so network work gets its own runner rather than
// spending the device's line doing nothing.
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, sqlx::Type)]
#[sqlx(type_name = "job_lane", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum JobLane {
    Device,
    Network,
}

impl JobKind {
    // A kind belongs to exactly one lane, and always the same one, so this is a
    // property of the kind rather than something a caller chooses (and could get
    // wrong, putting a training run on the network runner).
    pub fn lane(self) -> JobLane {
        match self {
            JobKind::Synth => JobLane::Network,
            JobKind::Train
            | JobKind::Benchmark
            | JobKind::Chat
            | JobKind::Autolabel
            | JobKind::Gather => JobLane::Device,
        }
    }
}

#[derive(Clone, Copy, PartialEq, serde::Serialize, sqlx::Type)]
#[sqlx(type_name = "job_status", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Running,
    Cancelling,
    Done,
    Cancelled,
    Failed,
}

impl JobStatus {
    // Terminal means "nothing further will happen to this row": no worker holds
    // it, no frame will arrive for it, and it is safe to delete.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobStatus::Done | JobStatus::Cancelled | JobStatus::Failed
        )
    }
}

pub struct Job {
    pub id_job: Uuid,
    pub kind: JobKind,
    // Which runner will execute this row. Derived from `kind` at insert and never
    // changed after: a job cannot move between lanes any more than a training run
    // can stop needing the GPU.
    pub lane: JobLane,
    // Background jobs run only when no foreground job is waiting. Cycle-spawned
    // work is background so an unattended overnight run never makes chat unusable.
    pub background: bool,
    pub status: JobStatus,
    pub label: String,
    pub queue_rank: f64,
    pub config: Value,
    pub progress: f64,
    pub stage: Option<String>,
    pub ref_id: Option<Uuid>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
}

// Enqueues a job at the back of the line. The rank is `max + 1` over the *live*
// band only: terminal rows keep their old ranks and would otherwise drag the
// maximum up forever.
pub async fn create(
    pool: &PgPool,
    kind: JobKind,
    label: &str,
    config: &Value,
    id_user: Option<Uuid>,
) -> Result<Job, sqlx::Error> {
    create_in_band(pool, kind, label, config, id_user, false, None, None).await
}

// Enqueues at the back of a *band*. The rank is `max + 1` over the live jobs in the
// same band only — a foreground job must not be pushed behind a background one just
// because the background one happens to hold the highest rank.
pub async fn create_in_band(
    pool: &PgPool,
    kind: JobKind,
    label: &str,
    config: &Value,
    id_user: Option<Uuid>,
    background: bool,
    id_cycle: Option<Uuid>,
    cycle_role: Option<&str>,
) -> Result<Job, sqlx::Error> {
    sqlx::query_as!(
        Job,
        r#"INSERT INTO jobs (kind, lane, label, config, id_user, background, id_cycle, cycle_role, queue_rank)
           VALUES ($1, $8, $2, $3, $4, $5, $6, $7, (
               SELECT COALESCE(MAX(queue_rank), 0) + 1 FROM jobs
               WHERE status IN ('queued', 'running', 'cancelling')
                 AND lane = $8 AND background = $5
           ))
           RETURNING id_job, kind as "kind: JobKind", lane as "lane: JobLane",
                     status as "status: JobStatus",
                     label, queue_rank, background, config as "config: Value", progress, stage,
                     ref_id, error, created_at, started_at, finished_at"#,
        kind as JobKind,
        label,
        config,
        id_user,
        background,
        id_cycle,
        cycle_role,
        // Not a parameter of this function: the lane follows from the kind, so
        // there is no call site that could disagree with the runner about which
        // line a job is standing in.
        kind.lane() as JobLane,
    )
    .fetch_one(pool)
    .await
}

// The live band — running, cancelling and queued — in the order it will execute.
//
// With two lanes there are two orders, not one, so the list is grouped by lane
// first: within a lane the sequence is exactly what that runner will do, and the
// board renders the groups as the separate lines they are.
pub async fn list_active(pool: &PgPool) -> Result<Vec<Job>, sqlx::Error> {
    sqlx::query_as!(
        Job,
        r#"SELECT id_job, kind as "kind: JobKind", lane as "lane: JobLane",
                  status as "status: JobStatus",
                  label, queue_rank, background, config as "config: Value", progress, stage,
                  ref_id, error, created_at, started_at, finished_at
           FROM jobs
           WHERE status IN ('queued', 'running', 'cancelling')
           ORDER BY lane, background, queue_rank"#,
    )
    .fetch_all(pool)
    .await
}

// Finished jobs, newest first. Separate from `list_active` because the two are
// ordered by different keys and only the history needs a limit.
pub async fn list_history(pool: &PgPool, limit: i64) -> Result<Vec<Job>, sqlx::Error> {
    sqlx::query_as!(
        Job,
        r#"SELECT id_job, kind as "kind: JobKind", lane as "lane: JobLane",
                  status as "status: JobStatus",
                  label, queue_rank, background, config as "config: Value", progress, stage,
                  ref_id, error, created_at, started_at, finished_at
           FROM jobs
           WHERE status IN ('done', 'cancelled', 'failed')
           ORDER BY created_at DESC
           LIMIT $1"#,
        limit,
    )
    .fetch_all(pool)
    .await
}

pub async fn find_by_id(pool: &PgPool, id_job: Uuid) -> Result<Option<Job>, sqlx::Error> {
    sqlx::query_as!(
        Job,
        r#"SELECT id_job, kind as "kind: JobKind", lane as "lane: JobLane",
                  status as "status: JobStatus",
                  label, queue_rank, background, config as "config: Value", progress, stage,
                  ref_id, error, created_at, started_at, finished_at
           FROM jobs WHERE id_job = $1"#,
        id_job,
    )
    .fetch_optional(pool)
    .await
}

// Takes the next queued job and marks it running, atomically. `FOR UPDATE SKIP
// LOCKED` is the standard Postgres queue take: it makes the claim safe if a second
// backend process ever exists, and costs nothing with one. `None` means the queue
// is empty — the runner then waits to be woken rather than spinning.
//
// Each runner claims only from its own lane. `SKIP LOCKED` is what makes two of
// them safe against each other even in the moment they claim simultaneously, and
// the lane filter means they never want the same row anyway.
pub async fn claim_next(pool: &PgPool, lane: JobLane) -> Result<Option<Job>, sqlx::Error> {
    sqlx::query_as!(
        Job,
        r#"UPDATE jobs SET status = 'running', started_at = now()
           WHERE id_job = (
               SELECT id_job FROM jobs WHERE status = 'queued' AND lane = $1
               -- Band first: a queued foreground job always outranks every
               -- background one, so a running cycle never delays a chat turn by
               -- more than the job already on the device.
               ORDER BY background, queue_rank
               LIMIT 1
               FOR UPDATE SKIP LOCKED
           )
           RETURNING id_job, kind as "kind: JobKind", lane as "lane: JobLane",
                     status as "status: JobStatus",
                     label, queue_rank, background, config as "config: Value", progress, stage,
                     ref_id, error, created_at, started_at, finished_at"#,
        lane as JobLane,
    )
    .fetch_optional(pool)
    .await
}

// Coarse progress for a running job. Called on a throttle (see queue.rs) — this is
// the restart-durable summary, not the live frame stream.
pub async fn set_progress(
    pool: &PgPool,
    id_job: Uuid,
    progress: f64,
    stage: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        "UPDATE jobs SET progress = $2, stage = $3 WHERE id_job = $1",
        id_job,
        progress,
        stage,
    )
    .execute(pool)
    .await?;
    Ok(())
}

// Closes a job out. The caller passes the final `progress` (1.0 on success) rather
// than having the SQL derive it, so a cancelled run keeps the fraction it actually
// reached instead of being rounded to either end.
pub async fn finish(
    pool: &PgPool,
    id_job: Uuid,
    status: JobStatus,
    progress: f64,
    ref_id: Option<Uuid>,
    error: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"UPDATE jobs
           SET status = $2, progress = $3, ref_id = $4, error = $5,
               finished_at = now()
           WHERE id_job = $1"#,
        id_job,
        status as JobStatus,
        progress,
        ref_id,
        error,
    )
    .execute(pool)
    .await?;
    Ok(())
}

// Cancels a job that hasn't started. Conditional on the row still being `queued`,
// so a job the runner claimed in the meantime isn't quietly marked cancelled while
// a worker is still running it — the caller checks the returned flag and falls
// back to `request_cancel`.
pub async fn cancel_queued(pool: &PgPool, id_job: Uuid) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!(
        r#"UPDATE jobs SET status = 'cancelled', finished_at = now()
           WHERE id_job = $1 AND status = 'queued'"#,
        id_job,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

// Marks a running job as cancelling. The worker's own atomic flag is what actually
// stops it; this only records that the request was made, so the UI can say so
// during the gap before the next step boundary.
pub async fn request_cancel(pool: &PgPool, id_job: Uuid) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!(
        r#"UPDATE jobs SET status = 'cancelling'
           WHERE id_job = $1 AND status = 'running'"#,
        id_job,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

pub async fn delete(pool: &PgPool, id_job: Uuid) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!("DELETE FROM jobs WHERE id_job = $1", id_job)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// The bounds a job would land between if placed directly after `after` (or at the
// front, when `after` is None), for `reorder` to bisect.
async fn neighbour_ranks(
    pool: &PgPool,
    moving: Uuid,
    after: Option<Uuid>,
    lane: JobLane,
    background: bool,
) -> Result<(Option<f64>, Option<f64>), sqlx::Error> {
    // The anchor's own rank is the lower bound. Front-of-queue has no lower bound.
    let lower = match after {
        Some(id) => match sqlx::query!("SELECT queue_rank FROM jobs WHERE id_job = $1", id)
            .fetch_optional(pool)
            .await?
        {
            Some(row) => Some(row.queue_rank),
            // The anchor vanished between the drag and the drop.
            None => return Ok((None, None)),
        },
        None => None,
    };

    // The next live job after the anchor, excluding the one being moved — it is
    // currently sitting somewhere in this band and must not bound itself.
    // Bounded within the moving job's own lane *and* band: ranks only order jobs
    // against others they actually queue behind, so a neighbour from another lane
    // or band would bisect against a number that means nothing here.
    let upper = sqlx::query!(
        r#"SELECT MIN(queue_rank) as "rank" FROM jobs
           WHERE status IN ('queued', 'running', 'cancelling')
             AND id_job <> $1
             AND lane = $4 AND background = $3
             AND ($2::double precision IS NULL OR queue_rank > $2)"#,
        moving,
        lower,
        background,
        lane as JobLane,
    )
    .fetch_one(pool)
    .await?
    .rank;

    Ok((lower, upper))
}

// Gaps below this can't be bisected much further in f64, so the band gets
// renumbered instead. Reordering across the same pair ~50 times reaches it.
const MIN_GAP: f64 = 1e-9;

// Moves a queued job to sit directly after `after` (or to the front). Writes one
// row in the common case; renormalizes the whole live band to 1, 2, 3… only when
// the midpoint would collapse.
pub async fn reorder(pool: &PgPool, id_job: Uuid, after: Option<Uuid>) -> Result<(), sqlx::Error> {
    // A job only ever moves within its own lane and band; there is no drag that
    // promotes background work past foreground work or moves a job onto the other
    // runner, by construction.
    let row = sqlx::query!(
        r#"SELECT lane as "lane: JobLane", background FROM jobs WHERE id_job = $1"#,
        id_job,
    )
    .fetch_optional(pool)
    .await?;
    let (lane, background) = match row {
        Some(row) => (row.lane, row.background),
        // The job vanished between the drag and the drop; the UPDATE below will
        // match nothing, which is the right outcome.
        None => (JobLane::Device, false),
    };

    let (lower, upper) = neighbour_ranks(pool, id_job, after, lane, background).await?;

    let rank = match (lower, upper) {
        // Between two jobs: the midpoint, unless there's no room left.
        (Some(lo), Some(hi)) if hi - lo > MIN_GAP => (lo + hi) / 2.0,
        (Some(lo), Some(_)) => {
            renormalize(pool, lane, background).await?;
            // Ranks are whole numbers again, so re-bisecting now succeeds.
            match neighbour_ranks(pool, id_job, after, lane, background).await? {
                (Some(lo2), Some(hi2)) => (lo2 + hi2) / 2.0,
                (Some(lo2), None) => lo2 + 1.0,
                _ => lo + 1.0,
            }
        }
        // Last in the line.
        (Some(lo), None) => lo + 1.0,
        // First in the line: half of whatever currently leads.
        (None, Some(hi)) => hi / 2.0,
        // Nothing else is live, so any rank will do.
        (None, None) => 1.0,
    };

    sqlx::query!(
        "UPDATE jobs SET queue_rank = $2 WHERE id_job = $1 AND status = 'queued'",
        id_job,
        rank,
    )
    .execute(pool)
    .await?;
    Ok(())
}

// Rewrites the live band's ranks as 1, 2, 3…, preserving the current order. Only
// called when bisection runs out of precision.
async fn renormalize(pool: &PgPool, lane: JobLane, background: bool) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"UPDATE jobs SET queue_rank = ordered.position
           FROM (
               SELECT id_job, ROW_NUMBER() OVER (ORDER BY queue_rank) AS position
               FROM jobs
               WHERE status IN ('queued', 'running', 'cancelling')
                 AND lane = $2 AND background = $1
           ) AS ordered
           WHERE jobs.id_job = ordered.id_job"#,
        background,
        lane as JobLane,
    )
    .execute(pool)
    .await?;
    Ok(())
}

// Closes out jobs left mid-flight by a process that died. Called once at startup:
// a worker holds no durable state of its own, so a row still marked `running` has
// no thread behind it and would otherwise show a throbber that never resolves
// while holding the head of the queue forever. Queued rows are deliberately
// untouched — they never started, and draining them is what should happen on boot.
pub async fn fail_orphaned(pool: &PgPool) -> Result<u64, sqlx::Error> {
    let result = sqlx::query!(
        r#"UPDATE jobs
           SET status = 'failed', error = 'interrupted by a server restart',
               finished_at = now()
           WHERE status IN ('running', 'cancelling')"#,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

pub async fn is_paused(pool: &PgPool) -> Result<bool, sqlx::Error> {
    let row = sqlx::query!("SELECT paused FROM queue_state WHERE only_row")
        .fetch_one(pool)
        .await?;
    Ok(row.paused)
}

pub async fn set_paused(pool: &PgPool, paused: bool) -> Result<(), sqlx::Error> {
    sqlx::query!(
        "UPDATE queue_state SET paused = $1, updated_at = now() WHERE only_row",
        paused,
    )
    .execute(pool)
    .await?;
    Ok(())
}
