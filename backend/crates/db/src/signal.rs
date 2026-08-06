// Gathered replies awaiting a verdict, and the count of verdicts collected so far.
// Pure SQL — what "enough" means lives in `gather.rs`, the HTTP shape in
// `routes/signals.rs`.

use sqlx::PgPool;
use uuid::Uuid;

use crate::message::Feedback;

// One generated reply, as the rating panel sees it. No `feedback` field: every row
// this struct is used for is by definition unrated — freshly inserted, or selected
// by a query whose whole purpose is `WHERE feedback IS NULL` — so carrying the
// column would only ever restate that.
pub struct Sample {
    pub id_sample: Uuid,
    pub id_job: Option<Uuid>,
    pub prompt: String,
    pub response: String,
}

// Records one generated reply, unrated. Called from the queue runner as the worker
// produces them, so a batch can be rated while the rest is still generating.
pub async fn insert(
    pool: &PgPool,
    id_job: Uuid,
    id_model: Uuid,
    prompt: &str,
    response: &str,
) -> Result<Sample, sqlx::Error> {
    sqlx::query_as!(
        Sample,
        r#"INSERT INTO signal_samples (id_job, id_model, prompt, response)
           VALUES ($1, $2, $3, $4)
           RETURNING id_sample, id_job, prompt, response"#,
        id_job,
        id_model,
        prompt,
        response,
    )
    .fetch_one(pool)
    .await
}

// Unrated samples, oldest first. `id_job` narrows to one batch — which is what the
// panel on a job page wants — and `None` returns the whole backlog, so a batch
// abandoned halfway can be finished later from anywhere.
pub async fn list_pending(
    pool: &PgPool,
    id_job: Option<Uuid>,
    limit: i64,
) -> Result<Vec<Sample>, sqlx::Error> {
    sqlx::query_as!(
        Sample,
        r#"SELECT id_sample, id_job, prompt, response
           FROM signal_samples
           WHERE feedback IS NULL
             AND ($1::uuid IS NULL OR id_job = $1)
           ORDER BY created_at
           LIMIT $2"#,
        id_job,
        limit,
    )
    .fetch_all(pool)
    .await
}

// Rates a sample, or clears the rating with `None`. `rated_at` tracks the verdict
// rather than the row, so clearing puts it back in the pending queue exactly as it
// was — a misclick shouldn't cost the sample.
pub async fn set_feedback(
    pool: &PgPool,
    id_sample: Uuid,
    feedback: Option<Feedback>,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!(
        r#"UPDATE signal_samples
           SET feedback = $2,
               rated_at = CASE WHEN $2::message_feedback IS NULL THEN NULL ELSE now() END
           WHERE id_sample = $1"#,
        id_sample,
        feedback as Option<Feedback>,
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

// Drops a sample outright — the "skip" button. Deleted rather than marked skipped:
// an unjudgeable reply carries no signal in either direction, and keeping it would
// mean every pending query had to learn to exclude it.
pub async fn delete(pool: &PgPool, id_sample: Uuid) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!("DELETE FROM signal_samples WHERE id_sample = $1", id_sample)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// How much preference signal exists, split by class.
pub struct SignalCounts {
    pub desirable: i64,
    pub undesirable: i64,
    // Generated but not yet rated. Not signal — it is how much is sitting in the
    // panel waiting to become signal, which is the difference between "gather more"
    // and "finish rating what you have".
    pub pending: i64,
}

// Counts both sources of preference signal exactly as `export_preference_data`
// reads them: rated chat replies plus rated gathered samples.
//
// The `EXISTS` clause is not incidental — the export drops a rated reply with no
// preceding user turn, because a response scored without its prompt measures the
// wrong conditional. Counting it here would let the composer say "ready" and then
// have the dataset build come up short, which is precisely the failure this whole
// gate exists to remove.
pub async fn counts(pool: &PgPool) -> Result<SignalCounts, sqlx::Error> {
    let row = sqlx::query!(
        r#"WITH rated AS (
               SELECT m.feedback
               FROM messages m
               WHERE m.feedback IS NOT NULL
                 AND m.id_user IS NULL
                 AND EXISTS (
                     SELECT 1 FROM messages p
                     WHERE p.id_chat = m.id_chat
                       AND p.id_user IS NOT NULL
                       AND p.created_at < m.created_at
                 )
               UNION ALL
               SELECT s.feedback FROM signal_samples s WHERE s.feedback IS NOT NULL
           )
           SELECT
               COUNT(*) FILTER (WHERE feedback = 'up') as "desirable!",
               COUNT(*) FILTER (WHERE feedback = 'down') as "undesirable!",
               (SELECT COUNT(*) FROM signal_samples WHERE feedback IS NULL) as "pending!"
           FROM rated"#,
    )
    .fetch_one(pool)
    .await?;

    Ok(SignalCounts {
        desirable: row.desirable,
        undesirable: row.undesirable,
        pending: row.pending,
    })
}

// Every rated gathered sample, oldest first — the second half of the preference
// export. Mirrors `db::message::list_rated`, and returns the prompt as a plain
// field because unlike a chat reply this row already carries it.
pub struct RatedSample {
    pub prompt: String,
    pub response: String,
    pub feedback: Option<Feedback>,
}

pub async fn list_rated(pool: &PgPool) -> Result<Vec<RatedSample>, sqlx::Error> {
    sqlx::query_as!(
        RatedSample,
        r#"SELECT prompt, response, feedback as "feedback: Feedback"
           FROM signal_samples
           WHERE feedback IS NOT NULL
           ORDER BY created_at ASC"#,
    )
    .fetch_all(pool)
    .await
}
