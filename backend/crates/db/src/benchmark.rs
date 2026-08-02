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

// Which benchmark produced this run.
//
// `Language` measures whether the model can model English — held-out bits per byte,
// minimal-pair grammar, the surface properties of its own prose. `Markup` measures
// whether it uses its special tokens correctly. They are separate rows because a
// supervised fine-tune is expected to move them in opposite directions, and a single
// blended number cannot report that: the whole reason for the split was a fine-tune
// that improved on three of four language components, lost two grammar pairs, and
// read as a regression.
#[derive(
    Clone, Copy, PartialEq, Debug, Default, serde::Serialize, serde::Deserialize, sqlx::Type,
)]
#[sqlx(type_name = "benchmark_suite", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum BenchmarkSuite {
    // Callers that predate the split — and any stored config without the field —
    // mean the language suite, which is what every existing row is.
    #[default]
    Language,
    Markup,
    // Held-out bits per assistant-reply token, mapped to 0–100. The post-train
    // cycle's actual objective, and for a long time the only one of the three with
    // no suite of its own — it lived inside the language report as a field nothing
    // scored.
    Assistant,
}

impl BenchmarkSuite {
    pub fn as_str(self) -> &'static str {
        match self {
            BenchmarkSuite::Language => "language",
            BenchmarkSuite::Markup => "markup",
            BenchmarkSuite::Assistant => "assistant",
        }
    }
}


pub struct BenchmarkRun {
    pub id_run: Uuid,
    pub id_model: Uuid,
    pub model_name: String,
    pub status: BenchmarkStatus,
    pub suite: BenchmarkSuite,
    pub suite_version: i32,
    pub score: Option<f64>,
    pub report: Option<Value>,
    pub error: Option<String>,
    // Tokens the model was trained on, snapshotted at run start (see the migration
    // that added the column). None for a model whose config didn't record it.
    pub train_tokens: Option<i64>,
    // Tokens the model forwarded to complete the run, lifted out of the report by
    // `complete`. None while the run is open, and for any run older than the
    // counter — the charts leave those off rather than plotting them at zero.
    pub eval_tokens: Option<i64>,
    // The thinking budget every held-out conversation was rendered under, as the
    // marker's own name (`low`/`med`/`high`). None means each kept the budget it was
    // written with — the absence of a forced marker, not a fourth value.
    pub think_budget: Option<String>,
    // The sweep this run belongs to, when it was queued as one of three. None for a
    // run queued on its own.
    pub id_series: Option<Uuid>,
    // Held-out conversations scored, lifted from the report by `complete`.
    pub eval_conversations: Option<i64>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

// Opens a run. The id is chosen by the caller so the spawned worker thread can
// report progress against it before this insert has necessarily landed.
// `train_tokens` comes from db::model::train_tokens on the model being scored.
//
// `think_budget` and `id_series` are stored at open rather than at completion: a
// sweep's three rows have to be identifiable as a sweep while they are still
// running, or the board shows three identical-looking benchmarks of one model.
#[allow(clippy::too_many_arguments)]
pub async fn create(
    pool: &PgPool,
    id_run: Uuid,
    id_model: Uuid,
    model_name: &str,
    train_tokens: Option<i64>,
    suite: BenchmarkSuite,
    suite_version: i32,
    think_budget: Option<&str>,
    id_series: Option<Uuid>,
) -> Result<BenchmarkRun, sqlx::Error> {
    sqlx::query_as!(
        BenchmarkRun,
        r#"INSERT INTO benchmark_runs
               (id_run, id_model, model_name, status, train_tokens, suite, suite_version,
                think_budget, id_series)
           VALUES ($1, $2, $3, 'running', $4, $5, $6, $7, $8)
           RETURNING id_run, id_model, model_name,
                     status as "status: BenchmarkStatus",
                     suite as "suite: BenchmarkSuite", suite_version, score,
                     report as "report: Value", error, train_tokens, eval_tokens,
                     think_budget, id_series, eval_conversations,
                     created_at, completed_at"#,
        id_run,
        id_model,
        model_name,
        train_tokens,
        suite as BenchmarkSuite,
        suite_version,
        think_budget,
        id_series,
    )
    .fetch_one(pool)
    .await
}

// Closes a run with its report. `score`, `eval_tokens` and `eval_conversations` are
// duplicated out of the report so the history list and the trend charts don't have to
// parse the JSON.
pub async fn complete(
    pool: &PgPool,
    id_run: Uuid,
    score: Option<f64>,
    eval_tokens: Option<i64>,
    eval_conversations: Option<i64>,
    report: &Value,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"UPDATE benchmark_runs
           SET status = 'done', score = $2, eval_tokens = $3,
               eval_conversations = $4, report = $5, completed_at = now()
           WHERE id_run = $1"#,
        id_run,
        score,
        eval_tokens,
        eval_conversations,
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
    pub suite: BenchmarkSuite,
    pub suite_version: i32,
    pub score: Option<f64>,
    pub error: Option<String>,
    pub train_tokens: Option<i64>,
    pub eval_tokens: Option<i64>,
    pub think_budget: Option<String>,
    pub id_series: Option<Uuid>,
    pub eval_conversations: Option<i64>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

pub async fn list(
    pool: &PgPool,
    suite: BenchmarkSuite,
    limit: i64,
) -> Result<Vec<BenchmarkSummary>, sqlx::Error> {
    sqlx::query_as!(
        BenchmarkSummary,
        r#"SELECT id_run, id_model, model_name,
                  status as "status: BenchmarkStatus",
                  suite as "suite: BenchmarkSuite", suite_version, score, error,
                  train_tokens, eval_tokens, think_budget, id_series,
                  eval_conversations, created_at, completed_at
           FROM benchmark_runs
           WHERE suite = $1
           ORDER BY created_at DESC
           LIMIT $2"#,
        suite as BenchmarkSuite,
        limit,
    )
    .fetch_all(pool)
    .await
}

// Every run of one model, newest first — the model page's "how did it score"
// section. Same summary shape as `list`, so the two share a response type.
pub async fn list_for_model(
    pool: &PgPool,
    id_model: Uuid,
    suite: BenchmarkSuite,
    limit: i64,
) -> Result<Vec<BenchmarkSummary>, sqlx::Error> {
    sqlx::query_as!(
        BenchmarkSummary,
        r#"SELECT id_run, id_model, model_name,
                  status as "status: BenchmarkStatus",
                  suite as "suite: BenchmarkSuite", suite_version, score, error,
                  train_tokens, eval_tokens, think_budget, id_series,
                  eval_conversations, created_at, completed_at
           FROM benchmark_runs
           WHERE id_model = $1 AND suite = $2
           ORDER BY created_at DESC
           LIMIT $3"#,
        id_model,
        suite as BenchmarkSuite,
        limit,
    )
    .fetch_all(pool)
    .await
}

// The full report of a model's most recent completed run, for the same section:
// the scores it shows are the detail of one run, and this is the run meant.
pub async fn latest_report_for_model(
    pool: &PgPool,
    id_model: Uuid,
    suite: BenchmarkSuite,
) -> Result<Option<BenchmarkRun>, sqlx::Error> {
    sqlx::query_as!(
        BenchmarkRun,
        r#"SELECT id_run, id_model, model_name,
                  status as "status: BenchmarkStatus",
                  suite as "suite: BenchmarkSuite", suite_version, score,
                  report as "report: Value", error, train_tokens, eval_tokens,
                  think_budget, id_series, eval_conversations,
                  created_at, completed_at
           FROM benchmark_runs
           WHERE id_model = $1 AND suite = $2
             AND status = 'done' AND report IS NOT NULL
           ORDER BY created_at DESC LIMIT 1"#,
        id_model,
        suite as BenchmarkSuite,
    )
    .fetch_optional(pool)
    .await
}

pub async fn find_by_id(pool: &PgPool, id_run: Uuid) -> Result<Option<BenchmarkRun>, sqlx::Error> {
    sqlx::query_as!(
        BenchmarkRun,
        r#"SELECT id_run, id_model, model_name,
                  status as "status: BenchmarkStatus",
                  suite as "suite: BenchmarkSuite", suite_version, score,
                  report as "report: Value", error, train_tokens, eval_tokens,
                  think_budget, id_series, eval_conversations,
                  created_at, completed_at
           FROM benchmark_runs WHERE id_run = $1"#,
        id_run,
    )
    .fetch_optional(pool)
    .await
}

// The most recent completed score for a model, if it has ever been benchmarked.
//
// Used by the cycle to give round 1 something to beat: without it the first round
// compares against nothing and accepts whatever it produced, however much worse
// than the model the cycle started from.
// The suite argument is not optional and must not be defaulted at the call site: the
// cycle judges rounds on one suite's scale, and letting a markup run answer "the
// latest score for this model" would compare two numbers that share only a range.
pub async fn latest_score_for_model(
    pool: &PgPool,
    id_model: Uuid,
    suite: BenchmarkSuite,
) -> Result<Option<f64>, sqlx::Error> {
    let row = sqlx::query!(
        r#"SELECT score FROM benchmark_runs
           WHERE id_model = $1 AND suite = $2
             AND status = 'done' AND score IS NOT NULL
           ORDER BY created_at DESC LIMIT 1"#,
        id_model,
        suite as BenchmarkSuite,
    )
    .fetch_optional(pool)
    .await?;
    Ok(row.and_then(|r| r.score))
}

/// What one model's latest run of a suite says about it — the numbers `jobs::cycle`
/// weighs a round on, and their uncertainties.
///
/// Separate from `latest_score_for_model` rather than replacing it, because the two
/// answer different questions: that one is "what does the trend chart show for this
/// model", this one is "may the cycle accept this round". Only the second needs to
/// refuse a run it cannot compare against.
/// One suite's verdict on one model, and nothing suite-specific: every suite reports
/// a 0–100 `score` and a `breakdown.score_stderr`, so the cycle can weigh three of
/// these against three others without knowing what any of them measured.
///
/// It used to carry `chat_bits_per_token` alongside the composite, because the reply
/// metric had no suite of its own and had to be smuggled out of the language report.
/// Now that `BenchmarkSuite::Assistant` scores it directly, that is one more suite's
/// `score` and the special case is gone.
pub struct RoundEvidence {
    pub score: f64,
    // From `breakdown.score_stderr`. NaN if an older report predates the field, which
    // the caller must treat as "no error bar", not as "zero error".
    pub score_stderr: f64,
}

/// The most recent completed run of `suite` at exactly `suite_version`.
///
/// **The version filter is the point.** `latest_score_for_model` has none, so once a
/// scoring constant moves it happily returns a number from the old formula and the
/// cycle compares two scales as though they were one — the exact failure
/// `SCORE_VERSION` exists to make visible. A round judged against a bar it cannot be
/// compared to is worse than a round judged against nothing, because it looks valid.
/// Returning `None` here makes the cycle say so instead.
pub async fn latest_evidence_for_model(
    pool: &PgPool,
    id_model: Uuid,
    suite: BenchmarkSuite,
    suite_version: i32,
) -> Result<Option<RoundEvidence>, sqlx::Error> {
    let row = sqlx::query!(
        r#"SELECT score, report FROM benchmark_runs
           WHERE id_model = $1 AND suite = $2 AND suite_version = $3
             AND status = 'done' AND score IS NOT NULL
           ORDER BY created_at DESC LIMIT 1"#,
        id_model,
        suite as BenchmarkSuite,
        suite_version,
    )
    .fetch_optional(pool)
    .await?;

    let Some(row) = row else {
        return Ok(None);
    };
    let Some(score) = row.score else {
        return Ok(None);
    };

    // Read out of the stored report rather than added as columns: these are the
    // report's own fields, and a column per metric means a migration every time the
    // suite learns to measure something. A missing field reads as absent, which is
    // what an older report genuinely is.
    let report = row.report.unwrap_or(serde_json::Value::Null);
    let number = |path: [&str; 2]| -> Option<f64> {
        report.get(path[0])?.get(path[1])?.as_f64().filter(|v| v.is_finite())
    };

    Ok(Some(RoundEvidence {
        score,
        score_stderr: number(["breakdown", "score_stderr"]).unwrap_or(f64::NAN),
    }))
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
