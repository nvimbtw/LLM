//! A message to a Discord channel when a long run lands.
//!
//! The queue exists so work can be started and walked away from. Everything that
//! made that possible — durable rows, a queue that survives a restart, follow-ups
//! that queue themselves — still requires opening the board to find out what
//! happened. This closes that loop: the outcome arrives where the person already is.
//!
//! **Two rules keep the channel worth reading.**
//!
//! It fires only for foreground `train` and `benchmark` jobs. Every chat turn is a
//! job (see `JobKind::Chat`), and a ten-round post-training cycle is fifty of them
//! (`background = true`); notifying on either would produce a feed nobody reads,
//! which is the same as no feed at all. `should_notify` is the one statement of
//! that rule.
//!
//! And it never fails a job. The POST is spawned and detached, every error is
//! logged and swallowed, and an unset webhook simply means the feature is off.
//! A training run that survived four hours of CUDA must not be marked failed
//! because Discord returned a 500.

use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{Value, json};
use sqlx::PgPool;

use db::job::{Job, JobKind, JobStatus};

use crate::chain::Resolution;

/// The environment variable holding the webhook URL. The URL is a credential —
/// anyone with it can post to the channel — so it lives in `backend/.env` beside
/// `DATABASE_URL` and is never written into source or into a log line.
const WEBHOOK_ENV: &str = "DISCORD_WEBHOOK_URL";

/// How long to wait on Discord before giving up. Short on purpose: this is a
/// courtesy message about work that has already finished, and nothing downstream
/// waits for it.
const TIMEOUT: Duration = Duration::from_secs(10);

// Discord's embed colours, as the integers the API wants.
const COLOUR_DONE: u32 = 0x3B_A5_5D;
const COLOUR_FAILED: u32 = 0xED_42_45;
const COLOUR_CANCELLED: u32 = 0x99_AA_B5;

/// The configured webhook, read from the environment once.
///
/// `None` means notifications are off, which is a supported configuration rather
/// than an error: the queue works exactly as it did before, and the one line logged
/// at startup says so.
fn webhook() -> Option<&'static str> {
    static WEBHOOK: OnceLock<Option<String>> = OnceLock::new();
    WEBHOOK
        .get_or_init(|| {
            std::env::var(WEBHOOK_ENV)
                .ok()
                .map(|url| url.trim().to_string())
                .filter(|url| !url.is_empty())
        })
        .as_deref()
}

/// Log once, at startup, whether the channel is wired up. Called from `jobs::start`
/// so the answer is visible in the boot log rather than discovered by waiting eight
/// hours for a message that was never going to come.
pub fn report_configuration() {
    match webhook() {
        // Deliberately not the URL: it is a credential, and a boot log is the
        // easiest place in the system to leak one from.
        Some(_) => println!("discord notifications are on"),
        None => println!("discord notifications are off ({WEBHOOK_ENV} is not set)"),
    }
}

/// Whether a finished job is worth a message.
///
/// The whole filter, stated once. See the module comment for why it is this narrow.
fn should_notify(job: &Job) -> bool {
    matches!(job.kind, JobKind::Train | JobKind::Benchmark) && !job.background
}

/// Post the outcome of a finished job, and what it did to the chain behind it.
///
/// Returns immediately: the request runs on a detached task, because the caller is
/// the queue runner and the next job should not wait on an HTTP round trip to
/// somebody else's service.
pub fn notify_finished(pool: &PgPool, job: &Job, resolution: Resolution) {
    if webhook().is_none() || !should_notify(job) {
        return;
    }

    // Everything the message needs is copied out here, on the caller's side, so the
    // spawned task borrows nothing.
    let pool = pool.clone();
    let id_job = job.id_job;
    let kind = job.kind;
    let status = job.status;
    let label = job.label.clone();
    let error = job.error.clone();
    let ref_id = job.ref_id;
    let duration = duration_text(job);

    tokio::spawn(async move {
        // The benchmark's score lives in its own table, so the message reads it
        // back rather than duplicating it onto the job row. One query, on the async
        // side, after the job has already finished.
        let detail = match (kind, status, ref_id) {
            (JobKind::Benchmark, JobStatus::Done, Some(id_run)) => {
                benchmark_detail(&pool, id_run).await
            }
            _ => None,
        };

        let payload = embed(
            &label,
            kind,
            status,
            duration.as_deref(),
            error.as_deref(),
            detail.as_deref(),
            resolution,
        );

        if let Err(e) = post(&payload).await {
            // Logged and dropped. The job is already closed out correctly; a
            // missing notification is a missing convenience, not a failure.
            eprintln!("[discord] could not post the result of job {}: {}", id_job, e);
        }
    });
}

/// One message when a restart closed out jobs that were mid-flight, rather than one
/// per row: the interesting fact is that the machine went down, and it is the same
/// fact however many runs it took with it.
pub fn notify_interrupted(count: u64) {
    if webhook().is_none() || count == 0 {
        return;
    }
    let payload = json!({
        "embeds": [{
            "title": "⚠️ Interrupted by a restart",
            "description": format!(
                "{count} job{} {} running when the server went down and {} been closed out.",
                if count == 1 { "" } else { "s" },
                if count == 1 { "was" } else { "were" },
                if count == 1 { "has" } else { "have" },
            ),
            "color": COLOUR_FAILED,
        }],
    });
    tokio::spawn(async move {
        if let Err(e) = post(&payload).await {
            eprintln!("[discord] could not post the restart summary: {}", e);
        }
    });
}

/// Builds the embed. Pure, so the shape of a message is testable without a network
/// or a database.
fn embed(
    label: &str,
    kind: JobKind,
    status: JobStatus,
    duration: Option<&str>,
    error: Option<&str>,
    detail: Option<&str>,
    resolution: Resolution,
) -> Value {
    let (glyph, colour, verb) = match status {
        JobStatus::Done => ("✅", COLOUR_DONE, "finished"),
        JobStatus::Failed => ("❌", COLOUR_FAILED, "failed"),
        _ => ("⚪", COLOUR_CANCELLED, "was cancelled"),
    };

    let mut fields = vec![json!({
        "name": "Job",
        "value": match kind {
            JobKind::Train => "Training run",
            JobKind::Benchmark => "Benchmark",
            _ => "Job",
        },
        "inline": true,
    })];

    if let Some(duration) = duration {
        fields.push(json!({ "name": "Took", "value": duration, "inline": true }));
    }
    if let Some(detail) = detail {
        fields.push(json!({ "name": "Result", "value": detail, "inline": true }));
    }
    if let Some(error) = error {
        // The first line only. A CUDA failure's message can run to hundreds of
        // characters and Discord truncates a field at 1024 anyway; the first line
        // is the part that says what went wrong.
        let first = error.lines().next().unwrap_or(error);
        fields.push(json!({
            "name": "Why",
            "value": truncate(first, 500),
            "inline": false,
        }));
    }
    // The line that makes an unattended chain legible: whether the thing queued
    // behind this one actually started.
    if resolution.touched() {
        let mut parts = Vec::new();
        if resolution.promoted > 0 {
            parts.push(format!("queued {} follow-up job(s)", resolution.promoted));
        }
        if resolution.cancelled > 0 {
            parts.push(format!("cancelled {} follow-up job(s)", resolution.cancelled));
        }
        fields.push(json!({
            "name": "Chain",
            "value": parts.join(", "),
            "inline": false,
        }));
    }

    json!({
        "embeds": [{
            "title": format!("{glyph} {label}"),
            "description": format!("The {} {verb}.", match kind {
                JobKind::Train => "training run",
                JobKind::Benchmark => "benchmark",
                _ => "job",
            }),
            "color": colour,
            "fields": fields,
        }],
    })
}

/// The suite and score of a finished benchmark, for the message's Result field.
async fn benchmark_detail(pool: &PgPool, id_run: uuid::Uuid) -> Option<String> {
    let run = db::benchmark::find_by_id(pool, id_run).await.ok()??;
    let suite = run.suite.as_str();
    match run.score {
        // Two decimals: the suites report a standard error around ±1 point, so more
        // digits would be noise dressed as precision.
        Some(score) => Some(format!("{suite} — {score:.2}/100")),
        None => Some(suite.to_string()),
    }
}

/// Wall clock from claim to close, as something readable at a glance.
fn duration_text(job: &Job) -> Option<String> {
    let started = job.started_at?;
    let finished = job.finished_at?;
    let seconds = (finished - started).num_seconds().max(0);
    Some(match seconds {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m {}s", s / 60, s % 60),
        s => format!("{}h {}m", s / 3600, (s % 3600) / 60),
    })
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let kept: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{kept}…")
}

/// The one outbound request. A non-2xx is an error like any other — Discord's body
/// says what it objected to, and it is worth having in the log.
async fn post(payload: &Value) -> Result<(), String> {
    let url = webhook().ok_or("no webhook configured")?;
    let client = reqwest::Client::builder()
        .timeout(TIMEOUT)
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .post(url)
        .json(payload)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let status = response.status();
    if status.is_success() {
        return Ok(());
    }
    let body = response.text().await.unwrap_or_default();
    Err(format!("{status}: {}", truncate(body.trim(), 200)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use uuid::Uuid;

    fn job(kind: JobKind, background: bool) -> Job {
        Job {
            id_job: Uuid::new_v4(),
            kind,
            lane: kind.lane(),
            background,
            status: JobStatus::Done,
            label: "Balanced 2".into(),
            queue_rank: 1.0,
            config: json!({}),
            progress: 1.0,
            stage: None,
            ref_id: None,
            result: None,
            depends_on: None,
            error: None,
            created_at: Utc.timestamp_opt(0, 0).unwrap(),
            started_at: None,
            finished_at: None,
        }
    }

    #[test]
    fn only_foreground_training_and_benchmarks_are_worth_a_message() {
        assert!(should_notify(&job(JobKind::Train, false)));
        assert!(should_notify(&job(JobKind::Benchmark, false)));

        // Every assistant turn is a job. Notifying on those is the failure mode
        // this filter exists to prevent.
        assert!(!should_notify(&job(JobKind::Chat, false)));
        // As is a cycle round, which is ~50 jobs of its own.
        assert!(!should_notify(&job(JobKind::Train, true)));
        assert!(!should_notify(&job(JobKind::Autolabel, false)));
        assert!(!should_notify(&job(JobKind::Synth, false)));
    }

    #[test]
    fn a_duration_reads_in_the_largest_unit_that_fits() {
        let mut j = job(JobKind::Train, false);
        assert_eq!(duration_text(&j), None);

        j.started_at = Some(Utc.timestamp_opt(0, 0).unwrap());
        j.finished_at = Some(Utc.timestamp_opt(42, 0).unwrap());
        assert_eq!(duration_text(&j).as_deref(), Some("42s"));

        j.finished_at = Some(Utc.timestamp_opt(125, 0).unwrap());
        assert_eq!(duration_text(&j).as_deref(), Some("2m 5s"));

        j.finished_at = Some(Utc.timestamp_opt(3 * 3600 + 25 * 60, 0).unwrap());
        assert_eq!(duration_text(&j).as_deref(), Some("3h 25m"));
    }

    #[test]
    fn a_failure_carries_its_first_line_and_a_chain_carries_its_count() {
        let payload = embed(
            "Balanced 2",
            JobKind::Train,
            JobStatus::Failed,
            Some("1h 2m"),
            Some("out of memory on the device\n  at llm::train::step\n  at …"),
            None,
            Resolution { promoted: 0, cancelled: 2 },
        );
        let fields = payload["embeds"][0]["fields"].as_array().unwrap();
        let why = fields.iter().find(|f| f["name"] == "Why").unwrap();
        assert_eq!(why["value"], json!("out of memory on the device"));
        let chain = fields.iter().find(|f| f["name"] == "Chain").unwrap();
        assert_eq!(chain["value"], json!("cancelled 2 follow-up job(s)"));
        assert_eq!(payload["embeds"][0]["color"], json!(COLOUR_FAILED));
    }

    #[test]
    fn a_clean_run_with_nothing_behind_it_carries_neither_field() {
        let payload = embed(
            "Balanced 2",
            JobKind::Benchmark,
            JobStatus::Done,
            Some("48s"),
            None,
            Some("assistant — 53.94/100"),
            Resolution::default(),
        );
        let fields = payload["embeds"][0]["fields"].as_array().unwrap();
        assert!(fields.iter().all(|f| f["name"] != "Why"));
        assert!(fields.iter().all(|f| f["name"] != "Chain"));
        let result = fields.iter().find(|f| f["name"] == "Result").unwrap();
        assert_eq!(result["value"], json!("assistant — 53.94/100"));
    }
}
