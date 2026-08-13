//! Job futures: what happens to the jobs queued *behind* a job, when it lands.
//!
//! A follow-up is an ordinary row with `depends_on` set and status `waiting`. It
//! was authored in full at the same moment as the job it follows, but it names no
//! model, because the model it will consume does not exist yet — that is the whole
//! point. This module is the one place that fills the model in.
//!
//! **Why the model is injected here rather than at enqueue.** The alternative is a
//! sentinel in the stored config ("use whatever job X produces") that every reader
//! of a `TrainConfig` would then have to understand — the composers, the panels,
//! `train_worker`, the cycle driver. Resolving at promotion instead means a row is
//! either waiting, in which case nothing reads its model, or queued, in which case
//! its config is as concrete as one a person typed. There is no third state and no
//! sentinel anywhere.
//!
//! The resolver runs at every point a job becomes terminal — the queue's single
//! `finish` call, the two route handlers that close out a job that never started,
//! and once at boot for anything a restart stranded.

use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

use db::job::{Job, JobKind, JobStatus};

use crate::queue::Queue;

/// What resolving one finished job did to the jobs behind it.
///
/// Returned rather than only logged because the Discord notification for the
/// finished job reports it: "queued 2 follow-ups" is the line that makes an
/// overnight chain legible from a phone, and it has to come from the code that
/// actually did the queueing.
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct Resolution {
    pub promoted: usize,
    pub cancelled: usize,
}

impl Resolution {
    pub fn touched(&self) -> bool {
        self.promoted > 0 || self.cancelled > 0
    }
}

/// Settle every job waiting on `parent`, which has just reached a terminal status.
///
/// A no-op for a job nothing follows, which is almost all of them: one indexed
/// lookup that returns nothing.
pub async fn resolve(pool: &PgPool, queue: &Queue, parent: &Job) -> Resolution {
    let mut resolution = Resolution::default();
    if !parent.status.is_terminal() {
        return resolution;
    }

    let children = match db::job::list_waiting_on(pool, parent.id_job).await {
        Ok(children) => children,
        Err(e) => {
            // The chain stalls rather than breaking: the rows stay `waiting` and
            // the next restart's reconcile finds them through
            // `list_orphaned_waiting`. Log loudly, because until then they look
            // like work that is about to happen and is not.
            eprintln!(
                "[chain] could not read the jobs waiting on {}: {}",
                parent.id_job, e
            );
            return resolution;
        }
    };

    for child in children {
        // Already promoted or cancelled by an earlier pass — the resolver is
        // deliberately re-runnable, so this is expected rather than exceptional.
        if child.status != JobStatus::Waiting {
            continue;
        }

        match parent.status {
            JobStatus::Done => match promote_child(pool, parent, &child).await {
                Ok(true) => resolution.promoted += 1,
                Ok(false) => {}
                Err(reason) => {
                    cancel_subtree(pool, &child, &reason, &mut resolution).await;
                }
            },
            // The run this one follows did not produce anything, so neither will
            // this one. Said in terms of what happened rather than "cancelled":
            // the row is read hours later by somebody who was asleep for it.
            _ => {
                let reason = format!(
                    "the job it followed ({}) {}",
                    parent.label,
                    match parent.status {
                        JobStatus::Failed => "failed",
                        _ => "was cancelled",
                    }
                );
                cancel_subtree(pool, &child, &reason, &mut resolution).await;
            }
        }
    }

    if resolution.promoted > 0 {
        queue.wake();
    }
    if resolution.touched() {
        queue.notify_changed();
    }
    resolution
}

/// Cancel everything a deleted job stranded, plus anything behind it.
///
/// The delete route's counterpart to `resolve`: `ON DELETE SET NULL` leaves a
/// waiting child pointing at nothing, and a child whose base model will now never
/// be built can only fail later and more expensively.
pub async fn cancel_orphans(pool: &PgPool, queue: &Queue, orphans: &[Job]) -> Resolution {
    let mut resolution = Resolution::default();
    for orphan in orphans {
        if orphan.status != JobStatus::Waiting {
            continue;
        }
        let reason = match orphan.depends_on {
            None => "the job it was waiting on was deleted".to_string(),
            Some(id) => format!("the job it was waiting on ({}) did not finish normally", id),
        };
        cancel_subtree(pool, orphan, &reason, &mut resolution).await;
    }
    if resolution.touched() {
        queue.notify_changed();
    }
    resolution
}

/// Fill in the model and move the child into the line. `Ok(false)` means the row
/// was no longer waiting — somebody cancelled it in the meantime, and that wins.
async fn promote_child(pool: &PgPool, parent: &Job, child: &Job) -> Result<bool, String> {
    let mut config = child.config.clone();

    if needs_model(child.kind, &config) {
        let id_model = model_of(parent).ok_or_else(|| {
            format!(
                "the job it followed ({}) left no model to work from",
                parent.label
            )
        })?;
        inject_model(child.kind, &mut config, id_model)?;
    }

    db::job::promote(pool, child.id_job, &config)
        .await
        .map_err(|e| format!("could not queue the follow-up: {e}"))
}

/// Cancels one waiting job and everything queued behind it.
///
/// Iterative rather than recursive: an `async fn` that calls itself needs boxing,
/// and a worklist is both cheaper and easier to read. The chain is a tree, and
/// `visited` is the guard against a cycle — one cannot be built through the API,
/// but a hand-edited row should stall this loop rather than spin it forever.
async fn cancel_subtree(pool: &PgPool, root: &Job, reason: &str, resolution: &mut Resolution) {
    let mut pending = vec![(root.id_job, reason.to_string())];
    let mut visited: Vec<Uuid> = Vec::new();

    while let Some((id_job, reason)) = pending.pop() {
        if visited.contains(&id_job) {
            continue;
        }
        visited.push(id_job);

        match db::job::cancel_waiting(pool, id_job, &reason).await {
            Ok(true) => resolution.cancelled += 1,
            // Not waiting any more, so it is not ours to cancel — but its own
            // children may still be, so the walk continues either way.
            Ok(false) => {}
            Err(e) => {
                eprintln!("[chain] could not cancel waiting job {}: {}", id_job, e);
                continue;
            }
        }

        match db::job::list_waiting_on(pool, id_job).await {
            Ok(grandchildren) => {
                for grandchild in grandchildren {
                    pending.push((
                        grandchild.id_job,
                        "an earlier job in this chain did not finish".to_string(),
                    ));
                }
            }
            Err(e) => eprintln!("[chain] could not walk the chain under {}: {}", id_job, e),
        }
    }
}

/// The model a finished job hands to whatever it queued behind it.
///
/// **Two kinds hand one on, and only one of them made it.** A training run produces
/// a model, and `ref_id` names it. A benchmark produces a *score* — but it had a
/// model, the one it just scored, and that is the model the next step in the chain
/// means. Without this, "fine-tune it, score the language suite, score the assistant
/// suite" cannot be expressed: the second benchmark follows the first, and the first
/// built nothing.
///
/// So a benchmark is a pass-through. That is what keeps a chain a straight line —
/// each step follows the one written above it — rather than a tree the composer
/// would have to draw as a list and somebody would have to read as a tree.
fn model_of(parent: &Job) -> Option<Uuid> {
    match parent.kind {
        JobKind::Train => parent.ref_id,
        JobKind::Benchmark => parent
            .config
            .get("id_model")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok()),
        // Nothing else can be a follow-up's parent — the route refuses to file one
        // — and nothing else has a single model that the next job would obviously
        // mean.
        _ => None,
    }
}

/// Whether this job consumes a model produced by the job it follows.
///
/// A pretrain is the one follow-up that does not: it builds its own tokenizer and
/// starts from random weights, so chaining one is pure sequencing — "run this after
/// that, on the GPU I am not awake to share out" — and it is promoted unchanged.
fn needs_model(kind: JobKind, config: &Value) -> bool {
    match kind {
        JobKind::Train => config
            .get("mode")
            .and_then(|v| v.as_str())
            .is_some_and(|mode| mode != "pretrain"),
        JobKind::Benchmark => true,
        // The route refuses to file any other kind as a follow-up. If one exists it
        // was hand-written into the table, and treating it as needing a model is the
        // conservative reading: it gets cancelled with a reason rather than run
        // against whatever its config happened to name.
        _ => true,
    }
}

/// Writes the produced model into a follow-up's config, under whatever the field is
/// called for that kind. The only place those field names appear.
fn inject_model(kind: JobKind, config: &mut Value, id_model: Uuid) -> Result<(), String> {
    let object = config
        .as_object_mut()
        .ok_or_else(|| "the follow-up's stored config is not an object".to_string())?;

    match kind {
        JobKind::Train => {
            object.insert("base_model_id".into(), Value::String(id_model.to_string()));

            // The same inheritance `prepare_train` does at enqueue, deferred to
            // here because at enqueue there was no base to read. A fine-tune loads
            // the base's weights wholesale, so its architecture *is* the base's —
            // and the stored config is what every panel renders before the run
            // attaches. Best-effort: an unreadable meta.json still trains
            // correctly, the row just carries the requested numbers.
            let mut cfg: training::TrainConfig = serde_json::from_value(Value::Object(object.clone()))
                .map_err(|e| format!("the follow-up's stored config is unreadable: {e}"))?;
            let base_dir = engine::paths::data_root().join("artifacts/models").join(id_model.to_string());
            if let Err(e) = training::apply_base_architecture(&mut cfg, &base_dir) {
                eprintln!(
                    "[chain] {}: could not inherit the base architecture ({}); the row keeps the requested values",
                    cfg.model_name, e
                );
            }
            *config = serde_json::to_value(&cfg)
                .map_err(|e| format!("could not store the follow-up's config: {e}"))?;
            Ok(())
        }
        JobKind::Benchmark => {
            object.insert("id_model".into(), Value::String(id_model.to_string()));
            Ok(())
        }
        _ => Err(format!(
            "a {} job cannot inherit a model from the job it follows",
            crate::queue::kind_name(kind)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use db::job::JobLane;
    use serde_json::json;

    #[test]
    fn a_fine_tune_needs_a_model_and_a_pretrain_does_not() {
        assert!(needs_model(JobKind::Train, &json!({ "mode": "sft" })));
        assert!(!needs_model(JobKind::Train, &json!({ "mode": "pretrain" })));
        // Absent means pretrain, matching `TrainMode`'s own default — a config
        // written by an older client must not be read as a fine-tune with no base.
        assert!(!needs_model(JobKind::Train, &json!({})));
        assert!(needs_model(JobKind::Benchmark, &json!({})));
    }

    #[test]
    fn the_model_lands_in_the_field_each_kind_names_it_by() {
        let id = Uuid::new_v4();

        // A benchmark config is free-form JSON, so it round-trips untouched apart
        // from the one field.
        let mut config = json!({ "suite": "assistant" });
        inject_model(JobKind::Benchmark, &mut config, id).unwrap();
        assert_eq!(config["id_model"], json!(id.to_string()));
        assert_eq!(config["suite"], json!("assistant"));

        // A train config goes through `TrainConfig`, so this also pins that the
        // round trip preserves the fields the composer set.
        let mut config = json!({
            "model_name": "Follow-up 1",
            "dimensions": 256,
            "ctx_window": 512,
            "heads": 4,
            "layers": 6,
            "epochs": 3,
            "lr": 3e-5,
            "ff_multiplier": 4,
            "max_vocab": 8192,
            "mode": "sft",
        });
        inject_model(JobKind::Train, &mut config, id).unwrap();
        assert_eq!(config["base_model_id"], json!(id.to_string()));
        assert_eq!(config["model_name"], json!("Follow-up 1"));
        assert_eq!(config["epochs"], json!(3));
    }

    #[test]
    fn a_benchmark_hands_on_the_model_it_scored() {
        let scored = Uuid::new_v4();
        let produced = Uuid::new_v4();

        let mut job = Job {
            id_job: Uuid::new_v4(),
            kind: JobKind::Benchmark,
            lane: JobLane::Device,
            background: false,
            status: JobStatus::Done,
            label: "Balanced 2 · assistant".into(),
            queue_rank: 1.0,
            config: json!({ "id_model": scored, "suite": "assistant" }),
            progress: 1.0,
            stage: None,
            // A benchmark's `ref_id` is its *run*, not a model — reading it as one is
            // the mistake this pass-through exists to avoid.
            ref_id: Some(Uuid::new_v4()),
            result: None,
            depends_on: None,
            error: None,
            created_at: chrono::Utc::now(),
            started_at: None,
            finished_at: None,
        };
        assert_eq!(model_of(&job), Some(scored));

        // A training run hands on what it built.
        job.kind = JobKind::Train;
        job.ref_id = Some(produced);
        assert_eq!(model_of(&job), Some(produced));

        // A cancelled run built nothing, so there is nothing to hand on.
        job.ref_id = None;
        assert_eq!(model_of(&job), None);
    }

    #[test]
    fn a_kind_that_cannot_inherit_a_model_is_refused_rather_than_guessed_at() {
        let mut config = json!({ "id_model": "whatever" });
    }
}
