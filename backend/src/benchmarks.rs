// In-memory registry of benchmark runs, mirroring `jobs.rs` for training.
//
// The same constraint applies: `llm::eval::run` is blocking CUDA work and must not
// occupy the async executor, so a run gets its own OS thread and reports back
// through channels. It differs from training in two ways that let it be much
// smaller — a run is short (tens of seconds, not hours) and produces one result
// rather than a stream, so progress is polled rather than pushed over a socket,
// and there is no cancellation.
//
// Durability lives in Postgres: the row is inserted `running` before the thread
// starts and updated once at the end. This registry only holds the live progress
// of the run currently in flight.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use uuid::Uuid;

use crate::jobs::lock;
use crate::storage;

// How far along a run is. Polled by the frontend via `GET /benchmark/runs`, which
// merges this into the DB row of a `running` run.
#[derive(Clone, serde::Serialize)]
pub struct Progress {
    // Human-readable current step, e.g. "Grammar: filler_gap (325/600)".
    pub stage: String,
    // Overall completion in [0, 1].
    pub fraction: f64,
}

// What a finished worker thread hands to the async consumer that owns the pool.
pub enum Outcome {
    Done {
        id_run: Uuid,
        // NaN never reaches here — a report with no usable components stores NULL
        // rather than a number that would plot as a real zero.
        score: Option<f64>,
        report: serde_json::Value,
    },
    Failed {
        id_run: Uuid,
        error: String,
    },
}

pub struct BenchmarkRegistry {
    // Shared with each worker thread, which is why it's an Arc and not a plain
    // field: the registry itself sits behind `web::Data` and can't be borrowed
    // into a `'static` thread closure.
    active: Arc<Mutex<HashMap<Uuid, Progress>>>,
    tx: mpsc::UnboundedSender<Outcome>,
}

impl BenchmarkRegistry {
    pub fn new(tx: mpsc::UnboundedSender<Outcome>) -> BenchmarkRegistry {
        BenchmarkRegistry {
            active: Arc::new(Mutex::new(HashMap::new())),
            tx,
        }
    }

    // Is a run already in flight? The GPU fits one at a time, so the route rejects
    // a second start rather than letting two runs fight over the device and
    // surface as an opaque CUDA fault.
    pub fn is_busy(&self) -> bool {
        !lock(&self.active).is_empty()
    }

    pub fn progress(&self, id_run: Uuid) -> Option<Progress> {
        lock(&self.active).get(&id_run).cloned()
    }

    // Spawns the run. The DB row must already exist (the route inserts it before
    // calling this) so a failure always has somewhere to be recorded.
    pub fn start(&self, id_run: Uuid, model_dir: PathBuf) {
        lock(&self.active).insert(id_run, Progress {
            stage: "Loading model".to_string(),
            fraction: 0.0,
        });

        let tx = self.tx.clone();
        let active = self.active.clone();

        std::thread::spawn(move || {
            let config = llm::eval::EvalConfig::new(model_dir, &storage::eval_dir());

            let reporting = active.clone();
            let on_progress: llm::eval::ProgressCallback = Box::new(move |stage, fraction| {
                lock(&reporting).insert(id_run, Progress {
                    stage: stage.to_string(),
                    fraction,
                });
            });

            // `eval` is FFI into CUDA; a device fault or a failed invariant arrives
            // as a panic, which on a bare thread would die silently and leave the
            // run showing "running" forever. Same reasoning as `jobs::start`.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                llm::eval::run(&config, on_progress)
            }));

            let outcome = match result {
                Ok(Ok(report)) => build_outcome(id_run, &report),
                Ok(Err(e)) => Outcome::Failed { id_run, error: e },
                Err(payload) => Outcome::Failed {
                    id_run,
                    error: format!("benchmark crashed: {}", panic_message(&*payload)),
                },
            };

            // Cleared before the send so a client polling right after the row goes
            // terminal can't still see stale progress for it.
            lock(&active).remove(&id_run);
            // Sending only fails at shutdown, when the consumer is already gone.
            let _ = tx.send(outcome);
        });
    }
}

// Serializes a finished report into the outcome the DB consumer persists.
fn build_outcome(id_run: Uuid, report: &llm::eval::Report) -> Outcome {
    match serde_json::to_value(report) {
        // The composite score is NaN when nothing could be measured, and
        // `to_value` renders that as null — so reading the score back out of the
        // serialized form yields exactly the same verdict the report shows.
        Ok(value) => Outcome::Done {
            id_run,
            score: value.get("score").and_then(|s| s.as_f64()),
            report: value,
        },
        Err(e) => Outcome::Failed {
            id_run,
            error: format!("could not serialize report: {e}"),
        },
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}
