// In-memory registry of training runs. A run is started via the REST route,
// executes on its own OS thread (the `llm` training is blocking/GPU work and
// must not run on the async executor), and publishes progress two ways:
//   - a `broadcast` channel for live frames to any attached socket, and
//   - a `JobState` snapshot so a socket that attaches (or re-attaches after a
//     reconnect) can catch up to the current state.
// The job outlives the socket — closing the tab does not stop training; only an
// explicit cancel does. State lives in memory; the durable record is the run's
// meta.json + checkpoints on disk.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use serde::Serialize;
use tokio::sync::{broadcast, mpsc};
use uuid::Uuid;

use crate::db::model::NewModel;
use crate::training::{self, DatasetStats, TokenizeEvent, TrainConfig};

const BROADCAST_CAPACITY: usize = 256;
const MAX_LOG_LINES: usize = 500;
const MAX_LOSS_POINTS: usize = 4000;

#[derive(Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Tokenizing,
    Training,
    Done,
    Cancelled,
    Failed,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobStatus::Done | JobStatus::Cancelled | JobStatus::Failed
        )
    }
}

// Catch-up snapshot for a late / re-attaching socket. The live frames carry the
// same data incrementally; this is the accumulated view.
pub struct JobState {
    pub status: JobStatus,
    pub config: TrainConfig,
    pub tokenize: serde_json::Value,
    pub train: serde_json::Value,
    pub loss_series: Vec<[f64; 2]>,
    pub log: Vec<String>,
    pub error: Option<String>,
}

pub struct Job {
    pub cancel: Arc<AtomicBool>,
    pub tx: broadcast::Sender<String>,
    pub state: Mutex<JobState>,
    // Cloned into each job so the worker thread can hand a finished model off to
    // the async db consumer (see `finish`).
    pub model_tx: mpsc::UnboundedSender<NewModel>,
}

pub struct JobRegistry {
    jobs: Mutex<HashMap<Uuid, Arc<Job>>>,
    // The producer end of the model-persistence channel. The consumer (which owns
    // the db pool) is spawned in main; sending is synchronous, so it bridges the
    // blocking worker thread to the async pool without a runtime handle here.
    model_tx: mpsc::UnboundedSender<NewModel>,
}

impl JobRegistry {
    pub fn new(model_tx: mpsc::UnboundedSender<NewModel>) -> Self {
        JobRegistry {
            jobs: Mutex::new(HashMap::new()),
            model_tx,
        }
    }

    pub fn get(&self, id: Uuid) -> Option<Arc<Job>> {
        self.jobs.lock().unwrap().get(&id).cloned()
    }

    // (id, status, model_name) for each known run, for the resume listing.
    pub fn list(&self) -> Vec<(Uuid, JobStatus, String)> {
        self.jobs
            .lock()
            .unwrap()
            .iter()
            .map(|(id, job)| {
                let st = job.state.lock().unwrap();
                (*id, st.status, st.config.model_name.clone())
            })
            .collect()
    }

    // Registers a new run and spawns its worker thread. Returns the job id the
    // client uses to attach a progress socket.
    pub fn start(&self, config: TrainConfig) -> Uuid {
        let id = Uuid::new_v4();
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let job = Arc::new(Job {
            cancel: Arc::new(AtomicBool::new(false)),
            tx,
            model_tx: self.model_tx.clone(),
            state: Mutex::new(JobState {
                status: JobStatus::Tokenizing,
                config: config.clone(),
                tokenize: serde_json::json!({}),
                train: serde_json::json!({}),
                loss_series: Vec::new(),
                log: Vec::new(),
                error: None,
            }),
        });
        self.jobs.lock().unwrap().insert(id, job.clone());

        // The crate's GPU training is FFI into CUDA; a CUDA fault or a failed
        // invariant surfaces as a panic. On a bare thread that would die
        // silently, leaving the job stuck "training" with no feedback — so we
        // catch it and turn it into a terminal error frame.
        let watched = job.clone();
        std::thread::spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_job(job, config)
            }));
            if let Err(payload) = outcome {
                let msg = panic_message(&*payload);
                eprintln!("[training] job crashed: {}", msg);
                let mut st = watched.state.lock().unwrap();
                if !st.status.is_terminal() {
                    st.status = JobStatus::Failed;
                    st.error = Some(format!("training crashed: {}", msg));
                    drop(st);
                    let _ = watched.tx.send(frame(
                        "error",
                        serde_json::json!({ "message": format!("training crashed: {}", msg) }),
                    ));
                }
            }
        });
        id
    }
}

// Pulls a human-readable message out of a panic payload (panics carry either a
// &str or a String).
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

// Renders the accumulated state for a socket that just attached.
pub fn snapshot_frame(state: &JobState) -> String {
    serde_json::json!({
        "type": "snapshot",
        "status": state.status,
        "config": state.config,
        "tokenize": state.tokenize,
        "train": state.train,
        "loss_series": state.loss_series,
        "log": state.log,
        "error": state.error,
    })
    .to_string()
}

// ── worker ──────────────────────────────────────────────────────────────────

// Runs both phases for one job on a dedicated thread. Both crate calls are
// synchronous and blocking, which is exactly why this is a plain OS thread and
// not a task on the actix runtime.
fn run_job(job: Arc<Job>, config: TrainConfig) {
    let created_at = Utc::now().to_rfc3339();

    // ---- Phase 1: tokenize / build the dataset ----
    eprintln!("[training] {}: tokenizing uploads…", config.model_name);
    let dataset = training::build_dataset(config.max_vocab, config.num_books, |event| {
        handle_tokenize_event(&job, event);
    });

    let dataset = match dataset {
        Ok(stats) => {
            let obj = serde_json::json!({
                "files": stats.files,
                "chars": stats.chars,
                "vocab": stats.vocab_size,
                "merges": stats.merges,
                "progress": 1.0,
                "done": true,
            });
            {
                let mut st = job.state.lock().unwrap();
                st.tokenize = obj.clone();
                st.status = JobStatus::Training;
            }
            let _ = job.tx.send(frame("tokenize", obj));
            stats
        }
        Err(e) => {
            finish_error(&job, &format!("tokenization failed: {}", e));
            return;
        }
    };

    // ---- Phase 2: train ----
    eprintln!(
        "[training] {}: starting GPU training ({}d, {}ctx, {}h, {}L, {}ep, {} tokens)",
        config.model_name,
        config.dimensions,
        config.ctx_window,
        config.heads,
        config.layers,
        config.epochs,
        dataset.corpus_len,
    );
    let job_for_cb = job.clone();
    let on_progress: llm::train::ProgressCallback =
        Box::new(move |p| handle_train_progress(&job_for_cb, p));

    let outcome = training::run_training(&config, on_progress, job.cancel.clone());
    eprintln!("[training] {}: training returned {:?}", config.model_name, outcome);
    match outcome {
        Ok(true) => finish(&job, JobStatus::Cancelled, &config, dataset, &created_at, None),
        Ok(false) => finish(&job, JobStatus::Done, &config, dataset, &created_at, None),
        Err(e) => finish(
            &job,
            JobStatus::Failed,
            &config,
            dataset,
            &created_at,
            Some(format!("training failed: {}", e)),
        ),
    }
}

fn handle_tokenize_event(job: &Job, event: TokenizeEvent) {
    match event {
        TokenizeEvent::Start { files, chars } => {
            let obj = serde_json::json!({
                "files": files,
                "chars": chars,
                "vocab": 256,
                "merges": 0,
                "progress": 0.0,
                "done": false,
            });
            job.state.lock().unwrap().tokenize = obj.clone();
            let _ = job.tx.send(frame("tokenize", obj));
        }
        TokenizeEvent::Merge { done, total } => {
            let progress = if total > 0 {
                done as f64 / total as f64
            } else {
                0.0
            };
            // Keep files/chars already recorded by the Start event.
            let obj = {
                let mut st = job.state.lock().unwrap();
                if let Some(map) = st.tokenize.as_object_mut() {
                    map.insert("vocab".into(), serde_json::json!(256 + done as i64));
                    map.insert("merges".into(), serde_json::json!(done));
                    map.insert("progress".into(), serde_json::json!(progress));
                }
                st.tokenize.clone()
            };
            let _ = job.tx.send(frame("tokenize", obj));
        }
        TokenizeEvent::Log(line) => {
            push_capped(&mut job.state.lock().unwrap().log, line.to_string());
            let _ = job.tx.send(frame("log", serde_json::json!({ "line": line })));
        }
    }
}

fn handle_train_progress(job: &Job, p: llm::train::TrainingProgress) {
    let step = p.epoch.saturating_sub(1) * p.total_chunks + p.chunk;
    let obj = serde_json::json!({
        "epoch": p.epoch,
        "total_epochs": p.total_epochs,
        "chunk": p.chunk,
        "total_chunks": p.total_chunks,
        "loss": p.loss,
        "avg_loss": p.avg_loss,
        "min_loss": p.min_loss,
        "max_loss": p.max_loss,
        "eta_secs": p.eta_secs,
        "epoch_done": p.epoch_done,
        "epoch_duration_secs": p.epoch_duration_secs,
        "checkpoint_path": p.checkpoint_path,
        "step": step,
    });
    {
        let mut st = job.state.lock().unwrap();
        st.train = obj.clone();
        if p.loss.is_finite() {
            push_loss(&mut st.loss_series, [step as f64, p.loss]);
        }
    }
    let _ = job.tx.send(frame("train", obj));
}

// Terminal completion (done / cancelled / failed) once the dataset is known.
fn finish(
    job: &Job,
    status: JobStatus,
    config: &TrainConfig,
    dataset: DatasetStats,
    created_at: &str,
    error: Option<String>,
) {
    let status_str = match status {
        JobStatus::Done => "done",
        JobStatus::Cancelled => "cancelled",
        JobStatus::Failed => "failed",
        _ => "unknown",
    };

    if let Err(e) = training::write_meta(config, dataset, status_str, created_at) {
        eprintln!("failed to write run meta for {}: {}", config.model_name, e);
    }

    // Register a successful run in the database so it appears in the model
    // registry and can be selected in chat. Cancelled / failed runs may leave
    // partial checkpoints on disk but are not registered. The consumer task
    // (spawned in main) owns the pool and performs the async insert; sending is
    // non-blocking and only fails at shutdown, so the error is ignored.
    if status == JobStatus::Done {
        let record = NewModel {
            id_model: config.storage_id,
            name: config.model_name.clone(),
            description: None,
            parameters: serde_json::json!({ "config": config, "dataset": dataset }),
            context: config.ctx_window as i32,
            family: Some("transformer".to_string()),
        };
        let _ = job.model_tx.send(record);
    }

    {
        let mut st = job.state.lock().unwrap();
        st.status = status;
        st.error = error.clone();
    }

    let payload = match error {
        Some(msg) => frame("error", serde_json::json!({ "message": msg })),
        None => {
            let model_dir = crate::storage::models_dir().join(config.storage_id.to_string());
            frame(
                "done",
                serde_json::json!({
                    "status": status_str,
                    "model_dir": model_dir.to_string_lossy(),
                }),
            )
        }
    };
    let _ = job.tx.send(payload);
}

// Terminal failure before a dataset exists (i.e. tokenization failed).
fn finish_error(job: &Job, msg: &str) {
    {
        let mut st = job.state.lock().unwrap();
        st.status = JobStatus::Failed;
        st.error = Some(msg.to_string());
    }
    let _ = job.tx.send(frame("error", serde_json::json!({ "message": msg })));
}

// ── small helpers ───────────────────────────────────────────────────────────

// Tags a JSON object with its frame `type` and serializes it.
fn frame(kind: &str, mut obj: serde_json::Value) -> String {
    if let Some(map) = obj.as_object_mut() {
        map.insert("type".into(), serde_json::Value::String(kind.to_string()));
    }
    obj.to_string()
}

fn push_capped(buf: &mut Vec<String>, item: String) {
    buf.push(item);
    if buf.len() > MAX_LOG_LINES {
        let overflow = buf.len() - MAX_LOG_LINES;
        buf.drain(0..overflow);
    }
}

// Appends a loss point, halving the series by dropping every other point once
// it grows past the cap so a re-attach snapshot stays small.
fn push_loss(series: &mut Vec<[f64; 2]>, point: [f64; 2]) {
    series.push(point);
    if series.len() > MAX_LOSS_POINTS {
        let mut keep = false;
        series.retain(|_| {
            keep = !keep;
            keep
        });
    }
}
