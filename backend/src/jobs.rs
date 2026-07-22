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
use crate::training::{self, TokenizeEvent, TrainConfig, TrainMode};

const BROADCAST_CAPACITY: usize = 256;
const MAX_LOG_LINES: usize = 500;
const MAX_LOSS_POINTS: usize = 4000;
// The batch-spread chart is a short rolling window of the most recent reports.
const MAX_BATCH_POINTS: usize = 60;
// The diagnostics charts (grad norms, update ratios, activations, attention,
// embeddings) show the whole run trajectory, so — like the loss series — this is
// halved when it fills rather than windowed, preserving the curve shape.
const MAX_DIAG_POINTS: usize = 1000;

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
    // Rolling window of recent per-report spread: [step, min, avg, max].
    pub batch_series: Vec<[f64; 4]>,
    // Per-report training diagnostics, each a `{ step, grad_*, update_ratio_layers,
    // act_rms_layers, attn_entropy, emb_cos_* }` object (see `diag_frame`). Kept as
    // pre-rendered JSON so the snapshot replays it verbatim.
    pub diag_series: Vec<serde_json::Value>,
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
        lock(&self.jobs).get(&id).cloned()
    }

    // (id, status, model_name) for each known run, for the resume listing.
    pub fn list(&self) -> Vec<(Uuid, JobStatus, String)> {
        lock(&self.jobs)
            .iter()
            .map(|(id, job)| {
                let st = lock(&job.state);
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
                batch_series: Vec::new(),
                diag_series: Vec::new(),
                log: Vec::new(),
                error: None,
            }),
        });
        lock(&self.jobs).insert(id, job.clone());

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
                let mut st = lock(&watched.state);
                if !st.status.is_terminal() {
                    st.status = JobStatus::Failed;
                    st.error = Some(format!("training crashed: {}", msg));
                    drop(st);
                    emit(&watched.tx, frame(
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
    // Only a successful run registered a model, so a reattached, already-`done`
    // run can still offer "Continue to SFT". Absent (null) for in-flight,
    // cancelled, or failed runs. Mirrors the live `done` frame's identifiers.
    let (id_model, model_name) = if state.status == JobStatus::Done {
        (
            serde_json::json!(state.config.storage_id),
            serde_json::json!(state.config.model_name),
        )
    } else {
        (serde_json::Value::Null, serde_json::Value::Null)
    };
    serde_json::json!({
        "type": "snapshot",
        "status": state.status,
        "config": state.config,
        "tokenize": state.tokenize,
        "train": state.train,
        "loss_series": state.loss_series,
        "batch_series": state.batch_series,
        "diag_series": state.diag_series,
        "log": state.log,
        "error": state.error,
        "id_model": id_model,
        "model_name": model_name,
    })
    .to_string()
}

// ── worker ──────────────────────────────────────────────────────────────────

// Dispatches a job to its mode's worker. Both are two-phase (build dataset, then
// train) and share the progress/finish plumbing; they differ only in phase 1
// (BPE tokenize vs. SFT render) and which train entry point phase 2 drives.
fn run_job(job: Arc<Job>, config: TrainConfig) {
    match config.mode {
        TrainMode::Pretrain => run_pretrain_job(job, config),
        TrainMode::Sft => run_sft_job(job, config),
    }
}

// Runs both phases for a pretrain job on a dedicated thread. Both crate calls are
// synchronous and blocking, which is exactly why this is a plain OS thread and
// not a task on the actix runtime.
fn run_pretrain_job(job: Arc<Job>, config: TrainConfig) {
    let created_at = Utc::now().to_rfc3339();

    // ---- Phase 1: tokenize / build the dataset ----
    eprintln!("[training] {}: tokenizing uploads…", config.model_name);
    let dataset =
        training::build_dataset(config.max_vocab, &config.datasets, |event| {
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
                let mut st = lock(&job.state);
                st.tokenize = obj.clone();
                st.status = JobStatus::Training;
            }
            emit(&job.tx, frame("tokenize", obj));
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
    let dataset = serde_json::json!(dataset);
    finish_outcome(&job, &config, &dataset, &created_at, outcome);
}

// Runs both phases for an SFT job: render the chat conversations into the masked
// dataset using the base model's tokenizer, then fine-tune the base via
// `run_sft_training`. Shares the progress/finish plumbing with the pretrain path.
fn run_sft_job(job: Arc<Job>, mut config: TrainConfig) {
    let created_at = Utc::now().to_rfc3339();

    // The base model to fine-tune from; its tokenizer renders the dataset and its
    // weights are the starting point. The route guarantees this is present and
    // resolves, but we resolve defensively (the id is untrusted here).
    let base_model_dir = match config.base_model_id {
        Some(id) => crate::storage::models_dir().join(id.to_string()),
        None => {
            finish_error(&job, "SFT run is missing a base_model_id");
            return;
        }
    };

    // The SFT fine-tune inherits the base's architecture (it loads the base
    // weights), so overwrite the request's placeholder arch fields with the base's
    // real ones — otherwise the registered model records the wrong dimensions/
    // context/param-count. Best-effort: a failure here only affects metadata.
    if let Err(e) = training::apply_base_architecture(&mut config, &base_model_dir) {
        eprintln!(
            "[training] {}: could not inherit base architecture ({}); registry metadata may be off",
            config.model_name, e
        );
    }

    // ---- Phase 1: render conversations → sft_corpus.bin + sft_mask.bin ----
    eprintln!("[training] {}: building SFT dataset…", config.model_name);
    let dataset = match training::build_sft_dataset(&base_model_dir) {
        Ok(stats) => {
            let obj = serde_json::json!({
                "conversations": stats.conversations,
                "total_tokens": stats.total_tokens,
                "trainable_tokens": stats.trainable_tokens,
                "progress": 1.0,
                "done": true,
            });
            {
                let mut st = lock(&job.state);
                st.tokenize = obj.clone();
                st.status = JobStatus::Training;
            }
            emit(&job.tx, frame("tokenize", obj));
            serde_json::json!(stats)
        }
        Err(e) => {
            finish_error(&job, &format!("SFT dataset build failed: {}", e));
            return;
        }
    };

    // ---- Phase 2: fine-tune from the base ----
    eprintln!(
        "[training] {}: starting SFT fine-tune ({} epochs, base {})",
        config.model_name, config.epochs, base_model_dir.display(),
    );
    let job_for_cb = job.clone();
    let on_progress: llm::train::ProgressCallback =
        Box::new(move |p| handle_train_progress(&job_for_cb, p));

    let outcome = training::run_sft_training(&config, on_progress, job.cancel.clone());
    eprintln!("[training] {}: SFT returned {:?}", config.model_name, outcome);
    finish_outcome(&job, &config, &dataset, &created_at, outcome);
}

// Maps a training outcome (cancelled / completed / errored) onto the terminal
// `finish`. Shared by both modes so the status mapping lives in one place.
fn finish_outcome(
    job: &Job,
    config: &TrainConfig,
    dataset: &serde_json::Value,
    created_at: &str,
    outcome: std::io::Result<bool>,
) {
    match outcome {
        Ok(true) => finish(job, JobStatus::Cancelled, config, dataset, created_at, None),
        Ok(false) => finish(job, JobStatus::Done, config, dataset, created_at, None),
        Err(e) => finish(
            job,
            JobStatus::Failed,
            config,
            dataset,
            created_at,
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
            lock(&job.state).tokenize = obj.clone();
            emit(&job.tx, frame("tokenize", obj));
        }
        TokenizeEvent::Merge { done, total } => {
            let progress = if total > 0 {
                done as f64 / total as f64
            } else {
                0.0
            };
            // Keep files/chars already recorded by the Start event.
            let obj = {
                let mut st = lock(&job.state);
                if let Some(map) = st.tokenize.as_object_mut() {
                    map.insert("vocab".into(), serde_json::json!(tokenize::bpe::vocab_size(done as usize)));
                    map.insert("merges".into(), serde_json::json!(done));
                    map.insert("progress".into(), serde_json::json!(progress));
                }
                st.tokenize.clone()
            };
            emit(&job.tx, frame("tokenize", obj));
        }
        TokenizeEvent::Log(line) => {
            push_capped(&mut lock(&job.state).log, line.to_string());
            emit(&job.tx, frame("log", serde_json::json!({ "line": line })));
        }
    }
}

// Renders a diagnostics snapshot into the `{ step, … }` JSON the frontend charts
// consume. Non-finite metrics (e.g. an empty embedding metric on a tiny vocab)
// serialize to null via `json!`, matching the batch-spread convention, so the
// client simply skips those points. Kept separate from the `train` frame builder
// so the (larger, per-report-only) diagnostics payload lives in one place.
fn diag_frame(step: u64, d: &llm::training::diagnostics::Diagnostics) -> serde_json::Value {
    serde_json::json!({
        "step": step,
        "grad_global": d.grad_global,
        "grad_layers": d.grad_layers,
        "grad_embedding": d.grad_embedding,
        "grad_unembed": d.grad_unembed,
        "update_ratio_layers": d.update_ratio_layers,
        "act_rms_layers": d.act_rms_layers,
        "attn_entropy": d.attn_entropy,
        "emb_cos_mean": d.emb_cos_mean,
        "emb_cos_p5": d.emb_cos_p5,
        "emb_cos_p95": d.emb_cos_p95,
        "act_hist": d.act_hist,
        "emb_pca": d.emb_pca,
    })
}

fn handle_train_progress(job: &Job, p: llm::train::TrainingProgress) {
    let step = p.epoch.saturating_sub(1) * p.total_chunks + p.chunk;
    // Render the diagnostics (present only on mid-epoch report frames) before the
    // main object so it can be embedded in the live frame and stored for reattach.
    let diag = p.diagnostics.as_ref().map(|d| diag_frame(step, d));
    let obj = serde_json::json!({
        "epoch": p.epoch,
        "total_epochs": p.total_epochs,
        "chunk": p.chunk,
        "total_chunks": p.total_chunks,
        "loss": p.loss,
        "avg_loss": p.avg_loss,
        "min_loss": p.min_loss,
        "max_loss": p.max_loss,
        "batch_min": p.batch_min,
        "batch_avg": p.batch_avg,
        "batch_max": p.batch_max,
        "eta_secs": p.eta_secs,
        "epoch_done": p.epoch_done,
        "epoch_duration_secs": p.epoch_duration_secs,
        "checkpoint_path": p.checkpoint_path,
        "step": step,
        // Present only on mid-epoch report frames; null otherwise so the client
        // knows there's no fresh diagnostics sample on this frame.
        "diag": diag,
    });
    {
        let mut st = lock(&job.state);
        st.train = obj.clone();
        if p.loss.is_finite() {
            push_loss(&mut st.loss_series, [step as f64, p.loss]);
        }
        if !p.epoch_done && p.batch_avg.is_finite() {
            push_batch(
                &mut st.batch_series,
                [
                    step as f64,
                    p.batch_min as f64,
                    p.batch_avg as f64,
                    p.batch_max as f64,
                ],
            );
        }
        if let Some(d) = diag {
            push_diag(&mut st.diag_series, d);
        }
    }
    emit(&job.tx, frame("train", obj));
}

// Terminal completion (done / cancelled / failed) once the dataset is known.
fn finish(
    job: &Job,
    status: JobStatus,
    config: &TrainConfig,
    // Already-serialized per-mode dataset stats (DatasetStats or SftDatasetStats).
    dataset: &serde_json::Value,
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
        let mut st = lock(&job.state);
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
                    // The just-registered model's identifiers, so the client can
                    // seed a follow-up SFT run with this model as its base without
                    // hunting for it in the model list (the "Continue to SFT" flow).
                    "id_model": config.storage_id,
                    "model_name": config.model_name,
                }),
            )
        }
    };
    emit(&job.tx, payload);
}

// Terminal failure before a dataset exists (i.e. tokenization failed).
fn finish_error(job: &Job, msg: &str) {
    {
        let mut st = lock(&job.state);
        st.status = JobStatus::Failed;
        st.error = Some(msg.to_string());
    }
    emit(&job.tx, frame("error", serde_json::json!({ "message": msg })));
}

// ── small helpers ───────────────────────────────────────────────────────────

// Locks a job mutex, recovering the guard if a previous holder panicked while
// holding it. The guarded sections here are single field reads/writes that don't
// themselves panic, so poisoning isn't expected — but a long-running server
// shouldn't turn one unlucky panic into a cascade of `.unwrap()` panics across
// every socket handler, so we take the last state and carry on. Centralizes the
// justification that would otherwise repeat at each `.lock()` (R3).
pub(crate) fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// Best-effort broadcast of a live frame. `send` errors only when no socket is
// currently attached (no receivers); that's normal — the job runs regardless of
// who is listening, and a re-attaching socket catches up from the JobState
// snapshot. Dropping the result is intentional (R3).
fn emit(tx: &broadcast::Sender<String>, msg: String) {
    let _ = tx.send(msg);
}

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

// Appends a batch-spread point, keeping only the most recent MAX_BATCH_POINTS.
// Unlike push_loss we drop from the front rather than halving — this is a small
// rolling window where every-other-point thinning would distort it.
fn push_batch(series: &mut Vec<[f64; 4]>, point: [f64; 4]) {
    series.push(point);
    if series.len() > MAX_BATCH_POINTS {
        series.drain(0..series.len() - MAX_BATCH_POINTS);
    }
}

// Appends a diagnostics point, halving the series (like the loss series) once it
// grows past the cap so the whole-run trajectory is preserved while the reattach
// snapshot stays bounded.
fn push_diag(series: &mut Vec<serde_json::Value>, point: serde_json::Value) {
    series.push(point);
    if series.len() > MAX_DIAG_POINTS {
        let mut keep = false;
        series.retain(|_| {
            keep = !keep;
            keep
        });
    }
}
