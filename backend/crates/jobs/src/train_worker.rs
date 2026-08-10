// The training half of a queued job: everything that happens on the worker thread
// once queue.rs has claimed a `train` row and handed it the GPU.
//
// This was `jobs.rs`, an in-memory registry that both scheduled runs and tracked
// them. Scheduling moved to queue.rs (and durability to the `jobs` table), so what
// remains here is the part that was always specific to training: the two-phase
// run, and the accumulated telemetry a socket needs.
//
// Progress is published two ways, and both are needed:
//   - a `broadcast` channel of live frames, for any socket attached right now, and
//   - a `TrainState` snapshot, so a socket attaching late (or re-attaching after a
//     reconnect) catches up to where the run already is.
// The run outlives every socket — closing the tab does not stop training.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use serde::Serialize;
use tokio::sync::{broadcast, mpsc};

use db::model::NewModel;
use crate::queue::{ProgressTick, emit, frame, lock};
use training::{self, TrainConfig, TrainMode};

const MAX_LOG_LINES: usize = 500;
const MAX_LOSS_POINTS: usize = 4000;
// The batch-spread chart is a short rolling window of the most recent reports.
const MAX_BATCH_POINTS: usize = 60;
// The diagnostics charts (grad norms, update ratios, activations, attention,
// embeddings) show the whole run trajectory, so — like the loss series — this is
// halved when it fills rather than windowed, preserving the curve shape.
const MAX_DIAG_POINTS: usize = 1000;

// Phase 1 is a small share of a run's wall clock but needs to move the bar, or a
// long BPE train looks like a hang. The split is deliberately pessimistic about
// tokenization so the bar never stalls at the handover.
const TOKENIZE_SHARE: f64 = 0.1;

// How phase 1's own bar splits between learning the merges and encoding the
// corpus. Encoding is the long pole at scale — the merge table comes off a bounded
// sample in minutes while the streaming pass reads every byte — so it owns the
// larger share and the bar keeps moving through the part that used to be silent.
const MERGE_SHARE: f64 = 0.35;

// Where a run is within itself. Distinct from the job's own `job_status`: that
// says whether the queue is running this row, while this says which of the two
// phases the run is in. The frontend has always switched on these names, so the
// wire shape is unchanged.
#[derive(Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainPhase {
    // Kept for the stored shape of runs recorded before the engine took over
    // dataset building. Nothing enters it now: the engine tokenizes as part of a
    // run and does not report it separately, because at these corpus sizes it is
    // over in seconds and a phase nobody watches is a phase worth not having.
    Tokenizing,
    Training,
    Done,
    Cancelled,
    Failed,
}

impl TrainPhase {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TrainPhase::Done | TrainPhase::Cancelled | TrainPhase::Failed
        )
    }
}

// Catch-up snapshot for a late / re-attaching socket. The live frames carry the
// same data incrementally; this is the accumulated view.
pub struct TrainState {
    pub phase: TrainPhase,
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

// What the worker thread needs to report against. Assembled by queue.rs before the
// thread is spawned, so the socket layer can reach the same state and cancel flag
// while the run is in flight.
pub struct TrainCtx {
    pub cancel: Arc<AtomicBool>,
    pub engine: engine::Client,
    // The worker runs on a plain thread with no reactor, so engine calls are
    // driven through this rather than awaited directly.
    pub handle: tokio::runtime::Handle,
    pub tx: broadcast::Sender<String>,
    pub state: Mutex<TrainState>,
    // Coarse progress for the durable row. The runner throttles these; this side
    // sends freely.
    pub progress_tx: mpsc::UnboundedSender<ProgressTick>,
}

impl TrainCtx {
    pub fn new(
        config: TrainConfig,
        cancel: Arc<AtomicBool>,
        tx: broadcast::Sender<String>,
        progress_tx: mpsc::UnboundedSender<ProgressTick>,
        engine: engine::Client,
        handle: tokio::runtime::Handle,
    ) -> TrainCtx {
        TrainCtx {
            cancel,
            engine,
            handle,
            tx,
            progress_tx,
            state: Mutex::new(TrainState {
                phase: TrainPhase::Training,
                config,
                tokenize: serde_json::json!({}),
                train: serde_json::json!({}),
                loss_series: Vec::new(),
                batch_series: Vec::new(),
                diag_series: Vec::new(),
                log: Vec::new(),
                error: None,
            }),
        }
    }

    fn tick(&self, fraction: f64, stage: &str) {
        // Dropped only once the runner has stopped draining, i.e. after the run is
        // already over; there is nothing to do about it here.
        let _ = self.progress_tx.send(ProgressTick {
            fraction,
            stage: stage.to_string(),
        });
    }
}

// What the run produced. The runner turns this into the job's terminal row: a
// `Done` also registers the model, which is why the record travels back rather
// than being inserted here (this thread can't touch the async pool).
pub enum Outcome {
    Done { model: Box<NewModel> },
    Cancelled,
    Failed(String),
}

// Renders the accumulated state for a socket that just attached.
pub fn snapshot_frame(state: &TrainState) -> String {
    // Only a successful run registered a model, so a reattached, already-`done`
    // run can still offer "Continue to SFT". Absent (null) for in-flight,
    // cancelled, or failed runs. Mirrors the live `done` frame's identifiers.
    let (id_model, model_name) = if state.phase == TrainPhase::Done {
        (
            serde_json::json!(state.config.storage_id),
            serde_json::json!(state.config.model_name),
        )
    } else {
        (serde_json::Value::Null, serde_json::Value::Null)
    };
    serde_json::json!({
        "type": "snapshot",
        "status": state.phase,
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

// Dispatches a run to its mode's worker. Both are two-phase (build dataset, then
// train) and share the progress/finish plumbing; they differ only in phase 1
// (BPE tokenize vs. SFT render) and which train entry point phase 2 drives.
// Takes the context by `Arc` rather than by reference because `ProgressCallback`
// is a `'static` boxed closure: the crate's train loop holds it for the duration
// of the run, so it has to own its handle on the state it reports into.
// Drives one engine run to its terminal state, translating its progress frames
// into the frames the dashboard already speaks.
//
// The engine owns tokenization and the training loop now, so the two phases this
// used to run itself collapse into one call and one event stream.
fn drive(ctx: &Arc<TrainCtx>, start: Result<engine::RunSnapshot, engine::EngineError>) -> Result<serde_json::Value, String> {
    let started = start.map_err(|e| e.to_string())?;
    let engine_client = ctx.engine.clone();
    let cancel = ctx.cancel.clone();
    let reporting = ctx.clone();

    ctx.handle.block_on(async move {
        // The queue signals cancellation with a flag; the engine takes an HTTP
        // call that lands at a step boundary.
        let watcher_client = engine_client.clone();
        let run_id = started.run_id.clone();
        let watcher = tokio::spawn(async move {
            loop {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = watcher_client.cancel(&run_id).await;
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        });

        let finished = engine_client
            .follow(&started.run_id, |event| {
                if let engine::RunEvent::Progress(p) = event {
                    report(&reporting, &p);
                }
            })
            .await
            .map_err(|e| e.to_string());
        watcher.abort();

        outcome(finished?)
    })
}

// The same, over one socket: the config goes up on it, every frame comes back on
// it, and cancellation is a message on it rather than a polling loop holding a
// second connection open. Pretraining is the long run and the one being watched
// frame by frame, so it is the one that takes this path.
fn drive_socket(
    ctx: &Arc<TrainCtx>,
    request: &engine::client::TrainRequest,
) -> Result<serde_json::Value, String> {
    let reporting = ctx.clone();
    let cancel = ctx.cancel.clone();

    ctx.handle.block_on(async {
        let finished = ctx
            .engine
            .train_stream(request, cancel, |event| {
                if let engine::RunEvent::Progress(p) = event {
                    report(&reporting, &p);
                }
            })
            .await
            .map_err(|e| e.to_string())?;

        outcome(finished)
    })
}

// One engine progress frame, translated into the frames the dashboard already
// speaks and folded into the catch-up snapshot.
fn report(ctx: &Arc<TrainCtx>, p: &serde_json::Value) {
    let num = |k: &str| p.get(k).and_then(|v| v.as_f64());
    // Fraction of the whole run, for the durable row's coarse bar.
    let fraction = match (num("step"), num("total_steps")) {
        (Some(step), Some(total)) if total > 0.0 => step / total,
        _ => 0.0,
    };

    if let (Some(step), Some(loss)) = (num("step"), num("loss")) {
        let mut st = lock(&ctx.state);
        push_loss(&mut st.loss_series, [step, loss]);
        // The per-window spread, which the chart draws as a band around the line.
        if let (Some(lo), Some(avg), Some(hi)) =
            (num("batch_min"), num("batch_avg"), num("batch_max"))
        {
            push_batch(&mut st.batch_series, [step, lo, avg, hi]);
        }
        if let Some(d) = p.get("diag").filter(|d| !d.is_null()) {
            push_diag(&mut st.diag_series, d.clone());
        }
        // Kept whole so a socket attaching mid-run gets the same shape as one
        // that watched from the start.
        st.train = p.clone();
        st.phase = TrainPhase::Training;
    }

    let stage = p
        .get("stage_name")
        .and_then(|v| v.as_str())
        .map(|n| format!("Training: {n}"))
        .unwrap_or_else(|| "Training".to_string());
    ctx.tick(fraction, &stage);
    emit(&ctx.tx, frame("train", p.clone()));
}

// A finished run as the queue's Result: cancellation is not a failure, but it
// travels the same channel, so it goes as a sentinel.
fn outcome(finished: engine::RunSnapshot) -> Result<serde_json::Value, String> {
    if finished.is_cancelled() {
        return Err(CANCELLED.to_string());
    }
    if let Some(error) = finished.error {
        return Err(error);
    }
    Ok(finished.result.unwrap_or(serde_json::Value::Null))
}

// Sentinel distinguishing "stopped on request" from a real failure, since both
// come back through the same Result.
const CANCELLED: &str = "\u{0}cancelled";

fn run_pretrain(ctx: Arc<TrainCtx>, mut config: TrainConfig) -> Outcome {
    let created_at = Utc::now().to_rfc3339();
    config.seed = training::resolve_seed(config.seed);

    ctx.tick(0.0, "Training");
    let request = engine::client::TrainRequest {
        model_name: config.storage_id.to_string(),
        dimensions: Some(config.dimensions),
        heads: Some(config.heads),
        layers: Some(config.layers),
        ff_multiplier: Some(config.ff_multiplier),
        ctx_window: Some(config.ctx_window),
        max_vocab: Some(config.max_vocab as i64),
        epochs: Some(config.epochs),
        lr: Some(config.lr as f64),
        datasets: config.datasets.iter().map(|d| d.name.clone()).collect(),
        shuffle: Some(config.shuffle),
        seed: Some(config.seed),
        rebuild_dataset: Some(true),
        ..Default::default()
    };

    match drive_socket(&ctx, &request) {
        Ok(result) => finish(&ctx, &config, &result, &created_at, Ok(false)),
        Err(e) if e == CANCELLED => {
            lock(&ctx.state).phase = TrainPhase::Cancelled;
            Outcome::Cancelled
        }
        Err(e) => fail(&ctx, format!("training failed: {e}")),
    }
}

fn run_sft(ctx: Arc<TrainCtx>, config: TrainConfig) -> Outcome {
    let created_at = Utc::now().to_rfc3339();
    let Some(base) = config.base_model_id else {
        return fail(&ctx, "SFT run is missing a base_model_id".to_string());
    };

    ctx.tick(0.0, "Training");
    let request = engine::client::SftRequest {
        model_name: config.storage_id.to_string(),
        base_model: base.to_string(),
        epochs: Some(config.epochs),
        lr: Some(config.lr as f64),
        strip_thinking: Some(training::strip_thinking_for(config.mode, config.strip_thinking)),
        rebuild_dataset: Some(true),
        ..Default::default()
    };

    let started = ctx.handle.block_on(ctx.engine.sft(&request));
    match drive(&ctx, started) {
        Ok(result) => finish(&ctx, &config, &result, &created_at, Ok(false)),
        Err(e) if e == CANCELLED => {
            lock(&ctx.state).phase = TrainPhase::Cancelled;
            Outcome::Cancelled
        }
        Err(e) => fail(&ctx, format!("fine-tuning failed: {e}")),
    }
}

pub fn run(ctx: Arc<TrainCtx>, config: TrainConfig) -> Outcome {
    // A socket that was already open when this run started never receives the
    // attach-time snapshot, so without this it never learns the run's phase at
    // all — `status` stays null and the page renders the pre-training placeholder
    // for the whole run however many live frames arrive.
    emit(&ctx.tx, snapshot_frame(&lock(&ctx.state)));

    match config.mode {
        TrainMode::Pretrain => run_pretrain(ctx, config),
        TrainMode::Sft => run_sft(ctx, config),
        // RAFT is SFT with a different phase 1, so it shares the whole path and
        // only swaps the dataset builder.
    }
}









// Maps a training outcome (cancelled / completed / errored) onto the terminal
// state, writing the run's meta.json on the way out. `Ok(true)` is the crate's
// "stopped because it was asked to" signal.
fn finish(
    ctx: &TrainCtx,
    config: &TrainConfig,
    // Already-serialized per-mode dataset stats (DatasetStats or SftDatasetStats).
    dataset: &serde_json::Value,
    created_at: &str,
    outcome: std::io::Result<bool>,
) -> Outcome {
    let (phase, status_str) = match &outcome {
        Ok(true) => (TrainPhase::Cancelled, "cancelled"),
        Ok(false) => (TrainPhase::Done, "done"),
        Err(_) => (TrainPhase::Failed, "failed"),
    };

    if let Err(e) = training::write_meta(config, dataset, status_str, created_at) {
        eprintln!("failed to write run meta for {}: {}", config.model_name, e);
    }

    let error = match &outcome {
        Err(e) => Some(format!("training failed: {}", e)),
        _ => None,
    };
    {
        let mut st = lock(&ctx.state);
        st.phase = phase;
        st.error = error.clone();
    }

    let payload = match &error {
        Some(msg) => frame("error", serde_json::json!({ "message": msg })),
        None => {
            let model_dir = engine::paths::data_root().join("artifacts/models").join(config.storage_id.to_string());
            frame(
                "done",
                serde_json::json!({
                    "status": status_str,
                    "model_dir": model_dir.to_string_lossy(),
                    // The just-trained model's identifiers, so the client can seed a
                    // follow-up SFT run with this model as its base without hunting
                    // for it in the model list (the "Continue to SFT" flow).
                    "id_model": config.storage_id,
                    "model_name": config.model_name,
                }),
            )
        }
    };
    emit(&ctx.tx, payload);

    // Only a run that finished registers a model. Cancelled and failed runs may
    // leave partial checkpoints on disk but are deliberately not registered — a
    // half-trained model in the picker is worse than no model.
    match outcome {
        Ok(false) => Outcome::Done {
            model: Box::new(NewModel {
                id_model: config.storage_id,
                name: config.model_name.clone(),
                description: None,
                parameters: serde_json::json!({ "config": config, "dataset": dataset }),
                context: config.ctx_window as i32,
                family: Some("transformer".to_string()),
            }),
        },
        Ok(true) => Outcome::Cancelled,
        Err(e) => Outcome::Failed(format!("training failed: {}", e)),
    }
}

// Terminal failure before a dataset exists (i.e. phase 1 failed).
fn fail(ctx: &TrainCtx, msg: String) -> Outcome {
    {
        let mut st = lock(&ctx.state);
        st.phase = TrainPhase::Failed;
        st.error = Some(msg.clone());
    }
    emit(
        &ctx.tx,
        frame("error", serde_json::json!({ "message": msg })),
    );
    Outcome::Failed(msg)
}

// ── series caps ─────────────────────────────────────────────────────────────

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
