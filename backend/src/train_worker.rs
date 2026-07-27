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

use crate::db::model::NewModel;
use crate::queue::{ProgressTick, emit, frame, lock};
use crate::training::{self, TokenizeEvent, TrainConfig, TrainMode};

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

// Where a run is within itself. Distinct from the job's own `job_status`: that
// says whether the queue is running this row, while this says which of the two
// phases the run is in. The frontend has always switched on these names, so the
// wire shape is unchanged.
#[derive(Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainPhase {
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
    ) -> TrainCtx {
        TrainCtx {
            cancel,
            tx,
            progress_tx,
            state: Mutex::new(TrainState {
                phase: TrainPhase::Tokenizing,
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
pub fn run(ctx: Arc<TrainCtx>, config: TrainConfig) -> Outcome {
    match config.mode {
        TrainMode::Pretrain => run_pretrain(ctx, config),
        TrainMode::Sft => run_sft(ctx, config),
        TrainMode::Kto => run_kto(ctx, config),
        // RAFT is SFT with a different phase 1, so it shares the whole path and
        // only swaps the dataset builder.
        TrainMode::Raft => run_sft(ctx, config),
    }
}

// Runs both phases for a KTO preference job: render the thumbs-rated replies into
// a masked dataset (scoring each one under the base model for its reference
// log-probability), then preference-tune the base.
//
// Structurally identical to `run_sft` — same base resolution, same architecture
// inheritance, same finish plumbing — because it is the same two-phase shape with a
// different phase 1 and a different train entry point.
fn run_kto(ctx: Arc<TrainCtx>, mut config: TrainConfig) -> Outcome {
    let created_at = Utc::now().to_rfc3339();

    let base_model_dir = match config.base_model_id {
        Some(id) => crate::storage::models_dir().join(id.to_string()),
        None => return fail(&ctx, "KTO run is missing a base_model_id".to_string()),
    };

    if let Err(e) = training::apply_base_architecture(&mut config, &base_model_dir) {
        eprintln!(
            "[training] {}: could not inherit base architecture ({}); registry metadata may be off",
            config.model_name, e
        );
    }

    // ---- Phase 1: rated replies → kto_corpus.bin + kto_mask.bin + kto_index.json ----
    //
    // This phase is slower than SFT's: it runs a forward pass per example to record
    // the reference log-probabilities. That cost is paid once here rather than every
    // epoch in phase 2.
    eprintln!("[training] {}: building KTO dataset…", config.model_name);
    ctx.tick(0.0, "Scoring rated replies under the base model");
    let dataset = match training::build_kto_dataset(&base_model_dir) {
        Ok(stats) => {
            if stats.desirable == 0 || stats.undesirable == 0 {
                // One-sided feedback has no contrast to learn from: the model can
                // only be told to push everything up or everything down, which is
                // not a preference. Better to refuse than to burn a run producing a
                // uniformly-shifted model.
                return fail(
                    &ctx,
                    format!(
                        "preference tuning needs both thumbs up and thumbs down — got {} up and {} down",
                        stats.desirable, stats.undesirable
                    ),
                );
            }
            let obj = serde_json::json!({
                "examples": stats.examples,
                "desirable": stats.desirable,
                "undesirable": stats.undesirable,
                "total_tokens": stats.total_tokens,
                "trainable_tokens": stats.trainable_tokens,
                "skipped_too_long": stats.skipped_too_long,
                "skipped_no_reply": stats.skipped_no_reply,
                "progress": 1.0,
                "done": true,
            });
            {
                let mut st = lock(&ctx.state);
                st.tokenize = obj.clone();
                st.phase = TrainPhase::Training;
            }
            emit(&ctx.tx, frame("tokenize", obj));
            serde_json::json!(stats)
        }
        Err(e) => return fail(&ctx, format!("KTO dataset build failed: {}", e)),
    };

    // ---- Phase 2: preference-tune from the base ----
    eprintln!(
        "[training] {}: starting KTO tune ({} epochs, beta {}, base {})",
        config.model_name,
        config.epochs,
        config.beta,
        base_model_dir.display(),
    );
    let reporting = ctx.clone();
    let on_progress: llm::train::ProgressCallback =
        Box::new(move |p| handle_train_progress(&reporting, p));

    let outcome = training::run_kto_training(&config, on_progress, ctx.cancel.clone());
    eprintln!("[training] {}: KTO returned {:?}", config.model_name, outcome);
    finish(&ctx, &config, &dataset, &created_at, outcome)
}

// Runs both phases for a pretrain job. Both crate calls are synchronous and
// blocking, which is exactly why the caller gives this a plain OS thread and not a
// task on the actix runtime.
fn run_pretrain(ctx: Arc<TrainCtx>, mut config: TrainConfig) -> Outcome {
    let created_at = Utc::now().to_rfc3339();

    // Resolve the shuffle seed now (0 => a fresh time seed) so the value that
    // actually drives the run is the one persisted to meta.json — a shuffled run
    // must be reproducible from its recorded config.
    config.seed = training::resolve_seed(config.seed);

    // ---- Phase 1: tokenize / build the dataset ----
    // Curriculum runs build per-stage document spans; a plain run collapses to a
    // single implicit stage. `resolve_stages` unifies the two.
    eprintln!("[training] {}: tokenizing uploads…", config.model_name);
    ctx.tick(0.0, "Tokenizing");
    let stages = training::resolve_stages(&config);
    let dataset = training::build_dataset(config.max_vocab, &stages, |event| {
        handle_tokenize_event(&ctx, event);
    });

    let dataset = match dataset {
        Ok(stats) => {
            let obj = serde_json::json!({
                "files": stats.files,
                "chars": stats.chars,
                "vocab": stats.vocab_size,
                "merges": stats.merges,
                "documents": stats.documents,
                "progress": 1.0,
                "done": true,
            });
            {
                let mut st = lock(&ctx.state);
                st.tokenize = obj.clone();
                st.phase = TrainPhase::Training;
            }
            emit(&ctx.tx, frame("tokenize", obj));
            stats
        }
        Err(e) => return fail(&ctx, format!("tokenization failed: {}", e)),
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
    let reporting = ctx.clone();
    let on_progress: llm::train::ProgressCallback =
        Box::new(move |p| handle_train_progress(&reporting, p));

    let outcome = training::run_training(&config, on_progress, ctx.cancel.clone());
    eprintln!("[training] {}: training returned {:?}", config.model_name, outcome);
    finish(&ctx, &config, &serde_json::json!(dataset), &created_at, outcome)
}

// Runs both phases for an SFT job: render the chat conversations into the masked
// dataset using the base model's tokenizer, then fine-tune the base. Shares the
// progress/finish plumbing with the pretrain path.
fn run_sft(ctx: Arc<TrainCtx>, mut config: TrainConfig) -> Outcome {
    let created_at = Utc::now().to_rfc3339();

    // The base model to fine-tune from; its tokenizer renders the dataset and its
    // weights are the starting point. The enqueue route guarantees this is present
    // and resolves, but we resolve defensively (the id is untrusted here).
    let base_model_dir = match config.base_model_id {
        Some(id) => crate::storage::models_dir().join(id.to_string()),
        None => return fail(&ctx, "SFT run is missing a base_model_id".to_string()),
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
    //
    // The two modes differ here and nowhere else: SFT renders the hand-written
    // conversations in raw/sft/, RAFT renders the catalogue entries this model's own
    // generations earned an "up" for. Both produce the same two files under the same
    // masking convention, so phase 2 is byte-for-byte the same run.
    let is_raft = config.mode == TrainMode::Raft;
    let what = if is_raft { "RAFT" } else { "SFT" };
    eprintln!("[training] {}: building {} dataset…", config.model_name, what);
    ctx.tick(0.0, &format!("Building {} dataset", what));
    let built = if is_raft {
        training::build_raft_dataset(&base_model_dir).map(|s| {
            (
                serde_json::json!({
                    "conversations": s.examples,
                    "total_tokens": s.total_tokens,
                    "trainable_tokens": s.trainable_tokens,
                    "duplicates": s.duplicates,
                    "skipped_too_long": s.skipped_too_long,
                    "progress": 1.0,
                    "done": true,
                }),
                serde_json::json!(s),
            )
        })
    } else {
        training::build_sft_dataset(&base_model_dir).map(|s| {
            (
                serde_json::json!({
                    "conversations": s.conversations,
                    "total_tokens": s.total_tokens,
                    "trainable_tokens": s.trainable_tokens,
                    "progress": 1.0,
                    "done": true,
                }),
                serde_json::json!(s),
            )
        })
    };
    let dataset = match built {
        Ok((obj, stats)) => {
            {
                let mut st = lock(&ctx.state);
                st.tokenize = obj.clone();
                st.phase = TrainPhase::Training;
            }
            emit(&ctx.tx, frame("tokenize", obj));
            stats
        }
        Err(e) => return fail(&ctx, format!("{} dataset build failed: {}", what, e)),
    };

    // ---- Phase 2: fine-tune from the base ----
    eprintln!(
        "[training] {}: starting {} fine-tune ({} epochs, base {})",
        config.model_name,
        what,
        config.epochs,
        base_model_dir.display(),
    );
    let reporting = ctx.clone();
    let on_progress: llm::train::ProgressCallback =
        Box::new(move |p| handle_train_progress(&reporting, p));

    let outcome = training::run_sft_training(&config, on_progress, ctx.cancel.clone());
    eprintln!("[training] {}: {} returned {:?}", config.model_name, what, outcome);
    finish(&ctx, &config, &dataset, &created_at, outcome)
}

fn handle_tokenize_event(ctx: &TrainCtx, event: TokenizeEvent) {
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
            lock(&ctx.state).tokenize = obj.clone();
            emit(&ctx.tx, frame("tokenize", obj));
        }
        TokenizeEvent::Merge { done, total } => {
            let progress = if total > 0 {
                done as f64 / total as f64
            } else {
                0.0
            };
            // Keep files/chars already recorded by the Start event.
            let obj = {
                let mut st = lock(&ctx.state);
                if let Some(map) = st.tokenize.as_object_mut() {
                    map.insert(
                        "vocab".into(),
                        serde_json::json!(tokenize::bpe::vocab_size(done as usize)),
                    );
                    map.insert("merges".into(), serde_json::json!(done));
                    map.insert("progress".into(), serde_json::json!(progress));
                }
                st.tokenize.clone()
            };
            ctx.tick(
                TOKENIZE_SHARE * progress,
                &format!("Tokenizing ({}/{} merges)", done, total),
            );
            emit(&ctx.tx, frame("tokenize", obj));
        }
        TokenizeEvent::Log(line) => {
            push_capped(&mut lock(&ctx.state).log, line.to_string());
            emit(&ctx.tx, frame("log", serde_json::json!({ "line": line })));
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
        "emb_pca3": d.emb_pca3,
    })
}

fn handle_train_progress(ctx: &TrainCtx, p: llm::train::TrainingProgress) {
    // Run-global chunk count: a monotonic x-axis across all curriculum stages
    // (per-epoch `chunk` resets and `total_chunks` differs between stages, so the
    // old (epoch-1)*total_chunks+chunk formula would jump backwards at a stage
    // boundary).
    let step = p.global_step;
    // Render the diagnostics (present only on mid-epoch report frames) before the
    // main object so it can be embedded in the live frame and stored for reattach.
    let diag = p.diagnostics.as_ref().map(|d| diag_frame(step, d));
    let obj = serde_json::json!({
        "epoch": p.epoch,
        "total_epochs": p.total_epochs,
        "chunk": p.chunk,
        "total_chunks": p.total_chunks,
        "stage": p.stage,
        "stage_name": p.stage_name,
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
        let mut st = lock(&ctx.state);
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
    ctx.tick(TOKENIZE_SHARE + (1.0 - TOKENIZE_SHARE) * epoch_fraction(&p), &stage_label(&p));
    emit(&ctx.tx, frame("train", obj));
}

// How far through the run we are, as epochs completed plus progress within the
// current one. Approximate for a curriculum run, whose stages have different chunk
// counts — good enough for a coarse bar, and the exact position is in the frames.
fn epoch_fraction(p: &llm::train::TrainingProgress) -> f64 {
    if p.total_epochs <= 0 {
        return 0.0;
    }
    let within = if p.total_chunks > 0 {
        p.chunk as f64 / p.total_chunks as f64
    } else {
        0.0
    };
    let done = (p.epoch as f64 - 1.0 + within) / p.total_epochs as f64;
    done.clamp(0.0, 1.0)
}

fn stage_label(p: &llm::train::TrainingProgress) -> String {
    match &p.stage_name {
        Some(name) => format!("{} · epoch {}/{}", name, p.epoch, p.total_epochs),
        None => format!("Epoch {}/{}", p.epoch, p.total_epochs),
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
            let model_dir = crate::storage::models_dir().join(config.storage_id.to_string());
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
