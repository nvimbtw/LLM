use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::training::gpu_workspace::Workspace;
use crate::training::transformer::Model;
use crate::training::optimizer::{gpu_adam_update, gpu_clip_gradients};
use crate::training::sync::sync_weights_from_gpu;

// Early-stopping patience: stop after this many consecutive epochs without an
// average-loss improvement.
const EARLY_STOP_PATIENCE: i32 = 3;
// Gradient-accumulation factor: the optimizer steps once per this many chunks,
// and the summed gradients are averaged by it.
const ACCUM_STEPS: usize = 4;
// Loss-spike guard: per-step gradient L2 norm to clip to. The gradient buffers
// hold the SUM over ACCUM_STEPS, so this is scaled by ACCUM_STEPS at the step to
// correspond to a ~1.0 per-step clip.
const MAX_GRAD_NORM: f32 = 1.0;
// Linear LR warmup length, in optimizer steps. The opening steps on a fresh
// (especially deep) model are the most unstable, so ramping lr up from ~0 avoids
// the early divergence that clipping alone only slows down.
//
// Note for SFT (`sft_train`): fine-tuning runs are short, so a fixed 100-step
// warmup can be a large fraction of (or exceed) the whole run, meaning the
// effective lr never reaches the requested `lr`. This is left as-is on purpose —
// a base model is already near a good minimum, so a conservative, mostly-warming
// lr is a safe default and avoids changing the well-tuned pretraining behaviour.
// Revisit (e.g. warmup as a fraction of total steps) only if SFT underfits.
const WARMUP_STEPS: i32 = 100;
// How often (in trained chunks) to emit a `train` progress frame over the
// WebSocket. At the old hard-coded interval of 10, a large corpus fired dozens
// of frames per second; Recharts re-renders synchronously on every point, so
// the chart visibly jittered. 100 keeps the graphs responsive while cutting
// the update rate 10x.
const PROGRESS_REPORT_INTERVAL: u64 = 100;

pub struct TrainingProgress {
    pub epoch: u64,
    pub total_epochs: u64,
    pub chunk: u64,
    pub total_chunks: u64,
    pub loss: f64,
    pub eta_secs: f64,
    pub epoch_done: bool,
    pub avg_loss: f32,
    pub min_loss: f32,
    pub max_loss: f32,
    // Spread of per-chunk loss over just this report window (the ~10 chunks since
    // the last report), as opposed to the epoch-cumulative avg/min/max above.
    pub batch_min: f32,
    pub batch_avg: f32,
    pub batch_max: f32,
    pub epoch_duration_secs: f64,
    pub checkpoint_path: Option<String>,
}

pub type ProgressCallback = Box<dyn Fn(TrainingProgress) + Send>;

pub fn train(
    dimensions: i64,
    ctx_window: i64,
    heads: i64,
    layers: i64,
    epochs: i64,
    lr: f32,
    ff_multiplier: i64,
    model_dir: &Path,
    corpus_path: &Path,
    merges_path: &Path,
    checkpoints_dir: Option<&Path>,
    on_progress: ProgressCallback,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<bool, io::Error> {
    let corpus_bytes = fs::read(corpus_path)?;
    let corpus: Vec<u32> = corpus_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let merges = tokenize::bpe::load_merges(&merges_path.to_string_lossy())?;
    let vocab = merges.len() as i64 + 256;

    // The caller owns the layout (via the backend's storage module); we only
    // create the run directory and stash a copy of the tokenizer merges next to
    // the weights so inference can load both from one place.
    fs::create_dir_all(model_dir)?;
    let target_merges = model_dir.join("merges.bin");
    fs::copy(merges_path, &target_merges)?;

    if let Some(cp_dir) = checkpoints_dir {
        fs::create_dir_all(cp_dir)?;
    }

    let mut transformer =
        Model::init(dimensions, vocab, ctx_window, heads, ff_multiplier, layers);
    let mut ws = Workspace::new(&transformer, ctx_window as usize);

    run_training_loop(
        &mut transformer,
        &mut ws,
        &corpus,
        None,
        ctx_window,
        layers,
        epochs,
        lr,
        model_dir,
        checkpoints_dir,
        on_progress,
        cancel,
    )
}

pub fn continue_train(
    model_dir: &Path,
    corpus_path: &Path,
    checkpoint_path: Option<&str>,
    additional_epochs: i64,
    lr: f32,
    checkpoints_dir: Option<&Path>,
    on_progress: ProgressCallback,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<bool, io::Error> {
    let mut transformer = if let Some(cp) = checkpoint_path {
        let bytes = fs::read(cp)?;
        Model::from_bytes(&bytes)?
    } else {
        Model::load(&model_dir.to_string_lossy())?
    };
    let ctx_window = transformer.positional_table.len() as i64;
    let layers = transformer.layers;
    let mut ws = Workspace::new(&transformer, ctx_window as usize);

    let corpus_bytes = fs::read(corpus_path)?;
    let corpus: Vec<u32> = corpus_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    if let Some(cp_dir) = checkpoints_dir {
        fs::create_dir_all(cp_dir)?;
    }

    run_training_loop(
        &mut transformer,
        &mut ws,
        &corpus,
        None,
        ctx_window,
        layers,
        additional_epochs,
        lr,
        model_dir,
        checkpoints_dir,
        on_progress,
        cancel,
    )
}

/// Fine-tune (supervised fine-tuning) a **pretrained base model** on a masked
/// chat dataset. Loads the base weights from `base_model_dir`, then runs the
/// training loop over `sft_corpus_path` (flat little-endian `u32` ids, exactly
/// like `corpus.bin`) driven by the per-position loss mask in `sft_mask_path`
/// (one byte per token, `1`=train / `0`=ignore per the ISSUE-0031 convention).
/// Only the masked-in positions — the assistant reply tokens — contribute loss
/// and gradient; system/user/scaffold tokens are forward-passed but produce no
/// learning signal, so the base model is nudged toward answering in the chat
/// shape without being retrained on the prompt text.
///
/// The corpus and mask must be the same length (asserted; SFT is meaningless if
/// they drift). The base `merges.bin` is copied into `out_model_dir` so the
/// fine-tuned model loads standalone at inference time.
///
/// `lr` and `epochs` are caller-supplied on purpose: SFT typically wants a
/// *lower* LR and *fewer* epochs than pretraining (a base model is already near
/// a good minimum). Good starting points are ~1e-4 to 5e-5 for a handful of
/// epochs — an order of magnitude below a from-scratch run. Returns `Ok(true)`
/// if training was cancelled, `Ok(false)` if it ran to completion.
#[allow(clippy::too_many_arguments)]
pub fn sft_train(
    base_model_dir: &Path,
    sft_corpus_path: &Path,
    sft_mask_path: &Path,
    out_model_dir: &Path,
    epochs: i64,
    lr: f32,
    checkpoints_dir: Option<&Path>,
    on_progress: ProgressCallback,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<bool, io::Error> {
    let mut transformer = Model::load(&base_model_dir.to_string_lossy())?;
    let ctx_window = transformer.positional_table.len() as i64;
    let layers = transformer.layers;
    let mut ws = Workspace::new(&transformer, ctx_window as usize);

    let corpus_bytes = fs::read(sft_corpus_path)?;
    let corpus: Vec<u32> = corpus_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mask: Vec<u8> = fs::read(sft_mask_path)?;

    // The mask is aligned 1:1 with the corpus tokens; a mismatch means the
    // dataset was built inconsistently and every chunk's mask slice would be
    // misaligned. Fail cleanly rather than train on garbage.
    if corpus.len() != mask.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "SFT corpus/mask length mismatch: {} tokens vs {} mask bytes",
                corpus.len(),
                mask.len()
            ),
        ));
    }

    // Copy the base tokenizer next to the fine-tuned weights so inference can
    // load both from `out_model_dir` alone (mirrors what `train` does).
    fs::create_dir_all(out_model_dir)?;
    let base_merges = base_model_dir.join("merges.bin");
    if base_merges.is_file() {
        fs::copy(&base_merges, out_model_dir.join("merges.bin"))?;
    }

    if let Some(cp_dir) = checkpoints_dir {
        fs::create_dir_all(cp_dir)?;
    }

    run_training_loop(
        &mut transformer,
        &mut ws,
        &corpus,
        Some(&mask),
        ctx_window,
        layers,
        epochs,
        lr,
        out_model_dir,
        checkpoints_dir,
        on_progress,
        cancel,
    )
}

struct EpochStats {
    avg_loss: f32,
    min_loss: f32,
    max_loss: f32,
    duration_secs: f64,
}

// Running min / mean / max over a set of per-chunk losses. Used both for the
// whole-epoch summary and for each progress-report window (reset between
// reports). `min`/`max` stay at ±INFINITY and `avg` is NaN until the first
// `record`, matching the sentinel behaviour of the pre-refactor accumulators.
struct LossStats {
    total: f32,
    n: u64,
    min: f32,
    max: f32,
}

impl LossStats {
    fn new() -> Self {
        LossStats { total: 0.0, n: 0, min: f32::INFINITY, max: f32::NEG_INFINITY }
    }

    fn record(&mut self, loss: f32) {
        self.total += loss;
        self.n += 1;
        self.min = self.min.min(loss);
        self.max = self.max.max(loss);
    }

    fn avg(&self) -> f32 {
        if self.n > 0 {
            self.total / self.n as f32
        } else {
            f32::NAN
        }
    }
}

// Estimated seconds remaining, from the average time per chunk so far. NaN
// before any chunk has completed.
fn compute_eta(training_start: Instant, chunks_done_global: u64, total_chunks_all: u64) -> f64 {
    if chunks_done_global == 0 {
        return f64::NAN;
    }
    let secs_per = training_start.elapsed().as_secs_f64() / chunks_done_global as f64;
    secs_per * (total_chunks_all - chunks_done_global) as f64
}

// Sync GPU weights back to the host and persist them: a per-epoch checkpoint (if
// a checkpoints dir is configured) plus the model dir for final-model access.
// Returns the checkpoint path, if one was written.
fn save_checkpoint(
    transformer: &mut Model,
    ws: &Workspace,
    model_dir: &Path,
    checkpoints_dir: Option<&Path>,
    epoch: i64,
) -> Option<String> {
    sync_weights_from_gpu(ws, transformer);
    let checkpoint_path = checkpoints_dir.map(|cp_dir| {
        let cp_str = cp_dir
            .join(format!("epoch_{:04}.bin", epoch + 1))
            .to_string_lossy()
            .to_string();
        transformer.save_to_path(&cp_str).ok();
        cp_str
    });
    let _ = transformer.save(&model_dir.to_string_lossy().to_string());
    println!("Checkpoint saved");
    checkpoint_path
}

// Train one full pass over the corpus: forward/backward each chunk, step the
// optimizer every ACCUM_STEPS chunks, and emit a mid-epoch progress report every
// PROGRESS_REPORT_INTERVAL chunks. Advances `adam_step` and returns the epoch's
// loss summary.
#[allow(clippy::too_many_arguments)]
fn train_epoch(
    transformer: &mut Model,
    ws: &mut Workspace,
    corpus: &[u32],
    // SFT loss mask aligned 1:1 with `corpus`, or `None` for plain pretraining.
    // When present, each chunk trains only on its masked-in (assistant) tokens.
    mask: Option<&[u8]>,
    ctx_window: i64,
    layers: i64,
    lr: f32,
    epoch: i64,
    epochs: i64,
    total_chunks: usize,
    total_chunks_all: u64,
    training_start: Instant,
    adam_step: &mut i32,
    on_progress: &ProgressCallback,
) -> EpochStats {
    let mut epoch_stats = LossStats::new();
    let mut window = LossStats::new();
    let epoch_start = Instant::now();

    ws.zero_accumulators();

    for (chunk_idx, chunk) in corpus.chunks(ctx_window as usize).enumerate() {
        if chunk.len() < 2 {
            continue;
        }
        // Slice the mask with the SAME boundaries as the corpus chunks. Chunks
        // are contiguous and fixed-size (only the last may be shorter), so the
        // chunk's flat offset is `chunk_idx * ctx_window`.
        let loss = match mask {
            Some(m) => {
                let start = chunk_idx * ctx_window as usize;
                let mask_slice = &m[start..start + chunk.len()];
                pollster::block_on(transformer.train_step_masked(
                    ws,
                    chunk,
                    mask_slice,
                    layers as usize,
                ))
            }
            None => pollster::block_on(transformer.train_step(ws, chunk, layers as usize)),
        };

        if !loss.is_finite() {
            eprint!("  ⚠ NaN loss — zeroing accumulators and skipping chunk");
            ws.zero_accumulators();
            continue;
        }

        epoch_stats.record(loss);
        window.record(loss);

        if epoch_stats.n % (ACCUM_STEPS as u64) == 0 || epoch_stats.n == total_chunks as u64 {
            step_optimizer(ws, layers, lr, adam_step);
        }

        if epoch_stats.n % PROGRESS_REPORT_INTERVAL == 0 || epoch_stats.n == total_chunks as u64 {
            report_chunk_progress(
                epoch, epochs, total_chunks, total_chunks_all, training_start, loss,
                &epoch_stats, &window, on_progress,
            );
            // Start a fresh window for the next report.
            window = LossStats::new();
        }
    }

    EpochStats {
        avg_loss: epoch_stats.avg(),
        min_loss: epoch_stats.min,
        max_loss: epoch_stats.max,
        duration_secs: epoch_start.elapsed().as_secs_f64(),
    }
}

// Apply one accumulated optimizer step: clip the summed gradient, take a
// warmup-scaled Adam step, then zero the accumulators for the next window.
fn step_optimizer(ws: &mut Workspace, layers: i64, lr: f32, adam_step: &mut i32) {
    *adam_step += 1;
    // Clip the accumulated gradient BEFORE the step so a rare huge gradient
    // can't blow the model up (the loss-spike-to-50 case).
    let clip = MAX_GRAD_NORM * ACCUM_STEPS as f32;
    let gnorm = gpu_clip_gradients(ws, layers as usize, clip);
    // Warmup-scaled lr. Adam averages the gradient internally (grad_scale), so
    // pass the full (warmed) lr, not lr/ACCUM_STEPS.
    let warm = (*adam_step as f32 / WARMUP_STEPS as f32).min(1.0);
    gpu_adam_update(ws, lr * warm, *adam_step, ACCUM_STEPS, layers as usize);
    ws.zero_accumulators();
    // Periodic gradient-norm readout (and always on a spike) so a divergent run
    // is visible: a norm that keeps growing = trouble.
    if *adam_step % 50 == 0 || gnorm > clip * 20.0 || !gnorm.is_finite() {
        eprintln!("  step {} grad_norm {:.2}", *adam_step, gnorm);
    }
}

// Emit a mid-epoch progress frame: epoch-cumulative avg/min/max plus the batch
// spread over just this report window. The caller resets `window` afterward.
#[allow(clippy::too_many_arguments)] // cohesive progress-frame builder; the fields belong together
fn report_chunk_progress(
    epoch: i64,
    epochs: i64,
    total_chunks: usize,
    total_chunks_all: u64,
    training_start: Instant,
    loss: f32,
    epoch_stats: &LossStats,
    window: &LossStats,
    on_progress: &ProgressCallback,
) {
    let chunks_done_global = (epoch as u64) * (total_chunks as u64) + epoch_stats.n;
    on_progress(TrainingProgress {
        epoch: (epoch + 1) as u64,
        total_epochs: epochs as u64,
        chunk: epoch_stats.n,
        total_chunks: total_chunks as u64,
        loss: loss as f64,
        eta_secs: compute_eta(training_start, chunks_done_global, total_chunks_all),
        epoch_done: false,
        // Running stats so far this epoch (n >= 1 here), so the UI shows a live
        // average instead of a frozen 0.
        avg_loss: epoch_stats.avg(),
        min_loss: epoch_stats.min,
        max_loss: epoch_stats.max,
        // Spread over just the chunks in this window.
        batch_min: window.min,
        batch_avg: window.avg(),
        batch_max: window.max,
        epoch_duration_secs: 0.0,
        checkpoint_path: None,
    });
}

#[allow(clippy::too_many_arguments)]
fn run_training_loop(
    transformer: &mut Model,
    ws: &mut Workspace,
    corpus: &[u32],
    // SFT loss mask aligned 1:1 with `corpus`, threaded to `train_epoch`; `None`
    // for pretraining (`train`/`continue_train`).
    mask: Option<&[u8]>,
    ctx_window: i64,
    layers: i64,
    epochs: i64,
    lr: f32,
    model_dir: &Path,
    checkpoints_dir: Option<&Path>,
    on_progress: ProgressCallback,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<bool, io::Error> {
    ws.ensure_layer_caches(layers as usize);
    let total_chunks: usize = corpus
        .chunks(ctx_window as usize)
        .filter(|c| c.len() >= 2)
        .count();
    let total_chunks_all: u64 = (epochs as u64) * (total_chunks as u64);
    let training_start = Instant::now();
    let mut best_loss = f32::INFINITY;
    let mut epochs_since_best = 0;
    // Global optimizer-step counter for Adam bias correction (persists across
    // epochs, advanced inside train_epoch).
    let mut adam_step: i32 = 0;

    for epoch in 0..epochs {
        let stats = train_epoch(
            transformer, ws, corpus, mask, ctx_window, layers, lr, epoch, epochs,
            total_chunks, total_chunks_all, training_start, &mut adam_step, &on_progress,
        );
        println!(
            "Epoch {}: avg loss = {:.6} (min={:.4} max={:.4})",
            epoch + 1,
            stats.avg_loss,
            stats.min_loss,
            stats.max_loss
        );

        let checkpoint_path = save_checkpoint(transformer, ws, model_dir, checkpoints_dir, epoch);
        report_epoch_progress(
            epoch, epochs, total_chunks, total_chunks_all, training_start, &stats,
            checkpoint_path, &on_progress,
        );

        if let Some(ref cancel) = cancel {
            if cancel.load(Ordering::SeqCst) {
                println!("Training cancelled by user at epoch {}", epoch + 1);
                return Ok(true);
            }
        }

        if should_stop_early(stats.avg_loss, epoch, &mut best_loss, &mut epochs_since_best) {
            break;
        }
    }

    Ok(false)
}

// Emit the end-of-epoch progress frame: the epoch summary, with no batch window
// (batch fields are NaN so the frontend skips them for the batch chart).
#[allow(clippy::too_many_arguments)] // cohesive progress-frame builder; the fields belong together
fn report_epoch_progress(
    epoch: i64,
    epochs: i64,
    total_chunks: usize,
    total_chunks_all: u64,
    training_start: Instant,
    stats: &EpochStats,
    checkpoint_path: Option<String>,
    on_progress: &ProgressCallback,
) {
    let chunks_done_global = ((epoch + 1) as u64) * (total_chunks as u64);
    on_progress(TrainingProgress {
        epoch: (epoch + 1) as u64,
        total_epochs: epochs as u64,
        chunk: total_chunks as u64,
        total_chunks: total_chunks as u64,
        loss: stats.avg_loss as f64,
        eta_secs: compute_eta(training_start, chunks_done_global, total_chunks_all),
        epoch_done: true,
        avg_loss: stats.avg_loss,
        min_loss: stats.min_loss,
        max_loss: stats.max_loss,
        // No window at an epoch boundary; NaN serializes to null and the
        // frontend skips it for the batch chart.
        batch_min: f32::NAN,
        batch_avg: f32::NAN,
        batch_max: f32::NAN,
        epoch_duration_secs: stats.duration_secs,
        checkpoint_path,
    });
}

// Advance early-stopping state and report whether training should halt: true
// after EARLY_STOP_PATIENCE consecutive epochs with no average-loss improvement.
fn should_stop_early(
    avg_loss: f32,
    epoch: i64,
    best_loss: &mut f32,
    epochs_since_best: &mut i32,
) -> bool {
    if avg_loss < *best_loss {
        *best_loss = avg_loss;
        *epochs_since_best = 0;
    } else {
        *epochs_since_best += 1;
    }
    if *epochs_since_best >= EARLY_STOP_PATIENCE {
        println!(
            "Epoch {}: early stopping (no improvement for {} epochs)",
            epoch + 1,
            EARLY_STOP_PATIENCE
        );
        return true;
    }
    false
}
