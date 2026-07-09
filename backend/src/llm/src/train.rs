use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::training::gpu_workspace::GpuWorkspace;
use crate::training::transformer::{
    gpu_adam_update, gpu_clip_gradients, sync_weights_from_gpu, Transformer,
};

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
        Transformer::init(dimensions, vocab, ctx_window, heads, ff_multiplier, layers);
    let mut ws = GpuWorkspace::new(&transformer, ctx_window as usize);

    run_training_loop(
        &mut transformer,
        &mut ws,
        &corpus,
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
        Transformer::from_bytes(&bytes)?
    } else {
        Transformer::load(&model_dir.to_string_lossy())?
    };
    let ctx_window = transformer.positional_table.len() as i64;
    let layers = transformer.layers;
    let mut ws = GpuWorkspace::new(&transformer, ctx_window as usize);

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

fn run_training_loop(
    transformer: &mut Transformer,
    ws: &mut GpuWorkspace,
    corpus: &[u32],
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

    let patience = 3;
    let mut best_loss = f32::INFINITY;
    let mut epochs_since_best = 0;
    let training_start = Instant::now();
    let accum_steps = 4usize;
    // Loss-spike guard: clip the per-step gradient to this L2 norm. The gradient
    // buffers hold the SUM over accum_steps, so the clip threshold is scaled by
    // accum_steps below to correspond to a ~1.0 per-step clip.
    let max_grad_norm = 1.0f32;
    // Linear LR warmup over the first N optimizer steps. The opening steps on a
    // fresh (especially deep) model are the most unstable, so ramping lr up from
    // ~0 avoids the early divergence that clipping alone only slows down.
    let warmup_steps = 100i32;
    // Global optimizer-step counter for Adam bias correction (persists across
    // epochs, incremented once per weight update).
    let mut adam_step: i32 = 0;

    for epoch in 0..epochs {
        let mut total_loss = 0.0f32;
        let mut n_chunks = 0;
        let mut min_loss = f32::INFINITY;
        let mut max_loss = f32::NEG_INFINITY;
        // Per-report-window accumulators, reset after each report emitted below.
        let mut win_total = 0.0f32;
        let mut win_n = 0u64;
        let mut win_min = f32::INFINITY;
        let mut win_max = f32::NEG_INFINITY;
        let epoch_start = Instant::now();

        ws.zero_accumulators();

        for chunk in corpus.chunks(ctx_window as usize) {
            if chunk.len() < 2 {
                continue;
            }
            let loss = pollster::block_on(transformer.train_step(ws, chunk, layers as usize));

            if !loss.is_finite() {
                eprint!("  ⚠ NaN loss — zeroing accumulators and skipping chunk");
                ws.zero_accumulators();
                continue;
            }

            total_loss += loss;
            n_chunks += 1;
            min_loss = min_loss.min(loss);
            max_loss = max_loss.max(loss);

            win_total += loss;
            win_n += 1;
            win_min = win_min.min(loss);
            win_max = win_max.max(loss);

            if n_chunks % (accum_steps as u64) == 0 || n_chunks == total_chunks as u64 {
                adam_step += 1;
                // Clip the accumulated gradient BEFORE the step so a rare huge
                // gradient can't blow the model up (the loss-spike-to-50 case).
                let clip = max_grad_norm * accum_steps as f32;
                let gnorm = gpu_clip_gradients(&ws, layers as usize, clip);
                // Warmup-scaled lr. Adam averages the gradient internally
                // (grad_scale), so pass the full (warmed) lr, not lr/accum_steps.
                let warm = (adam_step as f32 / warmup_steps as f32).min(1.0);
                gpu_adam_update(&ws, lr * warm, adam_step, accum_steps, layers as usize);
                ws.zero_accumulators();
                // Periodic gradient-norm readout (and always on a spike) so a
                // divergent run is visible: a norm that keeps growing = trouble.
                if adam_step % 50 == 0 || gnorm > clip * 20.0 || !gnorm.is_finite() {
                    eprintln!("  step {} grad_norm {:.2}", adam_step, gnorm);
                }
            }

            if n_chunks % 10 == 0 || n_chunks == total_chunks as u64 {
                let global_elapsed = training_start.elapsed().as_secs_f64();
                let chunks_done_global = (epoch as u64) * (total_chunks as u64) + n_chunks as u64;
                let eta_secs = if chunks_done_global > 0 {
                    let secs_per = global_elapsed / chunks_done_global as f64;
                    secs_per * (total_chunks_all - chunks_done_global) as f64
                } else {
                    f64::NAN
                };

                let batch_avg = if win_n > 0 {
                    win_total / win_n as f32
                } else {
                    f32::NAN
                };

                on_progress(TrainingProgress {
                    epoch: (epoch + 1) as u64,
                    total_epochs: epochs as u64,
                    chunk: n_chunks,
                    total_chunks: total_chunks as u64,
                    loss: loss as f64,
                    eta_secs,
                    epoch_done: false,
                    // Running stats so far this epoch (n_chunks >= 1 here), so the
                    // UI shows a live average instead of a frozen 0.
                    avg_loss: total_loss / n_chunks as f32,
                    min_loss,
                    max_loss,
                    // Spread over just the chunks in this window.
                    batch_min: win_min,
                    batch_avg,
                    batch_max: win_max,
                    epoch_duration_secs: 0.0,
                    checkpoint_path: None,
                });

                // Start a fresh window for the next report.
                win_total = 0.0;
                win_n = 0;
                win_min = f32::INFINITY;
                win_max = f32::NEG_INFINITY;
            }
        }

        let epoch_duration = epoch_start.elapsed().as_secs_f64();
        let avg_loss = total_loss / n_chunks as f32;
        println!(
            "Epoch {}: avg loss = {:.6} (min={:.4} max={:.4})",
            epoch + 1,
            avg_loss,
            min_loss,
            max_loss
        );

        let global_elapsed = training_start.elapsed().as_secs_f64();
        let chunks_done_global = ((epoch + 1) as u64) * (total_chunks as u64);
        let eta_secs = if chunks_done_global > 0 {
            let secs_per = global_elapsed / chunks_done_global as f64;
            secs_per * (total_chunks_all - chunks_done_global) as f64
        } else {
            f64::NAN
        };

        sync_weights_from_gpu(ws, transformer);

        // Save checkpoint if a directory is configured
        let checkpoint_path = checkpoints_dir.map(|cp_dir| {
            let cp_path = cp_dir.join(format!("epoch_{:04}.bin", epoch + 1));
            let cp_str = cp_path.to_string_lossy().to_string();
            transformer.save_to_path(&cp_str).ok();
            cp_str
        });

        // Save to model dir for backward compat / final model access
        let _ = transformer.save(&model_dir.to_string_lossy().to_string());

        println!("Checkpoint saved");

        on_progress(TrainingProgress {
            epoch: (epoch + 1) as u64,
            total_epochs: epochs as u64,
            chunk: total_chunks as u64,
            total_chunks: total_chunks as u64,
            loss: avg_loss as f64,
            eta_secs,
            epoch_done: true,
            avg_loss,
            min_loss,
            max_loss,
            // No window at an epoch boundary; NaN serializes to null and the
            // frontend skips it for the batch chart.
            batch_min: f32::NAN,
            batch_avg: f32::NAN,
            batch_max: f32::NAN,
            epoch_duration_secs: epoch_duration,
            checkpoint_path,
        });

        if let Some(ref cancel) = cancel {
            if cancel.load(Ordering::SeqCst) {
                println!("Training cancelled by user at epoch {}", epoch + 1);
                return Ok(true);
            }
        }

        if avg_loss < best_loss {
            best_loss = avg_loss;
            epochs_since_best = 0;
        } else {
            epochs_since_best += 1;
        }
        if epochs_since_best >= patience {
            println!(
                "Epoch {}: early stopping (no improvement for {} epochs)",
                epoch + 1,
                patience
            );
            break;
        }
    }

    Ok(false)
}
