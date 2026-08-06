use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::training::gpu_workspace::Workspace;
use crate::training::transformer::Model;
use crate::training::optimizer::{gpu_adam_update, gpu_clip_gradients, gpu_grad_norms, read_grad_norm};
use crate::training::sync::sync_weights_from_gpu;
use crate::training::diagnostics::{self, Diagnostics};

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
// Linear LR warmup, sized per run by `warmup_steps`. The opening steps on a fresh
// (especially deep) model are the most unstable, so ramping lr up from ~0 avoids
// the early divergence that clipping alone only slows down. After warmup the lr
// cosine-decays toward MIN_LR_RATIO * lr over the rest of the run (see
// `lr_at_step`) — the classic GPT schedule: warmup steadies the opening, decay
// lets the loss settle late instead of skating on a flat lr.
//
// The length is a FRACTION of the run's step budget, capped and floored. It used
// to be a flat 100 steps, which is fine for a pretrain of thousands of steps but
// wrong for SFT: `adam_step` restarts at 0 on every run, and a fine-tune's whole
// budget is a few hundred steps (~98/epoch on the current chat set at ctx 512),
// so 100 steps was 21% of a 5-epoch run and 34% of a 3-epoch one. The lr did not
// reach the configured value until the end of epoch 1 — the pass over the data
// where it matters most — and the cosine decay was compressed into the tail.
// Adding epochs only diluted the fixed 100, which is why undertrained SFT looked
// like it wanted more epochs.
//
// Mean lr over a run barely moves (~0.54 → 0.55 × base at 5 epochs): warmup's
// lost area goes to the cosine, which averages about the same. What changes is
// *when* the lr is high.
const WARMUP_RATIO: f32 = 0.05;
// Cap. Keeps every pretrain identical to the pre-ratio behaviour — at thousands
// of steps 5% would be 500+, and the pretraining schedule is tuned. The ratio
// therefore only ever *shortens* warmup, on the short runs where it was wrong.
const WARMUP_STEPS: i32 = 100;
// Floor, so a very short run still gets a real ramp rather than 2 steps. Warmup
// is not pointless on SFT even from a good base: Adam's moments start at zero and
// `warm_start_embeddings` may have just reset marker rows to the table mean.
const WARMUP_MIN_STEPS: i32 = 10;
// Cosine-decay floor as a fraction of the configured lr: after warmup the lr
// decays from `lr` down to `MIN_LR_RATIO * lr` by the final optimizer step. 0.1
// is the common GPT default — decaying all the way to 0 tends to waste the last
// stretch of training.
const MIN_LR_RATIO: f32 = 0.1;
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
    // Curriculum stage this frame belongs to (0-based) and its optional label, so
    // the UI can mark stage transitions. A non-curriculum run is a single stage 0.
    pub stage: u64,
    pub stage_name: Option<String>,
    // Chunks completed across the WHOLE run (all stages), a monotonically
    // increasing x-axis for the charts — `chunk` alone resets each epoch and
    // `total_chunks` differs between stages.
    pub global_step: u64,
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
    // Per-report training telemetry (grad norms, update:weight ratios, activation
    // RMS, attention entropy, embedding anisotropy). Present only on mid-epoch
    // report frames — `None` on epoch-boundary frames, which carry no fresh caches.
    pub diagnostics: Option<Diagnostics>,
}

pub type ProgressCallback = Box<dyn Fn(TrainingProgress) + Send>;

/// A contiguous span of the on-disk corpus that forms one training document.
/// `start`/`len` index the flat token stream, exclusive of the `<|endoftext|>`
/// separators baked between documents.
pub struct DocSpan {
    pub start: usize,
    pub len: usize,
}

/// One curriculum stage: a set of documents trained for `epochs` epochs before
/// the run moves on to the next stage.
pub struct Stage {
    pub name: Option<String>,
    pub epochs: i64,
    pub docs: Vec<DocSpan>,
}

/// How to order and segment the corpus for a run. Every training entry point
/// funnels through this: pretraining passes a real multi-stage plan; SFT and
/// resume pass a single-stage, whole-corpus, unshuffled plan (`whole_corpus`).
pub struct DataPlan {
    pub stages: Vec<Stage>,
    /// Token id inserted between documents when they're reordered.
    pub eot_id: u32,
    /// Reshuffle document order every epoch.
    pub shuffle: bool,
    /// Seed for the per-epoch shuffle (reproducible).
    pub seed: u64,
}

impl DataPlan {
    /// A trivial plan: one stage of `epochs` epochs over the whole corpus as a
    /// single document, no shuffling. Used by resume/SFT and as the fallback for
    /// datasets built before document boundaries were recorded. `start`/`len` are
    /// filled in against the actual corpus length inside `train`/the loop.
    pub fn whole_corpus(epochs: i64) -> DataPlan {
        DataPlan {
            stages: vec![Stage { name: None, epochs, docs: Vec::new() }],
            eot_id: 0,
            shuffle: false,
            seed: 0,
        }
    }

    /// True when this is the single-stage whole-corpus fallback (its one stage
    /// carries no explicit document spans), so the loop can synthesize a single
    /// document covering the entire corpus.
    fn is_whole_corpus(&self) -> bool {
        self.stages.len() == 1 && self.stages[0].docs.is_empty()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn train(
    dimensions: i64,
    ctx_window: i64,
    heads: i64,
    layers: i64,
    lr: f32,
    ff_multiplier: i64,
    model_dir: &Path,
    corpus_path: &Path,
    merges_path: &Path,
    checkpoints_dir: Option<&Path>,
    plan: DataPlan,
    on_progress: ProgressCallback,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<bool, io::Error> {
    let corpus = tokenize::bpe::load_corpus(&corpus_path.to_string_lossy())?;

    let merges = tokenize::bpe::load_merges(&merges_path.to_string_lossy())?;
    let vocab = tokenize::bpe::vocab_size(merges.len());

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
        &plan,
        ctx_window,
        layers,
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
    let ctx_window = transformer.max_seq_len;
    let layers = transformer.layers;
    let mut ws = Workspace::new(&transformer, ctx_window as usize);

    let corpus = tokenize::bpe::load_corpus(&corpus_path.to_string_lossy())?;

    if let Some(cp_dir) = checkpoints_dir {
        fs::create_dir_all(cp_dir)?;
    }

    // Resume trains over the whole corpus in fixed order (per-epoch shuffle for
    // the resume path is a follow-up; the fresh `train` path shuffles).
    let plan = DataPlan::whole_corpus(additional_epochs);
    run_training_loop(
        &mut transformer,
        &mut ws,
        &corpus,
        None,
        &plan,
        ctx_window,
        layers,
        lr,
        model_dir,
        checkpoints_dir,
        on_progress,
        cancel,
    )
}

/// Warm-start the input embeddings of `ids` before SFT, returning how many rows
/// were reset.
///
/// A marker absent from the pretrain corpus reaches SFT with its embedding still
/// at random Glorot init, and injects noise into the shared trunk the instant SFT
/// feeds it — at the usual SFT LR, wrecking the pretrained weights (the "SFT turns
/// coherent output into gibberish" failure). Resetting the row to the mean
/// embedding (a "typical token" vector) is the standard added-token
/// initialization, so the fine-tune starts from a sane point instead of noise.
///
/// **The caller supplies the ids and owns the policy.** This deliberately does not
/// decide which rows are untrained, because that is not decidable here. An earlier
/// version tried to *detect* untrained rows by "all components within the init
/// bound `sqrt(6/(d+vocab))`", but on a small model that bound is ~0.03 and around
/// half of all legitimately-trained-but-small embeddings (rare bytes, infrequent
/// merges, lightly-used markup) never cross it — so it reset ~half the vocabulary
/// onto one mean vector and *caused* the gibberish it was meant to prevent. A
/// weight value does not carry its own history; only a record made at pretrain
/// time does (`DatasetStats::special_ids_seen`), and that record lives in the
/// backend, above this crate.
///
/// Ids at or past the vocabulary are skipped rather than rejected: a mismatched
/// record should not fail a run that is otherwise fine.
///
/// Once the pretrain corpus contains the chat markers, the correct outcome here is
/// **zero** — the caller logs that, so a non-zero count is a signal that the
/// corpora drifted or a marker was added, not routine noise.
pub fn warm_start_embeddings(model: &mut Model, ids: &[u32]) -> usize {
    let vocab = model.embedding_table.len();
    let d = model.embedding_table.first().map_or(0, |r| r.len());
    if vocab == 0 || d == 0 || ids.is_empty() {
        return 0;
    }

    // Mean over the whole table — a stable "typical token" vector to seed from.
    let mut mean = vec![0.0f32; d];
    for row in &model.embedding_table {
        for (m, &w) in mean.iter_mut().zip(row) {
            *m += w;
        }
    }
    for m in &mut mean {
        *m /= vocab as f32;
    }

    let mut reset = 0usize;
    for &id in ids {
        if (id as usize) < vocab {
            model.embedding_table[id as usize].clone_from(&mean);
            reset += 1;
        }
    }
    reset
}

/// Warm-start the **output** (unembedding) columns of `ids` to the mean column,
/// returning how many were reset. The counterpart of `warm_start_embeddings` for
/// the other end of the network.
///
/// `w_output` is `[d][vocab]` — a token owns a *column*, not a row — so this walks
/// every dimension writing one element per id.
///
/// This is the half that matters at a curriculum stage boundary. An id absent from
/// the stages before it was never an *input*, so its embedding row is untouched
/// random init; but it was a target on every step, always with label 0, so softmax
/// spends the whole run pushing its logit down. By the time the stage that finally
/// uses it starts, the id costs ~20 nats the first time it is the answer, and the
/// gradient that produces is the loss spike. Seeding both ends from the table mean
/// makes a newly-introduced token merely *unknown* (~ln(vocab) nats) instead of
/// actively *refuted*.
pub fn warm_start_output_columns(model: &mut Model, ids: &[u32]) -> usize {
    let d = model.w_output.len();
    let vocab = model.w_output.first().map_or(0, |r| r.len());
    if vocab == 0 || d == 0 || ids.is_empty() {
        return 0;
    }

    // Mean over the vocabulary, per dimension — the column an "average token" owns.
    let mean: Vec<f32> = model
        .w_output
        .iter()
        .map(|row| row.iter().sum::<f32>() / vocab as f32)
        .collect();

    let mut reset = 0usize;
    for &id in ids {
        let col = id as usize;
        if col < vocab {
            for (row, &m) in model.w_output.iter_mut().zip(&mean) {
                row[col] = m;
            }
            reset += 1;
        }
    }
    reset
}

/// The ids `docs` contains that `seen` has not recorded yet, marking all of them
/// seen on the way through. `seen` is indexed by token id and persists across the
/// call sequence, so calling this once per stage in curriculum order yields, each
/// time, exactly the ids that stage *introduces*.
fn ids_introduced_by(corpus: &[u16], docs: &[DocSpan], seen: &mut [bool]) -> Vec<u32> {
    let mut introduced = Vec::new();
    for doc in docs {
        let end = (doc.start + doc.len).min(corpus.len());
        for &id in &corpus[doc.start.min(end)..end] {
            match seen.get_mut(id as usize) {
                Some(flag) if !*flag => {
                    *flag = true;
                    introduced.push(id as u32);
                }
                _ => {}
            }
        }
    }
    introduced.sort_unstable();
    introduced
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
/// `lr` and the plan's epoch count are caller-supplied on purpose: SFT typically
/// wants a *lower* LR and *fewer* epochs than pretraining (a base model is
/// already near a good minimum). Good starting points are ~1e-4 to 5e-5 for a
/// handful of epochs — an order of magnitude below a from-scratch run. Returns
/// `Ok(true)` if training was cancelled, `Ok(false)` if it ran to completion.
///
/// `plan` carries one document per *conversation* when the dataset builder
/// recorded them, so the conversations are reordered every epoch exactly the way
/// pretraining reorders its source documents. A dataset built without those spans
/// (or a caller with nothing to say about order) passes `DataPlan::whole_corpus`
/// and gets the old fixed-order pass.
#[allow(clippy::too_many_arguments)]
pub fn sft_train(
    base_model_dir: &Path,
    sft_corpus_path: &Path,
    sft_mask_path: &Path,
    out_model_dir: &Path,
    plan: DataPlan,
    // Embedding rows to warm-start before the first step — the markers this
    // dataset uses that the base model's pretrain corpus never contained. Empty is
    // the healthy case once the two corpora share a marker vocabulary; see
    // `warm_start_embeddings` for why the caller decides this and not us.
    warm_start_ids: &[u32],
    lr: f32,
    checkpoints_dir: Option<&Path>,
    on_progress: ProgressCallback,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<bool, io::Error> {
    let mut transformer = Model::load(&base_model_dir.to_string_lossy())?;
    let ctx_window = transformer.max_seq_len;
    let layers = transformer.layers;

    let corpus = tokenize::bpe::load_corpus(&sft_corpus_path.to_string_lossy())?;
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

    // Warm-start the untrained marker embeddings BEFORE the workspace uploads
    // weights to the GPU — after that the rows are on the device and a CPU-side
    // edit would be silently ignored.
    let reset = warm_start_embeddings(&mut transformer, warm_start_ids);
    if reset > 0 {
        println!(
            "SFT warm-start: reset {reset} input embedding(s) to the table mean — \
             this dataset uses markers the base model's pretrain corpus never contained"
        );
    }

    let mut ws = Workspace::new(&transformer, ctx_window as usize);

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
        &plan,
        ctx_window,
        layers,
        lr,
        out_model_dir,
        checkpoints_dir,
        on_progress,
        cancel,
    )
}

/// Preference fine-tuning (KTO) of an existing model on thumbs-rated replies.
///
/// The third training mode, and the closest sibling of `sft_train`: same two-phase
/// shape, same base-model loading, same tokenizer copy, same checkpointing. Two
/// things differ.
///
/// **The unit of training is an example, not a chunk.** Each rated reply is one
/// span, trained whole, because KTO's reward is a property of a complete sequence —
/// slicing a reply across two context windows would compute a reward for half a
/// reply and compare it against a reference logprob for the whole thing. The
/// dataset builder guarantees every span fits the context window.
///
/// **The gradient is scaled per example.** See `crate::kto`; the short version is
/// that a KTO gradient is the ordinary cross-entropy gradient times a scalar, so
/// nothing about the backward pass changes except its magnitude and, for a rejected
/// reply, its sign.
///
/// Unlike pretrain and SFT, there is **no early stopping** on average loss. KTO's
/// loss is bounded and saturating, so it flattens out while the model is still
/// moving; "loss stopped improving" does not mean the same thing here, and the
/// signal worth watching is the reward margin the benchmark reports instead.
#[allow(clippy::too_many_arguments)]
pub fn kto_train(
    base_model_dir: &Path,
    corpus_path: &Path,
    mask_path: &Path,
    index_path: &Path,
    out_model_dir: &Path,
    epochs: i64,
    // As `sft_train`'s: the markers this dataset uses that the base never trained.
    // A KTO run on top of an SFT model normally supplies nothing, since the SFT
    // trained them; one started from a bare pretrain still meets them cold.
    warm_start_ids: &[u32],
    lr: f32,
    beta: f64,
    checkpoints_dir: Option<&Path>,
    on_progress: ProgressCallback,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<bool, io::Error> {
    let transformer_owned = Model::load(&base_model_dir.to_string_lossy())?;
    let mut transformer = transformer_owned;
    let ctx_window = transformer.max_seq_len;
    let layers = transformer.layers;

    let corpus = tokenize::bpe::load_corpus(&corpus_path.to_string_lossy())?;
    let mask: Vec<u8> = fs::read(mask_path)?;
    if corpus.len() != mask.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "KTO corpus/mask length mismatch: {} tokens vs {} mask bytes",
                corpus.len(),
                mask.len()
            ),
        ));
    }

    let index: crate::kto::KtoIndex = serde_json::from_str(&fs::read_to_string(index_path)?)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad KTO index: {e}")))?;
    if index.examples.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "KTO dataset holds no examples",
        ));
    }
    // Every span must land inside the corpus and inside one context window; a bad
    // index would otherwise slice out of bounds or silently train a truncated reply
    // against a whole-reply reference logprob.
    for example in &index.examples {
        let end = example.offset + example.len;
        if end > corpus.len() || example.len < 2 || example.len > ctx_window as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "KTO example span {}..{} is invalid for a {}-token corpus and {}-token context",
                    example.offset, end, corpus.len(), ctx_window
                ),
            ));
        }
    }

    // Balanced from the dataset's actual composition rather than configured:
    // whether the thumbs skew up or down is a property of the data, not a knob
    // anyone should have to set by hand.
    let params = crate::kto::KtoParams::balanced(beta, index.desirable(), index.undesirable());
    println!(
        "KTO: {} examples ({} up / {} down), beta={:.3}, lambda={:.3}/{:.3}",
        index.examples.len(),
        index.desirable(),
        index.undesirable(),
        params.beta,
        params.lambda_desirable,
        params.lambda_undesirable,
    );

    let reset = warm_start_embeddings(&mut transformer, warm_start_ids);
    if reset > 0 {
        println!(
            "KTO warm-start: reset {reset} input embedding(s) to the table mean — \
             this dataset uses markers the base model's corpus never contained"
        );
    }

    let mut ws = Workspace::new(&transformer, ctx_window as usize);
    ws.ensure_layer_caches(layers as usize);

    fs::create_dir_all(out_model_dir)?;
    let base_merges = base_model_dir.join("merges.bin");
    if base_merges.is_file() {
        fs::copy(&base_merges, out_model_dir.join("merges.bin"))?;
    }
    if let Some(cp_dir) = checkpoints_dir {
        fs::create_dir_all(cp_dir)?;
    }

    let total_examples = index.examples.len();
    let total_all = (total_examples as u64) * epochs.max(0) as u64;
    let total_opt_steps = (total_examples.div_ceil(ACCUM_STEPS) as i32) * epochs.max(0) as i32;
    let training_start = Instant::now();

    // The KL baseline persists across epochs: it is an estimate of how far the
    // policy has drifted from the reference overall, and resetting it each epoch
    // would keep yanking the reference point back to zero.
    let mut kl = crate::kto::KlEstimate::new();
    let mut adam_step: i32 = 0;
    let mut done_global: u64 = 0;

    for epoch in 0..epochs.max(0) {
        let mut order: Vec<usize> = (0..total_examples).collect();
        // Shuffled every epoch: the export is ordered by time, so consecutive
        // examples come from the same conversation and often carry the same label.
        // Training them in that order makes each accumulation window one-sided.
        shuffle_seeded(&mut order, 0x4B54_4F00u64.wrapping_add(epoch as u64));

        let mut epoch_stats = LossStats::new();
        let mut window = LossStats::new();
        let epoch_start = Instant::now();
        let mut reward_total = 0.0f64;
        let mut trained = 0u64;
        ws.zero_accumulators();

        for idx in order {
            let example = index.examples[idx];
            // Widened per example for the same reason `train_epoch` widens per
            // chunk: the corpus is u16 in memory, the step APIs speak u32.
            let span: Vec<u32> = corpus[example.offset..example.offset + example.len]
                .iter()
                .map(|&t| t as u32)
                .collect();
            let mask_span = &mask[example.offset..example.offset + example.len];

            let step = pollster::block_on(transformer.train_step_kto(
                &mut ws,
                &span,
                mask_span,
                layers as usize,
                example.ref_logprob,
                example.desirable,
                kl.value(),
                &params,
            ));
            // No reply tokens, or a diverged forward: nothing was accumulated, so
            // the example is skipped rather than counted as a zero-loss step.
            let Some(step) = step else {
                continue;
            };

            kl.record(step.logprob, example.ref_logprob);
            reward_total += step.reward;
            epoch_stats.record(step.loss as f32);
            window.record(step.loss as f32);
            trained += 1;
            done_global += 1;

            let step_due = trained % ACCUM_STEPS as u64 == 0 || trained == total_examples as u64;
            let report_due =
                trained % PROGRESS_REPORT_INTERVAL == 0 || trained == total_examples as u64;

            let grad_norms = if report_due {
                Some(gpu_grad_norms(&ws, layers as usize))
            } else {
                None
            };
            if step_due {
                step_optimizer(&mut ws, layers, lr, total_opt_steps, &mut adam_step);
            }
            if report_due {
                let step_lr = lr_at_step(lr, adam_step, total_opt_steps);
                let diag = grad_norms.map(|g| {
                    diagnostics::compute(
                        &ws, g, layers as usize, span.len(), step_lr, adam_step, &[],
                    )
                });
                on_progress(TrainingProgress {
                    epoch: epoch as u64,
                    total_epochs: epochs.max(0) as u64,
                    chunk: trained,
                    total_chunks: total_examples as u64,
                    stage: 0,
                    // The mean implicit reward is the number that actually says
                    // whether the tune is working, and KTO's loss does not — so it
                    // rides along on the stage label, which is already displayed.
                    stage_name: Some(format!(
                        "KTO · mean reward {:+.4}",
                        reward_total / trained as f64
                    )),
                    global_step: done_global,
                    loss: step.loss,
                    eta_secs: compute_eta(training_start, done_global, total_all),
                    epoch_done: false,
                    avg_loss: epoch_stats.avg(),
                    min_loss: epoch_stats.min,
                    max_loss: epoch_stats.max,
                    batch_min: window.min,
                    batch_avg: window.avg(),
                    batch_max: window.max,
                    epoch_duration_secs: epoch_start.elapsed().as_secs_f64(),
                    checkpoint_path: None,
                    diagnostics: diag,
                });
                window = LossStats::new();
            }
        }

        let checkpoint_path =
            save_checkpoint(&mut transformer, &ws, out_model_dir, checkpoints_dir, epoch);
        let mean_reward = if trained == 0 { f64::NAN } else { reward_total / trained as f64 };
        println!(
            "KTO epoch {}/{}: avg loss = {:.6}, mean reward = {:+.4} ({} examples)",
            epoch + 1,
            epochs.max(0),
            epoch_stats.avg(),
            mean_reward,
            trained,
        );
        on_progress(TrainingProgress {
            epoch: epoch as u64,
            total_epochs: epochs.max(0) as u64,
            chunk: trained,
            total_chunks: total_examples as u64,
            stage: 0,
            stage_name: Some(format!("KTO · mean reward {mean_reward:+.4}")),
            global_step: done_global,
            loss: epoch_stats.avg() as f64,
            eta_secs: compute_eta(training_start, done_global, total_all),
            epoch_done: true,
            avg_loss: epoch_stats.avg(),
            min_loss: epoch_stats.min,
            max_loss: epoch_stats.max,
            batch_min: f32::NAN,
            batch_avg: f32::NAN,
            batch_max: f32::NAN,
            epoch_duration_secs: epoch_start.elapsed().as_secs_f64(),
            checkpoint_path,
            diagnostics: None,
        });

        if let Some(ref cancel) = cancel {
            if cancel.load(Ordering::SeqCst) {
                println!("KTO cancelled by user at epoch {}", epoch + 1);
                return Ok(true);
            }
        }
    }

    sync_weights_from_gpu(&ws, &mut transformer);
    transformer.save(&out_model_dir.to_string_lossy())?;
    Ok(false)
}

struct EpochStats {
    avg_loss: f32,
    min_loss: f32,
    max_loss: f32,
    duration_secs: f64,
    // Chunks actually trained this epoch (== the stage's chunk count), used to
    // advance the run-global chunk counter for the chart x-axis and ETA.
    chunks: usize,
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
    corpus: &[u16],
    // SFT loss mask aligned 1:1 with `corpus`, or `None` for plain pretraining.
    // When present, each chunk trains only on its masked-in (assistant) tokens.
    mask: Option<&[u8]>,
    ctx_window: i64,
    layers: i64,
    lr: f32,
    // The run's whole planned optimizer-step budget, for the cosine-decay curve.
    total_opt_steps: i32,
    // Global (whole-run) epoch index and total, for progress display.
    epoch: i64,
    epochs: i64,
    // Curriculum stage this epoch belongs to and its label, forwarded on frames.
    stage: u64,
    stage_name: Option<&str>,
    // Chunks in THIS epoch's stream (this stage), and the whole-run chunk total.
    total_chunks: usize,
    total_chunks_all: u64,
    // Chunks completed before this epoch, so a monotonic global step can be
    // reported across stages/epochs.
    chunks_done_base: u64,
    training_start: Instant,
    adam_step: &mut i32,
    // The corpus's most-frequent token ids, for the embedding-anisotropy metric.
    // Computed once per run and reused across epochs so the metric tracks a fixed
    // token set.
    freq_tokens: &[u32],
    on_progress: &ProgressCallback,
) -> EpochStats {
    let mut epoch_stats = LossStats::new();
    let mut window = LossStats::new();
    let epoch_start = Instant::now();

    ws.zero_accumulators();

    // The corpus is held as u16 (half the resident bytes of a multi-hundred-
    // million-token run), but every step API speaks u32. Widening one chunk at a
    // time into a buffer reused across the epoch keeps the saving and costs one
    // pass over ~ctx_window elements per step, against a full forward and backward.
    let mut chunk_ids: Vec<u32> = Vec::with_capacity(ctx_window as usize);

    for (chunk_idx, chunk) in corpus.chunks(ctx_window as usize).enumerate() {
        if chunk.len() < 2 {
            continue;
        }
        chunk_ids.clear();
        chunk_ids.extend(chunk.iter().map(|&t| t as u32));
        let chunk = chunk_ids.as_slice();
        // Slice the mask with the SAME boundaries as the corpus chunks. Chunks
        // are contiguous and fixed-size (only the last may be shorter), so the
        // chunk's flat offset is `chunk_idx * ctx_window`.
        let loss = match mask {
            Some(m) => {
                let start = chunk_idx * ctx_window as usize;
                let mask_slice = &m[start..start + chunk.len()];
                // Every position in this window is context — a long pasted user
                // turn fills many such windows back to back. `train_step_masked`
                // would run a full forward and backward to arrive at a loss of
                // exactly 0.0, and because 0.0 is finite it would then be folded
                // into the epoch average (dragging it toward zero) and counted
                // toward the accumulation window (stepping Adam on nothing but
                // leftover momentum). `train_step_kto` already skips this case for
                // the same reason; this is the SFT side of that guard.
                if !mask_slice[..mask_slice.len() - 1].iter().any(|&b| b != 0) {
                    continue;
                }
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

        let step_due =
            epoch_stats.n % (ACCUM_STEPS as u64) == 0 || epoch_stats.n == total_chunks as u64;
        let report_due =
            epoch_stats.n % PROGRESS_REPORT_INTERVAL == 0 || epoch_stats.n == total_chunks as u64;

        // Sample diagnostics coherently with the optimizer step. The gradient
        // norms must be read from the accumulated buffers BEFORE the step zeroes
        // them; the update:weight ratio and the activation/attention/embedding
        // metrics are read AFTER, from the post-step moments and the still-valid
        // forward caches. `report_due` implies `step_due` (the report interval is
        // a multiple of the accumulation window, and the epoch tail triggers
        // both), so a report always straddles a real step and the grad norms are
        // never a just-zeroed reading.
        let grad_norms = if report_due {
            Some(gpu_grad_norms(ws, layers as usize))
        } else {
            None
        };

        if step_due {
            step_optimizer(ws, layers, lr, total_opt_steps, adam_step);
        }

        if report_due {
            let step_lr = lr_at_step(lr, *adam_step, total_opt_steps);
            let diag = grad_norms.map(|g| {
                diagnostics::compute(
                    ws, g, layers as usize, chunk.len(), step_lr, *adam_step, freq_tokens,
                )
            });
            report_chunk_progress(
                epoch, epochs, stage, stage_name, total_chunks, chunks_done_base,
                total_chunks_all, training_start, loss, &epoch_stats, &window, diag, on_progress,
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
        chunks: epoch_stats.n as usize,
    }
}

// Warmup length for a run of `total_steps` optimizer steps: WARMUP_RATIO of the
// budget, clamped to [WARMUP_MIN_STEPS, WARMUP_STEPS]. See the constants for why
// this is a fraction rather than the flat 100 it used to be.
fn warmup_steps(total_steps: i32) -> i32 {
    let scaled = (total_steps.max(0) as f32 * WARMUP_RATIO) as i32;
    scaled.clamp(WARMUP_MIN_STEPS, WARMUP_STEPS)
}

// Learning-rate schedule: linear warmup over the first `warmup_steps(total_steps)`
// optimizer steps, then cosine decay from `base_lr` down to `MIN_LR_RATIO *
// base_lr` across the remaining steps. `step` is the 1-based optimizer-step count;
// `total_steps` is the run's whole planned optimizer-step budget.
fn lr_at_step(base_lr: f32, step: i32, total_steps: i32) -> f32 {
    let warmup = warmup_steps(total_steps);
    if step <= warmup {
        // Linear ramp from ~0 up to base_lr (base_lr itself at step == warmup).
        return base_lr * (step as f32 / warmup as f32).min(1.0);
    }
    let min_lr = base_lr * MIN_LR_RATIO;
    // Fraction of the post-warmup span already travelled. Clamp to [0, 1] and
    // floor the denominator at 1 so a run shorter than the warmup floor (a
    // handful of steps) can't divide by zero or produce a negative/out-of-range
    // lr — it just pins to min_lr past warmup instead of going haywire.
    let denom = (total_steps - warmup).max(1) as f32;
    let progress = ((step - warmup) as f32 / denom).clamp(0.0, 1.0);
    min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
}

// Apply one accumulated optimizer step: clip the summed gradient, take a
// scheduled (warmup + cosine-decay) Adam step, then zero the accumulators for
// the next window. `total_opt_steps` is the run's whole planned step budget,
// used only to place `adam_step` on the decay curve.
fn step_optimizer(ws: &mut Workspace, layers: i64, lr: f32, total_opt_steps: i32, adam_step: &mut i32) {
    *adam_step += 1;
    // Clip the accumulated gradient BEFORE the step so a rare huge gradient
    // can't blow the model up (the loss-spike-to-50 case).
    let clip = MAX_GRAD_NORM * ACCUM_STEPS as f32;
    gpu_clip_gradients(ws, layers as usize, clip);
    // Scheduled lr. Adam averages the gradient internally (grad_scale), so pass
    // the full scheduled lr, not lr/ACCUM_STEPS.
    let step_lr = lr_at_step(lr, *adam_step, total_opt_steps);
    gpu_adam_update(ws, step_lr, *adam_step, ACCUM_STEPS, layers as usize);
    ws.zero_accumulators();
    // Periodic gradient-norm readout so a divergent run is visible: a norm that
    // keeps growing = trouble.
    //
    // Reading the norm synchronizes, so this samples every 50th optimizer step
    // instead of testing it on every one. The old code checked `gnorm` each step
    // to log spikes immediately, which meant the value had to come back to the
    // host every step — the exact stall the device-side clip removes. Nothing is
    // lost numerically: the clip is applied unconditionally on the device
    // whether or not anyone looks at the norm, and the per-report diagnostics
    // still carry per-layer gradient norms.
    if *adam_step % 50 == 0 {
        let gnorm = read_grad_norm(ws);
        let spike = !gnorm.is_finite() || gnorm > clip * 20.0;
        eprintln!(
            "  step {} grad_norm {:.2}{}",
            *adam_step,
            gnorm,
            if spike { "  ⚠ spike (clipped)" } else { "" }
        );
    }
}

// Emit a mid-epoch progress frame: epoch-cumulative avg/min/max plus the batch
// spread over just this report window. The caller resets `window` afterward.
#[allow(clippy::too_many_arguments)] // cohesive progress-frame builder; the fields belong together
fn report_chunk_progress(
    epoch: i64,
    epochs: i64,
    stage: u64,
    stage_name: Option<&str>,
    total_chunks: usize,
    chunks_done_base: u64,
    total_chunks_all: u64,
    training_start: Instant,
    loss: f32,
    epoch_stats: &LossStats,
    window: &LossStats,
    diagnostics: Option<Diagnostics>,
    on_progress: &ProgressCallback,
) {
    let chunks_done_global = chunks_done_base + epoch_stats.n;
    on_progress(TrainingProgress {
        epoch: (epoch + 1) as u64,
        total_epochs: epochs as u64,
        chunk: epoch_stats.n,
        total_chunks: total_chunks as u64,
        stage,
        stage_name: stage_name.map(str::to_string),
        global_step: chunks_done_global,
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
        diagnostics,
    });
}

// Number of training chunks a token stream of length `len` yields under
// `corpus.chunks(ctx)` filtered to `len >= 2` — the last chunk is skipped when
// it holds a single token (nothing to predict). Order-independent, so it can be
// precomputed per stage for the step/ETA budget.
fn chunks_in(len: usize, ctx: usize) -> usize {
    if len < 2 {
        return 0;
    }
    let full = len / ctx;
    let rem = len % ctx;
    full + usize::from(rem >= 2)
}

// Of those chunks, how many actually train something.
//
// `chunks_in` counts forward passes; this counts the ones that produce a gradient.
// For SFT the two diverge sharply: a conversation carrying a long pasted user turn
// fills whole context windows with nothing but context, and `loss_value` returns
// `None` for those — a loss of exactly zero, and no gradient. Budgeting the LR
// curve and the progress bar off `chunks_in` would pace the run against work that
// does not happen.
//
// The `[..len - 1]` mirrors `upload_loss_targets`: mask entry `i` governs position
// `i`, which predicts token `i + 1`, so the last token's mask byte is never read.
fn trainable_chunks_in(mask: &[u8], ctx: usize) -> usize {
    mask.chunks(ctx)
        .filter(|c| c.len() >= 2 && c[..c.len() - 1].iter().any(|&b| b != 0))
        .count()
}

// Length of a stage's assembled token stream: every document's tokens plus one
// <|endoftext|> separator between adjacent documents.
fn stage_stream_len(docs: &[DocSpan]) -> usize {
    docs.iter().map(|d| d.len).sum::<usize>() + docs.len().saturating_sub(1)
}

// Fisher–Yates shuffle of document indices from an explicit seed, so each epoch's
// order is reproducible from `seed`. `| 1` keeps xorshift out of its zero state.
fn shuffle_seeded(order: &mut [usize], seed: u64) {
    let mut state = seed | 1;
    for i in (1..order.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

// Assemble one epoch's token stream (and aligned mask, for SFT) by walking the
// documents in `order`, concatenating each document's corpus slice with an
// `eot_id` (mask byte 0) inserted between adjacent documents. The default order
// reproduces the on-disk corpus exactly, since build baked the same separators
// in the same places.
fn assemble_epoch(
    corpus: &[u16],
    mask: Option<&[u8]>,
    docs: &[DocSpan],
    order: &[usize],
    eot_id: u32,
) -> (Vec<u16>, Option<Vec<u8>>) {
    let total = stage_stream_len(docs);
    let mut out = Vec::with_capacity(total);
    let mut out_mask = mask.map(|_| Vec::with_capacity(total));
    for (i, &di) in order.iter().enumerate() {
        let d = &docs[di];
        if i > 0 {
            // The corpus is u16 on disk and in memory; `eot_id` is carried as u32
            // because that is the width every token API speaks. It is a reserved
            // id far below the ceiling, so the narrowing is total.
            out.push(eot_id as u16);
            if let Some(m) = out_mask.as_mut() {
                m.push(0);
            }
        }
        out.extend_from_slice(&corpus[d.start..d.start + d.len]);
        if let (Some(m), Some(src)) = (out_mask.as_mut(), mask) {
            m.extend_from_slice(&src[d.start..d.start + d.len]);
        }
    }
    (out, out_mask)
}

#[allow(clippy::too_many_arguments)]
fn run_training_loop(
    transformer: &mut Model,
    ws: &mut Workspace,
    corpus: &[u16],
    // SFT loss mask aligned 1:1 with `corpus`, threaded to `train_epoch`; `None`
    // for pretraining (`train`/`continue_train`).
    mask: Option<&[u8]>,
    plan: &DataPlan,
    ctx_window: i64,
    layers: i64,
    lr: f32,
    model_dir: &Path,
    checkpoints_dir: Option<&Path>,
    on_progress: ProgressCallback,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<bool, io::Error> {
    ws.ensure_layer_caches(layers as usize);
    // The most-frequent tokens for the embedding-anisotropy metric — computed once
    // over the fixed corpus and reused every report so the metric tracks the same
    // token set across the whole run.
    let freq_tokens = diagnostics::frequent_tokens(corpus);

    // Resolve the stages' document spans. The whole-corpus fallback (resume/SFT,
    // or a legacy dataset with no segments.json) has no explicit docs, so
    // synthesize one document covering the entire corpus.
    let ctx = ctx_window as usize;
    let fallback_docs = [DocSpan { start: 0, len: corpus.len() }];
    let stage_docs: Vec<&[DocSpan]> = if plan.is_whole_corpus() {
        vec![&fallback_docs]
    } else {
        plan.stages.iter().map(|s| s.docs.as_slice()).collect()
    };

    // Whole-run budgets for the LR cosine curve and ETA: sum over every stage of
    // (its chunks per epoch × its epochs). Order doesn't affect chunk counts, so
    // these are stable regardless of per-epoch shuffling.
    // With an SFT mask the budget counts only the chunks that train something, so
    // the cosine curve and the ETA are paced against real optimizer steps rather
    // than against forward passes that contribute nothing. A mask always comes with
    // a single-stage plan, so counting over the whole mask is counting this stage.
    // Under a per-epoch shuffle the chunk *boundaries* move, which can shift the
    // count by a chunk or two either way — the trainable tokens themselves are a
    // permutation, so the budget stays right to within that rounding, and both
    // things it feeds (an LR curve and an ETA) are estimates at that resolution.
    let stage_chunks: Vec<usize> = stage_docs
        .iter()
        .map(|docs| match mask {
            Some(m) => trainable_chunks_in(m, ctx),
            None => chunks_in(stage_stream_len(docs), ctx),
        })
        .collect();
    let total_epochs: i64 = plan.stages.iter().map(|s| s.epochs.max(0)).sum();
    let total_chunks_all: u64 = plan
        .stages
        .iter()
        .zip(&stage_chunks)
        .map(|(s, &c)| c as u64 * s.epochs.max(0) as u64)
        .sum();
    // The optimizer steps once per ACCUM_STEPS chunks plus once at each epoch's
    // tail, i.e. ceil(chunks / ACCUM_STEPS) steps per epoch.
    let total_opt_steps: i32 = plan
        .stages
        .iter()
        .zip(&stage_chunks)
        .map(|(s, &c)| c.div_ceil(ACCUM_STEPS) as i32 * s.epochs.max(0) as i32)
        .sum();

    let training_start = Instant::now();
    // Global counters that persist across stages/epochs: Adam step (for bias
    // correction + the LR curve) and chunks completed (for a monotonic chart x-axis
    // and ETA).
    let mut adam_step: i32 = 0;
    let mut chunks_done_global: u64 = 0;
    let mut global_epoch: i64 = 0;

    // Which token ids the run has trained on so far, so each stage can be asked
    // what it introduces. See `warm_start_output_columns` for why a stage that
    // introduces ids is a hazard rather than a non-event.
    let mut seen_ids = vec![false; transformer.embedding_table.len()];

    for (stage_idx, docs) in stage_docs.iter().enumerate() {
        let stage = &plan.stages[stage_idx];
        let stage_total_chunks = stage_chunks[stage_idx];

        // Warm-start the ids this stage is the first to use.
        //
        // A curriculum only ever moves onto *new* text, so a late stage routinely
        // carries ids the earlier ones never did — the chat scaffold and document
        // markup are the extreme case, absent from ordinary prose and so absent
        // from every stage before the one that anneals chat in. Those ids arrive
        // with an untrained input row and an output column the softmax has spent
        // the whole run suppressing, and the first chunk containing one takes a
        // ~20-nat loss and a gradient to match. Reseeding both ends from the table
        // mean at the boundary is the same fix SFT already applies when a marker
        // reaches it untrained; a curriculum stage boundary is the same event.
        //
        // Stage 0 is skipped: nothing is trained yet, so every id is "new" and the
        // random init *is* the warm start. The round trip through the CPU costs one
        // weight download and upload per stage (there are a handful), and it has to
        // go through the model because the f32 master and the f16 view both need
        // rewriting — which is exactly what `upload_weights` does.
        let introduced = ids_introduced_by(corpus, docs, &mut seen_ids);
        if stage_idx > 0 && !introduced.is_empty() {
            sync_weights_from_gpu(ws, transformer);
            let rows = warm_start_embeddings(transformer, &introduced);
            let cols = warm_start_output_columns(transformer, &introduced);
            ws.upload_weights(transformer);
            println!(
                "Stage {} introduces {} token id(s) unseen by the stages before it \
                 — warm-started {rows} embedding row(s) and {cols} output column(s)",
                stage_idx + 1,
                introduced.len(),
            );
        }
        // Early-stopping is scoped to a stage: loss legitimately jumps when a new
        // corpus begins, so patience must not carry across a stage boundary.
        let mut best_loss = f32::INFINITY;
        let mut epochs_since_best = 0;

        for stage_epoch in 0..stage.epochs.max(0) {
            // Order the stage's documents for this epoch (shuffled reproducibly
            // from the run seed + global epoch when enabled), then assemble the
            // epoch's token stream (+ aligned mask for SFT).
            let mut order: Vec<usize> = (0..docs.len()).collect();
            if plan.shuffle {
                let seed = plan
                    .seed
                    .wrapping_add((global_epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                shuffle_seeded(&mut order, seed);
            }
            let (epoch_corpus, epoch_mask) =
                assemble_epoch(corpus, mask, docs, &order, plan.eot_id);

            let stats = train_epoch(
                transformer, ws, &epoch_corpus, epoch_mask.as_deref(), ctx_window, layers, lr,
                total_opt_steps, global_epoch, total_epochs, stage_idx as u64,
                stage.name.as_deref(), stage_total_chunks, total_chunks_all, chunks_done_global,
                training_start, &mut adam_step, &freq_tokens, &on_progress,
            );
            chunks_done_global += stats.chunks as u64;
            println!(
                "Stage {} epoch {}/{} (global {}): avg loss = {:.6} (min={:.4} max={:.4})",
                stage_idx + 1,
                stage_epoch + 1,
                stage.epochs.max(0),
                global_epoch + 1,
                stats.avg_loss,
                stats.min_loss,
                stats.max_loss
            );

            let checkpoint_path =
                save_checkpoint(transformer, ws, model_dir, checkpoints_dir, global_epoch);
            report_epoch_progress(
                global_epoch, total_epochs, stage_idx as u64, stage.name.as_deref(),
                stage_total_chunks, chunks_done_global, total_chunks_all, training_start, &stats,
                checkpoint_path, &on_progress,
            );

            global_epoch += 1;

            if let Some(ref cancel) = cancel {
                if cancel.load(Ordering::SeqCst) {
                    println!("Training cancelled by user at epoch {}", global_epoch);
                    return Ok(true);
                }
            }

            if should_stop_early(
                stats.avg_loss,
                global_epoch - 1,
                &mut best_loss,
                &mut epochs_since_best,
            ) {
                break;
            }
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
    stage: u64,
    stage_name: Option<&str>,
    total_chunks: usize,
    chunks_done_global: u64,
    total_chunks_all: u64,
    training_start: Instant,
    stats: &EpochStats,
    checkpoint_path: Option<String>,
    on_progress: &ProgressCallback,
) {
    on_progress(TrainingProgress {
        epoch: (epoch + 1) as u64,
        total_epochs: epochs as u64,
        chunk: total_chunks as u64,
        total_chunks: total_chunks as u64,
        stage,
        stage_name: stage_name.map(str::to_string),
        global_step: chunks_done_global,
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
        // Epoch-boundary frames carry no fresh forward caches to sample.
        diagnostics: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    // The LR schedule is pure arithmetic (no GPU), so it's unit-testable directly
    // and pins the three anchor points of warmup + cosine decay.
    #[test]
    fn lr_schedule_warmup_then_cosine_decay() {
        let base = 1e-3_f32;
        let total = 1000;
        let warmup = warmup_steps(total);

        // Warmup ramps linearly from ~0 to base_lr at `warmup`.
        assert!(lr_at_step(base, 1, total) < base * 0.05);
        assert!((lr_at_step(base, warmup / 2, total) - base * 0.5).abs() < 1e-9);
        assert!((lr_at_step(base, warmup, total) - base).abs() < 1e-9);

        // Just past warmup the lr is still ~base_lr; by the final step it has
        // decayed to the MIN_LR_RATIO floor; the midpoint sits between the two.
        assert!((lr_at_step(base, warmup + 1, total) - base).abs() < base * 0.01);
        assert!((lr_at_step(base, total, total) - base * MIN_LR_RATIO).abs() < 1e-9);
        let mid = lr_at_step(base, (warmup + total) / 2, total);
        assert!(mid < base && mid > base * MIN_LR_RATIO);

        // The curve is monotonically non-increasing across the decay span.
        let mut prev = lr_at_step(base, warmup, total);
        for step in (warmup + 1)..=total {
            let cur = lr_at_step(base, step, total);
            assert!(cur <= prev + 1e-9, "lr rose at step {step}: {cur} > {prev}");
            prev = cur;
        }
    }

    // Warmup scales with the run, but only downward: the cap keeps pretrains on
    // the flat-100 schedule they were tuned against, and the floor keeps a tiny
    // run from ramping in two steps.
    #[test]
    fn warmup_scales_with_run_length_within_cap_and_floor() {
        // Long pretrain: 5% would be 500, the cap holds it at the old constant.
        assert_eq!(warmup_steps(10_000), WARMUP_STEPS);
        assert_eq!(warmup_steps(WARMUP_STEPS * 20), WARMUP_STEPS);

        // Short SFT: the ratio binds, so warmup is a fraction of the run rather
        // than a third of it.
        assert_eq!(warmup_steps(485), 24); // ~5 epochs of the current chat set
        assert_eq!(warmup_steps(291), 14); // ~3 epochs
        assert!(warmup_steps(291) * 5 < 291, "warmup should be well under 20% here");

        // Degenerate runs: the floor applies and never exceeds the cap.
        assert_eq!(warmup_steps(3), WARMUP_MIN_STEPS);
        assert_eq!(warmup_steps(0), WARMUP_MIN_STEPS);
    }

    // A run shorter than the warmup floor must not divide by zero or emit a
    // negative/NaN lr.
    #[test]
    fn lr_schedule_handles_run_shorter_than_warmup() {
        let base = 1e-3_f32;
        let total = WARMUP_MIN_STEPS / 2; // total_steps < warmup_steps(total)
        assert!(warmup_steps(total) > total);
        for step in 1..=(WARMUP_MIN_STEPS + 5) {
            let lr = lr_at_step(base, step, total);
            assert!(lr.is_finite() && lr >= 0.0, "bad lr {lr} at step {step}");
        }
        // Past warmup it pins to the floor rather than going out of range.
        assert!(
            (lr_at_step(base, WARMUP_MIN_STEPS + 1, total) - base * MIN_LR_RATIO).abs() < 1e-9
        );
    }

    // chunks_in must match what `corpus.chunks(ctx).filter(|c| c.len() >= 2)`
    // actually yields — the last chunk is dropped only when it holds one token.
    #[test]
    fn trainable_chunks_counts_only_windows_with_a_target() {
        let ctx = 4usize;
        // Three windows: one all-context, one with a target, one all-context.
        let mask = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
        assert_eq!(chunks_in(mask.len(), ctx), 3);
        assert_eq!(trainable_chunks_in(&mask, ctx), 1);
    }

    #[test]
    fn a_target_in_the_final_position_of_a_window_does_not_count() {
        // Mask entry i governs the position predicting token i+1, so the last
        // byte of a window is never read — `upload_loss_targets` slices `[..s-1]`.
        // A window whose only 1 sits there has `active == 0` and trains nothing.
        assert_eq!(trainable_chunks_in(&[0, 0, 0, 1], 4), 0);
        assert_eq!(trainable_chunks_in(&[0, 0, 1, 0], 4), 1);
    }

    #[test]
    fn an_all_context_mask_trains_nothing() {
        assert_eq!(trainable_chunks_in(&[0; 64], 8), 0);
    }

    #[test]
    fn a_fully_masked_in_stream_matches_the_plain_chunk_count() {
        let ctx = 8usize;
        for len in 2..40usize {
            let mask = vec![1u8; len];
            assert_eq!(trainable_chunks_in(&mask, ctx), chunks_in(len, ctx), "len={len}");
        }
    }

    #[test]
    fn chunks_in_matches_slice_chunks() {
        let ctx = 4usize;
        for len in 0..40usize {
            let stream: Vec<u32> = (0..len as u32).collect();
            let actual = stream.chunks(ctx).filter(|c| c.len() >= 2).count();
            assert_eq!(chunks_in(len, ctx), actual, "len={len}");
        }
    }

    // The property the stage-boundary warm start is built on: walked in curriculum
    // order, each stage reports exactly the ids no earlier stage used — and reports
    // them once, to the stage that gets there first.
    #[test]
    fn each_stage_reports_only_the_ids_it_introduces() {
        // stage 0 = [7,7,8], stage 1 = [8,9], stage 2 = [9,300]
        let corpus: Vec<u16> = vec![7, 7, 8, 8, 9, 9, 300];
        let stages = [
            vec![DocSpan { start: 0, len: 3 }],
            vec![DocSpan { start: 3, len: 2 }],
            vec![DocSpan { start: 5, len: 2 }],
        ];
        let mut seen = vec![false; 512];

        assert_eq!(ids_introduced_by(&corpus, &stages[0], &mut seen), vec![7, 8]);
        assert_eq!(ids_introduced_by(&corpus, &stages[1], &mut seen), vec![9]);
        assert_eq!(ids_introduced_by(&corpus, &stages[2], &mut seen), vec![300]);
        // Nothing is left to introduce on a second pass.
        assert!(ids_introduced_by(&corpus, &stages[2], &mut seen).is_empty());
    }

    // The chat-scaffold shape of the same thing: prose stages never touch the
    // marker block, so the stage that anneals chat in is the one that introduces
    // every marker at once.
    #[test]
    fn a_late_chat_stage_introduces_the_whole_marker_block() {
        let corpus: Vec<u16> = vec![100, 101, 102, /* chat: */ 260, 100, 261, 262];
        let mut seen = vec![false; 512];
        assert!(ids_introduced_by(&corpus, &[DocSpan { start: 0, len: 3 }], &mut seen).len() == 3);
        assert_eq!(
            ids_introduced_by(&corpus, &[DocSpan { start: 3, len: 4 }], &mut seen),
            vec![260, 261, 262],
            "the markers, and only the markers, are new to the chat stage",
        );
    }

    // A span running past the corpus (a stale segments.json) is truncated rather
    // than panicking — the same tolerance `load_corpus_span` has.
    #[test]
    fn an_overlong_span_is_truncated_not_panicked() {
        let corpus: Vec<u16> = vec![5, 6];
        let mut seen = vec![false; 16];
        assert_eq!(
            ids_introduced_by(&corpus, &[DocSpan { start: 1, len: 99 }], &mut seen),
            vec![6]
        );
    }

    // Both ends of a warm-started token land on the table mean, and no other
    // token's weights move.
    #[test]
    fn warm_start_reseeds_both_ends_and_leaves_the_rest_alone() {
        let vocab = 4usize;
        // Even: RoPE pairs adjacent dimensions, so `Model::init` rejects an odd
        // head_dim (here head_dim == d, since heads == 1).
        let d = 4usize;
        let mut model = Model::init(d as i64, vocab as i64, 8, 1, 2, 1);
        // A suppressed output column and a random embedding row for id 2.
        for row in &mut model.w_output {
            row[2] = -40.0;
        }
        let untouched_embedding = model.embedding_table[1].clone();
        let untouched_column: Vec<f32> = model.w_output.iter().map(|r| r[1]).collect();

        let embed_mean: Vec<f32> = (0..d)
            .map(|j| model.embedding_table.iter().map(|r| r[j]).sum::<f32>() / vocab as f32)
            .collect();
        let out_mean: Vec<f32> =
            model.w_output.iter().map(|r| r.iter().sum::<f32>() / vocab as f32).collect();

        assert_eq!(warm_start_embeddings(&mut model, &[2]), 1);
        assert_eq!(warm_start_output_columns(&mut model, &[2]), 1);

        for j in 0..d {
            assert!((model.embedding_table[2][j] - embed_mean[j]).abs() < 1e-5);
            assert!((model.w_output[j][2] - out_mean[j]).abs() < 1e-5);
        }
        assert_eq!(model.embedding_table[1], untouched_embedding);
        let after: Vec<f32> = model.w_output.iter().map(|r| r[1]).collect();
        assert_eq!(after, untouched_column);
    }

    // An id past the vocabulary is skipped, not a panic — same tolerance
    // `warm_start_embeddings` documents for a drifted `special_ids_seen` record.
    #[test]
    fn warm_start_output_columns_skips_out_of_range_ids() {
        let mut model = Model::init(4, 4, 8, 1, 2, 1);
        assert_eq!(warm_start_output_columns(&mut model, &[99]), 0);
    }

    // A seeded shuffle is a deterministic permutation: same seed => same order,
    // and every index survives exactly once.
    #[test]
    fn shuffle_seeded_is_a_deterministic_permutation() {
        let mut a: Vec<usize> = (0..50).collect();
        let mut b = a.clone();
        shuffle_seeded(&mut a, 12345);
        shuffle_seeded(&mut b, 12345);
        assert_eq!(a, b, "same seed must give the same order");

        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>(), "must be a permutation");

        let mut c: Vec<usize> = (0..50).collect();
        shuffle_seeded(&mut c, 999);
        assert_ne!(a, c, "different seeds should (practically) differ");
    }

    // The default (identity) order reassembles the exact on-disk corpus, since
    // build baked an <|endoftext|> between the same documents; the mask rides
    // along 1:1. A shuffled order still yields a valid permutation of documents
    // separated by the eot.
    #[test]
    fn assemble_epoch_reproduces_and_separates() {
        // corpus: docA=[10,11] eot=1 docB=[20,21,22] eot=1 docC=[30]
        let corpus: Vec<u16> = vec![10, 11, 1, 20, 21, 22, 1, 30];
        let docs = [
            DocSpan { start: 0, len: 2 },
            DocSpan { start: 3, len: 3 },
            DocSpan { start: 7, len: 1 },
        ];
        // Identity order round-trips the on-disk bytes exactly.
        let (out, _) = assemble_epoch(&corpus, None, &docs, &[0, 1, 2], 1);
        assert_eq!(out, corpus);

        // A reorder keeps each document intact, separated by a single eot.
        let (out, _) = assemble_epoch(&corpus, None, &docs, &[2, 0, 1], 1);
        assert_eq!(out, vec![30, 1, 10, 11, 1, 20, 21, 22]);

        // Mask assembly stays aligned and marks separators as ignore (0).
        let mask: Vec<u8> = vec![1, 1, 0, 1, 1, 1, 0, 1];
        let (out, m) = assemble_epoch(&corpus, Some(&mask), &docs, &[2, 0, 1], 1);
        let m = m.unwrap();
        assert_eq!(out.len(), m.len());
        assert_eq!(m, vec![1, 0, 1, 1, 0, 1, 1, 1]);
    }

    #[test]
    fn warm_start_resets_exactly_the_named_rows_to_the_table_mean() {
        // A vocab large enough to cover the reserved special-token block. d=4.
        let d = 4usize;
        let vocab = 300usize;
        // Give every row a distinct, non-mean value so "was it overwritten with
        // the table mean?" is unambiguous.
        let mut rows: Vec<Vec<f32>> = (0..vocab)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32])
            .collect();
        // Force a known mean by making the table symmetric around a center value:
        // simplest is to just compute the expected mean from `rows` below.
        let mut mean = vec![0.0f32; d];
        for r in &rows {
            for (m, &w) in mean.iter_mut().zip(r) {
                *m += w;
            }
        }
        for m in &mut mean {
            *m /= vocab as f32;
        }

        let mut model = Model::init(d as i64, vocab as i64, 4, 1, 1, 1);
        model.embedding_table = std::mem::take(&mut rows);

        // A mix of markers the caller might name — scaffold and document markup,
        // which is the point of the caller owning the set.
        let markers = ["<|assistant|>", "<|/assistant|>", "<|code|>", "<|bold|>"];
        let ids: Vec<u32> = markers
            .iter()
            .map(|m| tokenize::bpe::encode(m, &[])[0])
            .collect();

        let reset = warm_start_embeddings(&mut model, &ids);
        assert_eq!(reset, ids.len(), "every named row reset");

        // Every named id now equals the table mean.
        for (marker, &id) in markers.iter().zip(&ids) {
            for j in 0..d {
                assert!(
                    (model.embedding_table[id as usize][j] - mean[j]).abs() < 1e-3,
                    "{marker} (id {id}) not set to table mean"
                );
            }
        }
        // A marker that was *not* named keeps its value — the regression that
        // matters, since over-resetting is what wrecked the model last time.
        let untouched = tokenize::bpe::encode("<|user|>", &[])[0] as usize;
        assert!(
            (model.embedding_table[untouched][0] - untouched as f32).abs() < 1e-6,
            "an unnamed marker was reset"
        );
        // A plain byte token (id 65 = 'A') is untouched.
        assert!((model.embedding_table[65][0] - 65.0).abs() < 1e-6);
    }

    #[test]
    fn warm_start_with_nothing_named_changes_nothing() {
        // The healthy steady state once the pretrain corpus carries the markers:
        // the caller names no rows and the model is passed through untouched.
        let mut model = Model::init(4, 300, 4, 1, 1, 1);
        let before = model.embedding_table.clone();
        assert_eq!(warm_start_embeddings(&mut model, &[]), 0);
        assert_eq!(model.embedding_table, before);
    }

    #[test]
    fn warm_start_skips_ids_past_the_vocabulary() {
        // A base model whose recorded marker set disagrees with its actual vocab
        // should not panic a run that is otherwise fine.
        let mut model = Model::init(4, 300, 4, 1, 1, 1);
        assert_eq!(warm_start_embeddings(&mut model, &[299, 300, 100_000]), 1);
    }
}
