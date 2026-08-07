use std::ffi::c_void;
use crate::training::embed::embed;
mod serialization;
mod forward;
mod backward;
pub use serialization::{LayerRow, ModelHeader, read_model_header};
use crate::training::gpu_workspace::Workspace;

// Layer-norm numerical-stability epsilon, shared by every layer-norm forward and
// backward call — including the inference forward in `training::infer_session`,
// which must normalize identically to the training one or the two forwards
// disagree.
pub(crate) const LAYER_NORM_EPS: f32 = 1e-5;
// Per-step gradient L2-norm clip applied inside train_step before accumulation.
const TRAIN_STEP_CLIP_NORM: f32 = 1.0;

// The per-step tensor dimensions, derived once from the model + chunk length and
// threaded through the train_step stage helpers.
#[derive(Clone, Copy)]
struct StepDims {
    s: usize,     // sequence length (this chunk)
    d: usize,     // model dimension
    h: usize,     // number of heads
    hd: usize,    // per-head dimension
    ff: usize,    // feed-forward width
    vocab: usize, // vocabulary size
}

#[derive(Clone)]
/// without loading any weights — used to report a model's architecture.
pub struct Model {
    pub embedding_table: Vec<Vec<f32>>,
    /// Longest sequence the model can attend over.
    ///
    /// Its own field because nothing else carries it: callers used to read
    /// `positional_table.len()`; RoPE removed that table, so the value has to
    /// live somewhere. Read it through `ctx_window()`.
    pub max_seq_len: i64,
    pub wq: Vec<Vec<Vec<Vec<f32>>>>,  // [layers][heads][d][hd]
    pub wk: Vec<Vec<Vec<Vec<f32>>>>,
    pub wv: Vec<Vec<Vec<Vec<f32>>>>,
    pub wo: Vec<Vec<Vec<f32>>>,       // [layers][d][d]
    pub heads: i64,
    pub head_dim: i64,
    pub layers: i64,
    pub w1: Vec<Vec<Vec<f32>>>,       // [layers][d][ff]
    pub w2: Vec<Vec<Vec<f32>>>,       // [layers][ff][d]
    pub b1: Vec<Vec<f32>>,            // [layers][ff]
    pub b2: Vec<Vec<f32>>,            // [layers][d]
    pub ln_gamma: Vec<Vec<f32>>,      // [layers][d]
    pub ln_beta: Vec<Vec<f32>>,       // [layers][d]
    /// Final LayerNorm over the residual stream, applied before `w_output`.
    ///
    /// The blocks are Pre-LN, so each one normalizes its *own* input and adds
    /// into the residual — nothing normalizes the residual stream itself, and
    /// its scale grows with depth. Without this the logits grow with it.
    ///
    /// **Empty for a pre-v5 model**, which trained with no final norm at all. A
    /// LayerNorm at gamma=1/beta=0 is not the identity, so synthesizing one for
    /// those weights would change what an existing model outputs; they keep the
    /// unnormalized projection they were trained against. Test with
    /// `has_final_ln`, never by inspecting the values.
    pub ln_final_gamma: Vec<f32>,     // [d]
    pub ln_final_beta: Vec<f32>,      // [d]
    /// The output projection, `[d, vocab]` — **empty when the model ties its
    /// output projection to its embedding table**, which every model from
    /// `MODEL_VERSION` 6 does.
    ///
    /// Tying means there is one `[vocab, d]` matrix instead of two transposed
    /// ones: the embedding gathers rows from it going in, and `project_logits`
    /// multiplies by it transposed coming out. Both paths then accumulate into
    /// the same gradient, and the optimizer steps it once.
    ///
    /// It is worth exactly one doubling of the vocabulary. The untied pair costs
    /// `2 * vocab * d` parameters and the tied single costs `vocab * d`, so tied
    /// at vocab `2V` is the same parameter count as untied at `V` — at d 512 both
    /// are 8,388,608 — while compressing ~10% better (3.89 → 4.28 held-out
    /// bytes/token on FineWeb at 8192 → 16384). It is also near-free in compute:
    /// the bigger output projection costs more per token but there are
    /// proportionally fewer tokens, which nets out at about +1% per corpus byte.
    ///
    /// Test with [`Model::ties_embeddings`], never by inspecting the values —
    /// the same rule as [`Model::has_final_ln`], and for the same reason. A
    /// pre-v6 model trained *against* a separate output projection, so
    /// synthesizing a tie for it would change what it outputs.
    pub w_output: Vec<Vec<f32>>,
}

impl Model {
    /// Build a fresh model with randomly-initialized weights. `dimensions` must
    /// be divisible by `heads`; the feed-forward width is `dimensions * ff_multiplier`.
    pub fn init(dimensions: i64, vocab: i64, ctx_window: i64, heads: i64, ff_multiplier: i64, layers: i64) -> Model {
        let head_dim: i64 = dimensions / heads;
        // RoPE rotates adjacent pairs (2i, 2i+1) of each head's dimensions, so an
        // odd head_dim would leave the last one unrotated — a silently
        // half-positioned model rather than a failure. Cheap to catch here.
        assert!(
            head_dim % 2 == 0,
            "head_dim must be even for RoPE: dimensions {dimensions} / heads {heads} = {head_dim}",
        );
        let d_ff = dimensions * ff_multiplier;
        let h = heads as usize;
        let d = dimensions as usize;
        let ff = d_ff as usize;
        let l = layers as usize;

        let mut wq: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();
        let mut wk: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();
        let mut wv: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();
        let mut wo: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut w1: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut w2: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut b1: Vec<Vec<f32>> = Vec::new();
        let mut b2: Vec<Vec<f32>> = Vec::new();
        let mut ln_gamma: Vec<Vec<f32>> = Vec::new();
        let mut ln_beta: Vec<Vec<f32>> = Vec::new();

        for _ in 0..l {
            let mut layer_wq = Vec::new();
            let mut layer_wk = Vec::new();
            let mut layer_wv = Vec::new();
            for _ in 0..h {
                layer_wq.push(embed(head_dim, dimensions));
                layer_wk.push(embed(head_dim, dimensions));
                layer_wv.push(embed(head_dim, dimensions));
            }
            wq.push(layer_wq);
            wk.push(layer_wk);
            wv.push(layer_wv);
            wo.push(embed(dimensions, dimensions));
            w1.push(embed(d_ff, dimensions));
            w2.push(embed(dimensions, d_ff));
            b1.push(vec![0.0f32; ff]);
            b2.push(vec![0.0f32; d]);
            ln_gamma.push(vec![1.0f32; d]);
            ln_beta.push(vec![0.0f32; d]);
        }

        Model {
            embedding_table: embed(dimensions, vocab),
            max_seq_len: ctx_window,
            wq, wk, wv, wo, w1, w2, b1, b2, ln_gamma, ln_beta,
            ln_final_gamma: vec![1.0f32; d],
            ln_final_beta: vec![0.0f32; d],
            heads, head_dim, layers,
            // Tied: the embedding table is the output projection. See the field.
            w_output: Vec::new(),
        }
    }

    /// Whether this model carries the final LayerNorm — false only for weights
    /// saved before `MODEL_VERSION` 5. See the `ln_final_gamma` field.
    pub fn has_final_ln(&self) -> bool {
        !self.ln_final_gamma.is_empty()
    }

    /// Whether the output projection is the embedding table transposed, rather
    /// than a second matrix. True for every model from `MODEL_VERSION` 6; false
    /// only for weights saved before it. See the `w_output` field.
    pub fn ties_embeddings(&self) -> bool {
        self.w_output.is_empty()
    }

    /// Vocabulary size, read off the embedding table — which is present whether
    /// or not the model ties, unlike `w_output`.
    pub fn vocab(&self) -> usize {
        self.embedding_table.len()
    }

    /// The model's context window, as a `usize` for slicing.
    pub fn ctx_window(&self) -> usize {
        self.max_seq_len as usize
    }

    /// Run one forward+backward pass over a single `chunk`, accumulating gradients
    /// into `ws` (which it mutates) and returning the scalar cross-entropy loss.
    /// `chunk` must hold at least 2 tokens (position i is trained to predict token
    /// i+1). Weights are not updated here — the caller drives the optimizer step.
    pub async fn train_step(&self, ws: &mut Workspace, chunk: &[u32], layers: usize) -> f32 {
        self.run_step(ws, chunk, None, layers).await
    }

    /// One SFT training step over `chunk` with a per-position loss `mask` (aligned
    /// to `chunk`, `1`=train / `0`=ignore per the ISSUE-0031 convention). Only the
    /// masked-in positions (the assistant reply) contribute loss and gradient;
    /// everything else is forward-passed but produces no learning signal.
    pub async fn train_step_masked(
        &self,
        ws: &mut Workspace,
        chunk: &[u32],
        mask: &[u8],
        layers: usize,
    ) -> f32 {
        self.run_step(ws, chunk, Some(mask), layers).await
    }

    /// One preference-training step over `chunk`: forward, then a backward whose
    /// gradient is scaled by KTO's per-example scalar.
    ///
    /// This is the same work as `train_step_masked` with one thing inserted in the
    /// middle. The masked cross-entropy the forward already produces is
    /// `-log π(reply) / active`, so the sequence log-probability KTO needs is free —
    /// it falls out of the loss, no extra forward pass. That value plus the
    /// precomputed `ref_logprob` gives the gradient scale, which the backward then
    /// applies. Hence one forward and one backward per example, the same cost as SFT.
    ///
    /// `ref_logprob` is `log π_ref(reply | prompt)` under the frozen base model,
    /// computed when the dataset was built. `kl` is the detached KL baseline.
    ///
    /// Returns `None` when the chunk has no reply tokens to train on (nothing was
    /// accumulated, and the caller should skip it rather than count a zero).
    pub async fn train_step_kto(
        &self,
        ws: &mut Workspace,
        chunk: &[u32],
        mask: &[u8],
        layers: usize,
        ref_logprob: f64,
        desirable: bool,
        kl: f64,
        params: &crate::kto::KtoParams,
    ) -> Option<crate::kto::KtoStep> {
        let dims = self.step_dims(chunk, layers);
        ws.ensure_layer_caches(layers);

        let mut d_x = self.embed_forward(ws, dims, chunk);
        for layer_idx in 0..layers {
            let x = self.attn_sublayer_forward(ws, dims, layer_idx, d_x);
            d_x = self.ffn_sublayer_forward(ws, dims, layer_idx, x);
        }
        let d_h = self.final_ln_forward(ws, dims, d_x);
        self.project_logits(ws, dims, d_h);

        let targets = self.upload_loss_targets(ws, dims, chunk, Some(mask));
        let loss = self.loss_value(ws, dims, &targets)?;
        let active = targets.active() as usize;

        // The masked loss is the *mean* cross-entropy over the reply's tokens, and
        // cross-entropy is the negative log-probability, so the sequence total is
        // just the mean scaled back up by the token count.
        let logprob = -(loss as f64) * active as f64;
        let step = crate::kto::step(logprob, ref_logprob, desirable, kl, active, params);

        // A non-finite forward (a diverged run) would otherwise write NaNs into
        // every accumulator, and the caller cannot tell that from a real step.
        if !step.grad_scale.is_finite() {
            return None;
        }
        self.loss_gradient(ws, dims, &targets, step.grad_scale as f32);

        let d_dh = self.backward_unembed(ws, dims, d_h);
        let mut d_dx = self.final_ln_backward(ws, dims, d_dh);
        for layer_idx in (0..layers).rev() {
            let g = self.ffn_sublayer_backward(ws, dims, layer_idx, d_dx);
            d_dx = self.attn_sublayer_backward(ws, dims, layer_idx, g);
        }
        self.clip_gradients(ws, dims, layers);
        self.accumulate_embedding_grad(ws, dims, d_dx);

        Some(step)
    }

    // The per-step tensor dimensions, shared by every step entry point so the
    // three of them cannot drift apart.
    fn step_dims(&self, chunk: &[u32], _layers: usize) -> StepDims {
        StepDims {
            s: chunk.len(),
            d: self.wo[0].len(),
            h: self.heads as usize,
            hd: self.head_dim as usize,
            ff: self.w1[0][0].len(),
            // From the embedding table, which is present whether or not the
            // model ties — `w_output` is empty for a tied one.
            vocab: self.vocab(),
        }
    }

    // Shared forward+backward body for both the pretraining (`mask == None`) and
    // SFT (`mask == Some`) steps. Only the loss stage differs between them.
    async fn run_step(
        &self,
        ws: &mut Workspace,
        chunk: &[u32],
        mask: Option<&[u8]>,
        layers: usize,
    ) -> f32 {
        let dims = self.step_dims(chunk, layers);
        ws.ensure_layer_caches(layers);

        let t = phase_start();
        let mut d_x = self.embed_forward(ws, dims, chunk);
        phase_end("1. embed", t);

        for layer_idx in 0..layers {
            let t = phase_start();
            let x = self.attn_sublayer_forward(ws, dims, layer_idx, d_x);
            phase_end("2. attn fwd", t);

            let t = phase_start();
            d_x = self.ffn_sublayer_forward(ws, dims, layer_idx, x);
            phase_end("3. ffn fwd", t);
        }

        let t = phase_start();
        let d_h = self.final_ln_forward(ws, dims, d_x);
        self.project_logits(ws, dims, d_h);
        phase_end("4. logits", t);

        let t = phase_start();
        let loss = self.compute_loss(ws, dims, chunk, mask);
        phase_end("5. loss", t);

        let t = phase_start();
        let d_dh = self.backward_unembed(ws, dims, d_h);
        let mut d_dx = self.final_ln_backward(ws, dims, d_dh);
        phase_end("6. unembed bwd", t);

        for layer_idx in (0..layers).rev() {
            let t = phase_start();
            let g = self.ffn_sublayer_backward(ws, dims, layer_idx, d_dx);
            phase_end("7. ffn bwd", t);

            let t = phase_start();
            d_dx = self.attn_sublayer_backward(ws, dims, layer_idx, g);
            phase_end("8. attn bwd", t);
        }

        let t = phase_start();
        self.clip_gradients(ws, dims, layers);
        self.accumulate_embedding_grad(ws, dims, d_dx);
        phase_end("9. clip + embed grad", t);

        loss
    }

}

// ── Optional per-phase step profiling ───────────────────────────────────────
//
// CUPTI kernel tracing is unavailable under WSL2, so `nsys` collects the CUDA
// API trace but cannot attribute time to individual kernels there. This fills
// that gap at stage granularity: with `LLM_PROFILE_STEP=1` set, every stage of
// `run_step` is bracketed by a device synchronize and its wall time
// accumulated, which is enough to say *which stage* a slow step is in.
//
// Off by default, and deliberately so: the synchronizes it inserts serialize
// the pipeline and distort the very number being measured. This is a diagnostic
// to run on purpose, not a metric to leave on. When off it costs one cached env
// lookup per stage and nothing else.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn profiling_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LLM_PROFILE_STEP").is_ok_and(|v| v != "0"))
}

// name -> (total milliseconds, number of calls)
static PHASE_TOTALS: Mutex<BTreeMap<&'static str, (f64, u64)>> = Mutex::new(BTreeMap::new());

fn phase_start() -> Option<Instant> {
    if !profiling_enabled() {
        return None;
    }
    // SAFETY: no arguments, no pointers; blocks until the device is idle so the
    // interval measured is this stage's GPU work rather than queue depth.
    unsafe { cudaDeviceSynchronize() };
    Some(Instant::now())
}

fn phase_end(name: &'static str, start: Option<Instant>) {
    let Some(start) = start else { return };
    // SAFETY: as above.
    unsafe { cudaDeviceSynchronize() };
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut totals = PHASE_TOTALS.lock().unwrap();
    let entry = totals.entry(name).or_insert((0.0, 0));
    entry.0 += ms;
    entry.1 += 1;
}

// ── Optional per-phase inference profiling ──────────────────────────────────
//
// The same idea as the step profiler above, on its own gate (`LLM_PROFILE_INFER`)
// and its own accumulator so the two can be read independently.
//
// The granularity is deliberately finer than the step profiler's. A forward
// stage here is not one thing: it flattens weights on the host, converts them
// f32→f16 element by element, allocates device memory, uploads, launches, then
// copies the result back and converts it again. Timing the stage as a whole
// would say "attention is slow" and stop there. Splitting it into `up` (host
// marshalling + H2D), `gpu` (the launch) and `dn` (D2H + conversion) is what
// distinguishes a kernel that is genuinely expensive from a kernel that is
// starved — and the latter is what a low GPU-utilization reading means.
//
// The synchronizes distort what they measure, so this is off by default too.

fn infer_profiling_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LLM_PROFILE_INFER").is_ok_and(|v| v != "0"))
}

static INFER_TOTALS: Mutex<BTreeMap<&'static str, (f64, u64)>> = Mutex::new(BTreeMap::new());

pub(crate) fn infer_start() -> Option<Instant> {
    if !infer_profiling_enabled() {
        return None;
    }
    // SAFETY: no arguments, no pointers. See `phase_start`.
    unsafe { cudaDeviceSynchronize() };
    Some(Instant::now())
}

pub(crate) fn infer_end(name: &'static str, start: Option<Instant>) {
    let Some(start) = start else { return };
    // SAFETY: as above.
    unsafe { cudaDeviceSynchronize() };
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut totals = INFER_TOTALS.lock().unwrap();
    let entry = totals.entry(name).or_insert((0.0, 0));
    entry.0 += ms;
    entry.1 += 1;
}

/// Print the accumulated per-stage inference profile (and reset it), sorted by
/// total time. `tokens` is the number of generated tokens the totals cover.
///
/// Does nothing unless `LLM_PROFILE_INFER` is set.
pub fn dump_infer_profile(tokens: usize) {
    if !infer_profiling_enabled() {
        eprintln!("(inference profiling off — set LLM_PROFILE_INFER=1)");
        return;
    }
    let mut totals = INFER_TOTALS.lock().unwrap();
    let grand: f64 = totals.values().map(|(ms, _)| *ms).sum();
    let mut rows: Vec<_> = totals.iter().map(|(n, (ms, c))| (*n, *ms, *c)).collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\n  stage                 ms/token     % of token   calls/token");
    for (name, ms, calls) in rows {
        eprintln!(
            "  {:<20} {:>9.2} {:>11.1}% {:>12.1}",
            name,
            ms / tokens as f64,
            ms / grand * 100.0,
            calls as f64 / tokens as f64,
        );
    }
    eprintln!("  {:<20} {:>9.2}\n", "TOTAL", grand / tokens as f64);
    totals.clear();
}

/// Print the accumulated per-stage profile (and reset it), sorted by total time.
///
/// Does nothing unless `LLM_PROFILE_STEP` is set. `steps` is the number of
/// training steps the totals cover, so the report can show per-step cost.
pub fn dump_step_profile(steps: usize) {
    if !profiling_enabled() {
        eprintln!("(step profiling off — set LLM_PROFILE_STEP=1)");
        return;
    }
    let mut totals = PHASE_TOTALS.lock().unwrap();
    let grand: f64 = totals.values().map(|(ms, _)| *ms).sum();
    let mut rows: Vec<_> = totals.iter().map(|(n, (ms, c))| (*n, *ms, *c)).collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\n  stage                  ms/step     % of step    calls/step");
    for (name, ms, calls) in rows {
        eprintln!(
            "  {:<20} {:>9.2} {:>11.1}% {:>12.1}",
            name,
            ms / steps as f64,
            ms / grand * 100.0,
            calls as f64 / steps as f64,
        );
    }
    eprintln!("  {:<20} {:>9.2}\n", "TOTAL", grand / steps as f64);
    totals.clear();
}

// ── Model (de)serialization ─────────────────────────────────────────────────
// One definition of the on-disk format, shared by the file-backed (`save`/
// `load`) and in-memory (`to_bytes`/`from_bytes`) paths by being generic over
// the writer/reader — so the byte layout is described exactly once.

unsafe extern "C" {
    fn cudaDeviceSynchronize() -> i32;
    fn cuda_memcpy_d2d(dst: *mut c_void, src: *const c_void, size: usize);
    fn cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize);

    fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;

    fn cuda_add_inplace_f32(dst: *mut f32, src: *const f32, n: i32);
    fn cuda_add_inplace_f16(dst: *mut c_void, src: *const c_void, n: i32);

    fn cuda_cross_entropy_backward(
        logits: *const c_void,
        targets: *const f32,
        mask: *const f32,
        dlogits: *mut c_void,
        seq_len: i32,
        vocab: i32,
        active: i32,
        // Multiplies the whole gradient. 1.0 for pretrain/SFT; KTO's per-example
        // scalar (negative for a rejected reply) otherwise.
        scale: f32,
    );

    fn cuda_cross_entropy_loss(
        logits: *const c_void,
        targets: *const f32,
        mask: *const f32,
        seq_len: i32,
        vocab: i32,
        active: i32,
    ) -> f32;

    fn cuda_matmul(
        a: *const c_void,
        b: *const c_void,
        result: *mut c_void,
        rows_a: i32,
        cols_a: i32,
        cols_b: i32,
    );

    fn cuda_matmul_bt(
        a: *const c_void,
        b: *const c_void,
        result: *mut c_void,
        rows_a: i32,
        cols_a: i32,
        rows_b: i32,
    );

    fn cuda_layer_norm(
        in_: *const c_void,
        out: *mut c_void,
        gamma: *const c_void,
        beta: *const c_void,
        rows: i32,
        cols: i32,
        eps: f32,
    );

    fn cuda_layer_norm_backward(
        x: *const c_void,
        dy: *const c_void,
        gamma: *const c_void,
        dx: *mut c_void,
        dgamma: *mut f32,
        dbeta: *mut f32,
        rows: i32,
        cols: i32,
        eps: f32,
    );

    fn cuda_d_weight(
        A: *const c_void,
        dB: *const c_void,
        dW: *mut f32,
        rows: i32,
        cols_a: i32,
        cols_b: i32,
    );

    fn cuda_concat_heads(
        head_ptrs: *mut *mut c_void,
        out: *mut c_void,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
    );

    fn cuda_clip_norm(g: *mut f32, n: i32, max_norm: f32);
    // Batched form: clips every gradient tensor in the workspace's clip table
    // in two launches. See `Model::clip_gradients`.
    fn cuda_clip_norm_batched(
        ptrs: *mut *mut f32,
        sizes: *const i32,
        norms: *mut f32,
        n_buffers: i32,
        max_norm: f32,
    );

    fn cuda_clamp_f16(buf: *mut c_void, n: i32);

    fn cuda_feed_forward_forward(
        input: *const c_void,
        w1: *const c_void,
        w2: *const c_void,
        b1: *const c_void,
        b2: *const c_void,
        out: *mut c_void,
        pre_gelu: *mut c_void,
        seq_len: i32,
        d_model: i32,
        d_ff: i32,
    );

    fn cuda_feed_forward_backward(
        input: *const c_void,
        pre_gelu: *const c_void,
        dout: *const c_void,
        w1: *const c_void,
        w2: *const c_void,
        dinput: *mut c_void,
        dw1: *mut f32,
        dw2: *mut f32,
        db1: *mut f32,
        db2: *mut f32,
        seq_len: i32,
        d_model: i32,
        d_ff: i32,
    );


    fn cuda_embedding_forward(
        token_ids: *const i32,
        embedding_table: *const c_void,
        output: *mut c_void,
        s: i32,
        d: i32,
        vocab: i32,
    );

    fn cuda_scatter_add_embedding_grad(
        d_dx: *const c_void,
        token_ids: *const i32,
        g_emb: *mut f32,
        s: i32,
        d: i32,
        vocab: i32,
    );

    fn cuda_attention_forward_batched(
        X: *const c_void,
        wq_all: *const c_void,
        wk_all: *const c_void,
        wv_all: *const c_void,
        out_all: *mut c_void,
        // Legacy path only — null on the flash path, which never materializes it.
        P_all: *mut c_void,
        // Flash path only: the per-row logsumexp the backward reads instead of
        // `P_all`, and an optional per-head entropy accumulator for diagnostics.
        lse_all: *mut f32,
        ent_all: *mut f32,
        Q_all: *mut c_void,
        K_all: *mut c_void,
        V_all: *mut c_void,
        // Inference KV cache; training passes null/0, which leaves every stride in
        // the kernel at its `seq`-based value. See `infer_session`.
        k_cache_f32: *mut f32,
        kv_stride: i32,
        seq: i32, d: i32, hd: i32, h: i32,
    );

    fn cuda_attention_backward_batched(
        X: *const c_void,
        wq_all: *const c_void,
        wk_all: *const c_void,
        wv_all: *const c_void,
        P_all: *const c_void,
        // Flash path only: the forward's output (for D = rowsum(dO ∘ O)) and its
        // per-row logsumexp, in place of `P_all`.
        O_all: *const c_void,
        lse_all: *const f32,
        Q_all: *const c_void,
        K_all: *const c_void,
        V_all: *const c_void,
        dout_full: *const c_void,
        dX: *mut c_void,
        dwq_all: *mut f32,
        dwk_all: *mut f32,
        dwv_all: *mut f32,
        seq: i32, d: i32, hd: i32, h: i32,
    );


}

// Every trainable gradient buffer as (ptr, element count). Shared by the
// clipping pass and (implicitly) mirrors what the Adam step touches.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::gpu_workspace::Workspace;
    use crate::training::optimizer::gpu_adam_update;
    use crate::training::sync::sync_weights_from_gpu;

    #[test]
    fn test_embedding_forward_cpu_vs_gpu() {
        let vocab = 10;
        let d = 4;
        let ctx = 8;
        let heads = 2;
        let ff_mult = 2;
        let layers = 1;

        let t = Model::init(d, vocab, ctx, heads, ff_mult, layers);
        let ws = Workspace::new(&t, ctx as usize);

        let chunk: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let s = chunk.len();
        let du = d as usize;

        // CPU embedding forward
        let flat_emb: Vec<f32> = chunk
            .iter()
            .flat_map(|&id| t.embedding_table[id as usize].clone())
            .collect();
        // Just the embedding row now — the kernel is a plain gather since RoPE
        // took over positional encoding.
        let expected = flat_emb.clone();

        // GPU embedding forward via extern "C" kernel
        let token_ids: Vec<i32> = chunk.iter().map(|&id| id as i32).collect();
        unsafe {
            super::cuda_memcpy_h2d(
                ws.d_token_ids.ptr as *mut std::ffi::c_void,
                token_ids.as_ptr() as *const std::ffi::c_void,
                (s * 4) as usize,
            );
            super::cuda_embedding_forward(
                ws.d_token_ids.ptr as *const i32,
                ws.d_embedding.ptr as *const std::ffi::c_void,
                ws.b_sd[0].ptr as *mut std::ffi::c_void,
                s as i32,
                d as i32,
                vocab as i32,
            );
        }
        let gpu_out = ws.b_sd[0].to_f32_vec();

        let mut max_diff = 0.0f32;
        let mut any_mismatch = false;
        for i in 0..s * du {
            let diff = (gpu_out[i] - expected[i]).abs();
            if diff > 1e-3 {
                any_mismatch = true;
                if diff > max_diff {
                    max_diff = diff;
                }
                eprintln!("  MISMATCH [{}] pos={} j={}  gpu={:.6}  cpu={:.6}  diff={:.6}",
                    i, i / du, i % du, gpu_out[i], expected[i], diff);
            }
        }
        if any_mismatch {
            panic!("GPU vs CPU embedding forward mismatch! max_diff={:.6}", max_diff);
        }
        eprintln!("✓ GPU and CPU embedding forward match (max_diff={:.6})", max_diff);
    }

    #[test]
    fn test_embedding_backward_gpu() {
        let vocab = 5;
        // 4, not 3: `d` must divide by `heads` (3/2 truncated to head_dim 1,
        // covering only 2 of 3 dimensions) and head_dim must be even for RoPE.
        let d = 4;
        let ctx = 8;
        let heads = 2;
        let ff_mult = 2;
        let layers = 1;

        let t = Model::init(d, vocab, ctx, heads, ff_mult, layers);
        let ws = Workspace::new(&t, ctx as usize);

        let chunk: Vec<u32> = vec![0, 1, 2, 0, 1];
        let s = chunk.len();
        let du = d as usize;

        // Create a synthetic gradient d_dx on CPU [s x d]
        let mut cpu_dx: Vec<f32> = vec![0.0; s * du];
        for i in 0..s * du {
            cpu_dx[i] = (i as f32) * 0.1 + 1.0;
        }

        // CPU reference: scatter-add into g_emb
        let mut cpu_grad = vec![0.0f32; vocab as usize * du];
        for pos in 0..s {
            let token = chunk[pos] as usize;
            for j in 0..du {
                cpu_grad[token * du + j] += cpu_dx[pos * du + j];
            }
        }

        // Upload d_dx to GPU (b_sd[0]), zero g_d_embedding, run scatter_add kernel
        ws.b_sd[0].upload_f32_slice(&cpu_dx);
        unsafe {
            super::cudaMemset(
                ws.g_d_embedding.ptr as *mut std::ffi::c_void,
                0,
                (vocab as usize * du * 4) as usize,
            );
            let token_ids: Vec<i32> = chunk.iter().map(|&id| id as i32).collect();
            super::cuda_memcpy_h2d(
                ws.d_token_ids.ptr as *mut std::ffi::c_void,
                token_ids.as_ptr() as *const std::ffi::c_void,
                (s * 4) as usize,
            );
            super::cuda_scatter_add_embedding_grad(
                ws.b_sd[0].ptr as *const std::ffi::c_void,
                ws.d_token_ids.ptr as *const i32,
                ws.g_d_embedding.ptr,
                s as i32,
                d as i32,
                vocab as i32,
            );
        }
        let gpu_grad = ws.g_d_embedding.to_vec();

        // Compare
        let mut max_diff = 0.0f32;
        let mut any_mismatch = false;
        for i in 0..(vocab as usize * du) {
            let diff = (gpu_grad[i] - cpu_grad[i]).abs();
            if diff > 1e-2 {
                any_mismatch = true;
                if diff > max_diff {
                    max_diff = diff;
                }
                eprintln!("  MISMATCH [{}]  gpu={:.6}  cpu={:.6}  diff={:.6}",
                    i, gpu_grad[i], cpu_grad[i], diff);
            }
        }
        if any_mismatch {
            panic!("GPU vs CPU embedding backward mismatch! max_diff={:.6}", max_diff);
        }
        eprintln!("✓ GPU and CPU embedding backward match (max_diff={:.6})", max_diff);
    }

    #[test]
    fn test_train_step_loss_reasonable() {
        let vocab = 20;
        let d = 8;
        let ctx = 8;
        let heads = 2;
        let ff_mult = 2;
        let layers = 2;

        let t = Model::init(d, vocab, ctx, heads, ff_mult, layers);
        let mut ws = Workspace::new(&t, ctx as usize);

        let chunk: Vec<u32> = (0..8).map(|i| i as u32 % (vocab as u32)).collect();
        let loss = pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));

        let expected_random_loss = (vocab as f32).ln();
        eprintln!("  loss={:.4}  expected≈ln(vocab={})={:.4}",
            loss, vocab, expected_random_loss);
        let ratio = loss / expected_random_loss;
        assert!(ratio > 0.5 && ratio < 2.0,
            "loss {:.4} too far from ln(vocab)={:.4} (ratio={:.2})",
            loss, expected_random_loss, ratio);
        eprintln!("✓ train_step produces reasonable loss (ratio={:.2})", ratio);
    }

    #[test]
    fn test_gpu_sgd_does_not_corrupt_weights() {
        let vocab = 20;
        let d = 8;
        let ctx = 8;
        let heads = 2;
        let ff_mult = 2;
        let layers = 2;

        let t = Model::init(d, vocab, ctx, heads, ff_mult, layers);
        let mut ws = Workspace::new(&t, ctx as usize);

        // Record pre-SGD embedding weights
        let pre_emb: Vec<f32> = t.embedding_table.iter().flatten().copied().collect();

        let chunk: Vec<u32> = (0..8).map(|i| i as u32 % (vocab as u32)).collect();
        let _loss = pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));

        let _loss = pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));
        gpu_adam_update(&ws, 0.001, 1, 1, layers as usize);
        ws.zero_accumulators();

        let mut t_synced = Model {
            embedding_table: t.embedding_table.clone(),
            max_seq_len: t.max_seq_len,
            wq: t.wq.clone(), wk: t.wk.clone(), wv: t.wv.clone(),
            wo: t.wo.clone(), w1: t.w1.clone(), w2: t.w2.clone(),
            b1: t.b1.clone(), b2: t.b2.clone(),
            ln_gamma: t.ln_gamma.clone(), ln_beta: t.ln_beta.clone(),
            ln_final_gamma: t.ln_final_gamma.clone(),
            ln_final_beta: t.ln_final_beta.clone(),
            w_output: t.w_output.clone(),
            heads: t.heads, head_dim: t.head_dim, layers: t.layers,
        };
        sync_weights_from_gpu(&ws, &mut t_synced);
        let post_emb: Vec<f32> = t_synced.embedding_table.iter().flatten().copied().collect();

        let mut total_change = 0.0f32;
        for i in 0..pre_emb.len() {
            total_change += (post_emb[i] - pre_emb[i]).abs();
            assert!(post_emb[i].is_finite(), "post-SGD weight not finite at {}", i);
        }
        assert!(total_change > 0.0, "SGD should change at least some weights");
        eprintln!("✓ GPU SGD changes weights (total_change={:.6})", total_change);
    }

    // The single most important training test: prove the model can actually
    // LEARN, not just that a step runs and mutates weights. We repeatedly train
    // one fixed tiny sequence (a batch of size 1) and assert the loss collapses
    // far below its random starting point ln(vocab). If the backward pass has a
    // wrong sign, a disconnected gradient, a bad optimizer step, or a
    // clip/scale that throttles learning to zero, every other test here still
    // passes but this one fails — the loss just sits near ln(vocab) forever.
    // This is the guardrail against the "loss plateaus and won't drop" class of
    // regression, distinguished from a genuine data/capacity plateau because on
    // a single memorizable sequence a correct trainer must reach ~0.
    #[test]
    fn test_overfits_single_sequence() {
        let vocab = 20;
        let d = 16;
        let ctx = 8;
        let heads = 2;
        let ff_mult = 2;
        let layers = 2;

        let t = Model::init(d, vocab, ctx, heads, ff_mult, layers);
        let mut ws = Workspace::new(&t, ctx as usize);

        // A fixed, arbitrary sequence the model must memorize.
        let chunk: Vec<u32> = vec![3, 7, 1, 14, 2, 9, 5, 11];

        let first_loss = pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));
        ws.zero_accumulators();

        // Plain per-step Adam (accum_steps = 1), no warmup/clip games — we want
        // to test the core learning mechanism in isolation.
        let mut last_loss = first_loss;
        for step in 1..=200 {
            last_loss = pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));
            gpu_adam_update(&ws, 0.01, step, 1, layers as usize);
            ws.zero_accumulators();
        }

        let random_loss = (vocab as f32).ln();
        eprintln!(
            "  first_loss={:.4}  last_loss={:.4}  ln(vocab)={:.4}",
            first_loss, last_loss, random_loss
        );
        assert!(last_loss.is_finite(), "loss went non-finite while overfitting");
        // A correct trainer memorizes 8 tokens easily; anything above a small
        // fraction of ln(vocab) means the gradient/optimizer path is broken.
        assert!(
            last_loss < 0.25 * random_loss,
            "model failed to overfit a single sequence: loss {:.4} did not drop \
             well below ln(vocab)={:.4} after 200 steps (started at {:.4}). \
             The backward pass or optimizer step is not actually learning.",
            last_loss, random_loss, first_loss
        );
        eprintln!("✓ model overfits a single sequence (loss {:.4} -> {:.4})", first_loss, last_loss);
    }

    // An all-ones mask must reproduce the unmasked loss exactly: same forward,
    // and active == s-1 makes the normalizer identical. This is the regression
    // guard that the masked path doesn't perturb pretraining numerics.
    #[test]
    fn test_train_step_masked_all_ones_matches_unmasked() {
        let (vocab, d, ctx, heads, ff_mult, layers) = (20, 8, 8, 2, 2, 2);
        let t = Model::init(d, vocab, ctx, heads, ff_mult, layers);
        let chunk: Vec<u32> = (0..8).map(|i| i as u32 % (vocab as u32)).collect();

        let mut ws_a = Workspace::new(&t, ctx as usize);
        let unmasked = pollster::block_on(t.train_step(&mut ws_a, &chunk, layers as usize));

        let mut ws_b = Workspace::new(&t, ctx as usize);
        let mask = vec![1u8; chunk.len()];
        let masked = pollster::block_on(t.train_step_masked(&mut ws_b, &chunk, &mask, layers as usize));

        eprintln!("  unmasked={:.6}  all-ones-masked={:.6}", unmasked, masked);
        assert!(
            (unmasked - masked).abs() < 1e-4,
            "all-ones mask changed the loss: {:.6} vs {:.6}",
            unmasked, masked
        );
    }

    // An all-zero mask trains on nothing: zero loss and zero gradient. Under
    // AdamW the optimizer step still applies DECOUPLED weight decay (a gradient-
    // independent shrink toward zero), so weights don't stay frozen — instead
    // every 2-D weight is scaled by exactly the same (1 - lr*wd) factor. That
    // uniform, gradient-free shrink is precisely what proves the mask zeroed the
    // gradient: any leaked gradient would perturb some weights off that ratio.
    #[test]
    fn test_train_step_masked_all_zero_trains_nothing() {
        let (vocab, d, ctx, heads, ff_mult, layers) = (20, 8, 8, 2, 2, 2);
        let t = Model::init(d, vocab, ctx, heads, ff_mult, layers);
        let mut ws = Workspace::new(&t, ctx as usize);

        let clone_of = |m: &Model| Model {
            embedding_table: m.embedding_table.clone(),
            max_seq_len: m.max_seq_len,
            wq: m.wq.clone(), wk: m.wk.clone(), wv: m.wv.clone(),
            wo: m.wo.clone(), w1: m.w1.clone(), w2: m.w2.clone(),
            b1: m.b1.clone(), b2: m.b2.clone(),
            ln_gamma: m.ln_gamma.clone(), ln_beta: m.ln_beta.clone(),
            ln_final_gamma: m.ln_final_gamma.clone(),
            ln_final_beta: m.ln_final_beta.clone(),
            w_output: m.w_output.clone(),
            heads: m.heads, head_dim: m.head_dim, layers: m.layers,
        };

        // Baseline: the GPU weights as uploaded (already f16-quantized). Comparing
        // against these — not the original f32 weights — isolates *training* change
        // from the harmless upload/sync quantization.
        let mut t_base = clone_of(&t);
        sync_weights_from_gpu(&ws, &mut t_base);
        let base_emb: Vec<f32> = t_base.embedding_table.iter().flatten().copied().collect();

        let chunk: Vec<u32> = (0..8).map(|i| i as u32 % (vocab as u32)).collect();
        let mask = vec![0u8; chunk.len()];
        let loss = pollster::block_on(t.train_step_masked(&mut ws, &chunk, &mask, layers as usize));
        assert_eq!(loss, 0.0, "an all-zero mask should yield zero loss");

        // lr and wd here mirror the call below and ADAM_WEIGHT_DECAY in
        // optimizer.rs (0.1). The embedding is a 2-D weight, so it decays.
        let (lr, wd) = (0.01_f32, 0.1_f32);
        gpu_adam_update(&ws, lr, 1, 1, layers as usize);
        let mut t_synced = clone_of(&t);
        sync_weights_from_gpu(&ws, &mut t_synced);
        let post_emb: Vec<f32> = t_synced.embedding_table.iter().flatten().copied().collect();

        // The ONLY movement must be the decoupled decay: post == base - lr*wd*base
        // (the kernel's exact arithmetic, since the Adam term is 0 with no
        // gradient). A tight per-weight tolerance catches any leaked gradient.
        for (b, p) in base_emb.iter().zip(&post_emb) {
            let expected = b - lr * (wd * b);
            assert!(
                (p - expected).abs() <= 1e-5 + 1e-4 * b.abs(),
                "masked step should apply weight decay only: got {p}, expected {expected} (base {b})"
            );
        }
        eprintln!("✓ all-zero mask trains nothing (loss=0, only decoupled weight decay applied)");
    }

    // ── Non-GPU tests: Model::init ──

    #[test]
    fn test_init_dimensions() {
        let t = Model::init(32, 100, 64, 4, 2, 2);
        assert_eq!(t.embedding_table.len(), 100);
        assert_eq!(t.embedding_table[0].len(), 32);
        assert_eq!(t.max_seq_len, 64, "ctx window must survive init");
        assert_eq!(t.layers, 2);
        assert_eq!(t.heads, 4);
        assert_eq!(t.head_dim, 8);
        assert_eq!(t.wq.len(), 2);
        assert_eq!(t.wq[0].len(), 4);
        assert_eq!(t.wq[0][0].len(), 32);
        assert_eq!(t.wq[0][0][0].len(), 8);
        assert_eq!(t.wo.len(), 2);
        assert_eq!(t.wo[0].len(), 32);
        assert_eq!(t.wo[0][0].len(), 32);
        let ff = 32 * 2;
        assert_eq!(t.w1[0].len(), 32);
        assert_eq!(t.w1[0][0].len(), ff);
        assert_eq!(t.w2[0].len(), ff);
        assert_eq!(t.w2[0][0].len(), 32);
        assert_eq!(t.b1[0].len(), ff);
        assert!(t.b1[0].iter().all(|&v| v == 0.0), "b1 should be zero-initialized");
        assert!(t.b2[0].iter().all(|&v| v == 0.0), "b2 should be zero-initialized");
        assert!(t.ln_gamma[0].iter().all(|&v| v == 1.0), "ln_gamma should be one-initialized");
        assert!(t.ln_beta[0].iter().all(|&v| v == 0.0), "ln_beta should be zero-initialized");
        // A fresh model ties, so there is no separate output projection: the
        // embedding table is both ends.
        assert!(t.ties_embeddings());
        assert!(t.w_output.is_empty());
        assert_eq!(t.embedding_table.len(), 100);
        assert_eq!(t.embedding_table[0].len(), 32);
    }

    #[test]
    fn test_init_single_layer() {
        let t = Model::init(16, 50, 32, 2, 4, 1);
        assert_eq!(t.layers, 1);
        assert_eq!(t.wq.len(), 1);
        assert_eq!(t.wq[0].len(), 2);
        let ff = 16 * 4;
        assert_eq!(t.w1[0][0].len(), ff);
        assert_eq!(t.b1[0].len(), ff);
    }

    // ── Non-GPU tests: serialization round-trips ──

    // The tie must survive a save/load, and it is represented by an *absence* —
    // the easiest kind of field to lose in a serializer. A model that came back
    // untied would silently project through a matrix of zeros.
    #[test]
    fn tying_survives_a_round_trip() {
        let t1 = Model::init(16, 50, 32, 2, 2, 1);
        assert!(t1.ties_embeddings());

        let t2 = Model::from_bytes(&t1.to_bytes().unwrap()).unwrap();
        assert!(t2.ties_embeddings(), "a tied model came back untied");
        assert!(t2.w_output.is_empty());
        assert_eq!(t1.embedding_table, t2.embedding_table);
        assert_eq!(t1.vocab(), t2.vocab());

        // And the same through the layer-rows form, which is a separate encoder
        // (used for imported checkpoints) and so a second place to lose the tie.
        let header = ModelHeader {
            dimensions: 16,
            vocab_size: 50,
            max_seq_len: 32,
            heads: 2,
            ff_multiplier: 2,
            layers: 1,
        };
        let t3 = Model::from_layer_rows(&header, &t1.to_layer_rows()).unwrap();
        assert!(t3.ties_embeddings(), "the layer-rows form lost the tie");
        assert_eq!(t1.embedding_table, t3.embedding_table);
    }

    // An untied model — the pre-v6 shape — must keep its own projection rather
    // than being silently converted. Those weights trained against it.
    #[test]
    fn an_untied_model_keeps_its_own_projection() {
        let mut t1 = Model::init(16, 50, 32, 2, 2, 1);
        let (vocab, d) = (t1.vocab(), 16usize);
        t1.w_output = vec![vec![0.0f32; vocab]; d];
        for v in 0..vocab {
            for j in 0..d {
                t1.w_output[j][v] = t1.embedding_table[v][j] * 0.5;
            }
        }
        assert!(!t1.ties_embeddings());

        // `write_model` stamps the current version, which declares a tie, so the
        // consistency check in `read_model` must reject this rather than load a
        // model whose projection would then be ignored.
        let err = match Model::from_bytes(&t1.to_bytes().unwrap()) {
            Err(e) => e,
            Ok(_) => panic!("a v6 file carrying an output projection must not load"),
        };
        assert!(
            err.to_string().contains("output projection"),
            "expected the version/tie consistency check to fire, got: {err}"
        );
    }

    #[test]
    fn test_to_bytes_from_bytes_roundtrip() {
        let t1 = Model::init(16, 50, 32, 2, 2, 1);
        let bytes = t1.to_bytes().unwrap();
        let t2 = Model::from_bytes(&bytes).unwrap();

        assert_eq!(t1.embedding_table, t2.embedding_table);
        assert_eq!(t1.wq, t2.wq);
        assert_eq!(t1.wk, t2.wk);
        assert_eq!(t1.wv, t2.wv);
        assert_eq!(t1.wo, t2.wo);
        assert_eq!(t1.w1, t2.w1);
        assert_eq!(t1.w2, t2.w2);
        assert_eq!(t1.b1, t2.b1);
        assert_eq!(t1.b2, t2.b2);
        assert_eq!(t1.ln_gamma, t2.ln_gamma);
        assert_eq!(t1.ln_beta, t2.ln_beta);
        assert_eq!(t1.w_output, t2.w_output);
        assert_eq!(t1.heads, t2.heads);
        assert_eq!(t1.layers, t2.layers);
        assert_eq!(t1.head_dim, t2.head_dim);
    }

    #[test]
    fn test_to_bytes_from_bytes_multi_layer_roundtrip() {
        let t1 = Model::init(32, 100, 64, 4, 2, 3);
        let bytes = t1.to_bytes().unwrap();
        let t2 = Model::from_bytes(&bytes).unwrap();

        for l in 0..3 {
            assert_eq!(t1.wq[l], t2.wq[l], "wq layer {} mismatch", l);
            assert_eq!(t1.wk[l], t2.wk[l], "wk layer {} mismatch", l);
            assert_eq!(t1.wv[l], t2.wv[l], "wv layer {} mismatch", l);
            assert_eq!(t1.wo[l], t2.wo[l], "wo layer {} mismatch", l);
            assert_eq!(t1.w1[l], t2.w1[l], "w1 layer {} mismatch", l);
            assert_eq!(t1.w2[l], t2.w2[l], "w2 layer {} mismatch", l);
            assert_eq!(t1.b1[l], t2.b1[l], "b1 layer {} mismatch", l);
            assert_eq!(t1.b2[l], t2.b2[l], "b2 layer {} mismatch", l);
            assert_eq!(t1.ln_gamma[l], t2.ln_gamma[l], "ln_gamma layer {} mismatch", l);
            assert_eq!(t1.ln_beta[l], t2.ln_beta[l], "ln_beta layer {} mismatch", l);
        }
        assert_eq!(t1.embedding_table, t2.embedding_table);
        assert_eq!(t1.w_output, t2.w_output);
    }

    #[test]
    fn test_from_bytes_rejects_bad_magic() {
        let bad = b"\xFF\xFF\xFF\xFF\x03\x00\x00\x00";
        assert!(Model::from_bytes(bad).is_err());
    }

    #[test]
    fn test_from_bytes_rejects_truncated() {
        let bad = b"GORK\x03\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00";
        assert!(Model::from_bytes(bad).is_err());
    }

    #[test]
    fn test_to_layer_rows_from_layer_rows_roundtrip() {
        let t1 = Model::init(16, 50, 32, 2, 2, 1);
        let header = ModelHeader {
            dimensions: 16,
            vocab_size: 50,
            max_seq_len: 32,
            heads: 2,
            ff_multiplier: 2,
            layers: 1,
        };
        let rows = t1.to_layer_rows();
        let t2 = Model::from_layer_rows(&header, &rows).unwrap();

        assert_eq!(t1.embedding_table, t2.embedding_table);
        assert_eq!(t1.wq, t2.wq);
        assert_eq!(t1.wk, t2.wk);
        assert_eq!(t1.wv, t2.wv);
        assert_eq!(t1.wo, t2.wo);
        assert_eq!(t1.w1, t2.w1);
        assert_eq!(t1.w2, t2.w2);
        assert_eq!(t1.b1, t2.b1);
        assert_eq!(t1.b2, t2.b2);
        assert_eq!(t1.ln_gamma, t2.ln_gamma);
        assert_eq!(t1.ln_beta, t2.ln_beta);
        assert_eq!(t1.w_output, t2.w_output);
    }

    #[test]
    fn test_to_layer_rows_from_layer_rows_multi_layer() {
        let t1 = Model::init(32, 100, 64, 4, 2, 3);
        let header = ModelHeader {
            dimensions: 32,
            vocab_size: 100,
            max_seq_len: 64,
            heads: 4,
            ff_multiplier: 2,
            layers: 3,
        };
        let rows = t1.to_layer_rows();
        let t2 = Model::from_layer_rows(&header, &rows).unwrap();

        for l in 0..3 {
            assert_eq!(t1.wq[l], t2.wq[l], "wq layer {} mismatch", l);
            assert_eq!(t1.wk[l], t2.wk[l], "wk layer {} mismatch", l);
            assert_eq!(t1.wv[l], t2.wv[l], "wv layer {} mismatch", l);
            assert_eq!(t1.wo[l], t2.wo[l], "wo layer {} mismatch", l);
            assert_eq!(t1.w1[l], t2.w1[l], "w1 layer {} mismatch", l);
            assert_eq!(t1.w2[l], t2.w2[l], "w2 layer {} mismatch", l);
        }
        assert_eq!(t1.embedding_table, t2.embedding_table);
        assert_eq!(t1.w_output, t2.w_output);
    }

    #[test]
    fn test_to_layer_rows_correct_names() {
        let t = Model::init(8, 20, 16, 2, 2, 2);
        let rows = t.to_layer_rows();
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"embedding"));
        assert!(names.contains(&"unembed"));
        assert!(names.contains(&"h.0.attn.wq"));
        assert!(names.contains(&"h.0.attn.wk"));
        assert!(names.contains(&"h.0.attn.wv"));
        assert!(names.contains(&"h.0.attn.wo"));
        assert!(names.contains(&"h.0.ffn.w1"));
        assert!(names.contains(&"h.0.ffn.w2"));
        assert!(names.contains(&"h.0.ffn.b1"));
        assert!(names.contains(&"h.0.ffn.b2"));
        assert!(names.contains(&"h.0.ln.gamma"));
        assert!(names.contains(&"h.0.ln.beta"));
        assert!(names.contains(&"h.1.attn.wq"));
        assert!(names.contains(&"h.1.attn.wk"));
        assert!(names.contains(&"h.1.attn.wv"));
        assert!(names.contains(&"h.1.attn.wo"));
        assert!(names.contains(&"h.1.ffn.w1"));
        assert!(names.contains(&"h.1.ffn.w2"));
        assert!(names.contains(&"h.1.ffn.b1"));
        assert!(names.contains(&"h.1.ffn.b2"));
        assert!(names.contains(&"h.1.ln.gamma"));
        assert!(names.contains(&"h.1.ln.beta"));
        // 22, not 23: the "positional" row went away with the absolute
        // positional table when RoPE replaced it.
        assert_eq!(names.len(), 22);
    }

    #[test]
    fn test_from_layer_rows_rejects_missing_layer() {
        let header = ModelHeader {
            dimensions: 8,
            vocab_size: 20,
            max_seq_len: 16,
            heads: 2,
            ff_multiplier: 2,
            layers: 1,
        };
        let rows = vec![
            LayerRow { name: "embedding".into(), shape: vec![20, 8], data: vec![0u8; 20 * 8 * 4] },
        ];
        assert!(Model::from_layer_rows(&header, &rows).is_err());
    }
}
