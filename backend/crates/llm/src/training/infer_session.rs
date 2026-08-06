//! Device-resident inference state — the forward-only counterpart of
//! [`Workspace`](crate::training::gpu_workspace::Workspace).
//!
//! # Why this exists
//!
//! The old inference path (`Model::forward_seq` and its per-stage helpers) held
//! nothing on the device between calls. Every layer norm, every attention head,
//! every feed-forward and the whole `[d, vocab]` output projection were flattened
//! on the host, converted f32→f16 element by element, `cudaMalloc`'d, uploaded,
//! launched, and copied straight back to host `Vec<Vec<f32>>` — once per
//! *generated token*. Measured at d256/L6/h4/ctx512, 86% of a token was host
//! marshalling and copies and 14% was kernels, which is what a GPU sitting at low
//! utilization during generation means. It was also ~1000 launches and ~250
//! malloc/free pairs per token.
//!
//! Training solved this in commit `6048702`; this is the same answer applied to
//! the forward pass. Weights are uploaded **once**, activations never leave the
//! device, and the only device→host copy per call is the logits.
//!
//! # This must agree with the training forward
//!
//! There are two forward passes in this crate and they have to compute the same
//! function — a model is trained under one and served under the other. The old
//! inference path silently did **not**: it accumulated the residual stream in f32
//! on the host where training accumulates in f16 on the device, and it skipped
//! both the post-embedding and post-residual `cuda_clamp_f16` calls training
//! applies. That is a real divergence between training and serving, and removing
//! it is half the point of this file.
//!
//! So `prefill` mirrors `transformer::forward`'s stage sequence exactly —
//! `attn_sublayer_forward`, `ffn_sublayer_forward`, `final_ln_forward`,
//! `project_logits` — using the identical kernels in the identical order, minus
//! only the activation caching the backward needs. `tests/infer_agrees_with_
//! training_forward.rs` is the guard. **If you change one, change the other.**

use std::ffi::c_void;

use crate::training::device_buffer::{Buffer, DeviceBufferF16};
use crate::training::transformer::{LAYER_NORM_EPS, Model, infer_end, infer_start};

unsafe extern "C" {
    fn cublas_init();
    fn cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize);
    fn cuda_infer_reserve(seq: i32, hd: i32, d: i32, h: i32, d_ff: i32);
    fn cuda_embedding_forward(
        token_ids: *const i32,
        embedding_table: *const c_void,
        output: *mut c_void,
        s: i32,
        d: i32,
        vocab: i32,
    );
    fn cuda_clamp_f16(buf: *mut c_void, n: i32);
    fn cuda_layer_norm(
        input: *const c_void,
        out: *mut c_void,
        gamma: *const c_void,
        beta: *const c_void,
        rows: i32,
        cols: i32,
        eps: f32,
    );
    fn cuda_attention_forward_batched(
        x: *const c_void,
        wq_all: *const c_void,
        wk_all: *const c_void,
        wv_all: *const c_void,
        out_all: *mut c_void,
        p_all: *mut c_void,
        q_all: *mut c_void,
        k_all: *mut c_void,
        v_all: *mut c_void,
        k_cache_f32: *mut f32,
        kv_stride: i32,
        seq: i32,
        d: i32,
        hd: i32,
        h: i32,
    );
    fn cuda_attention_decode(
        x_row: *const c_void,
        wq_all: *const c_void,
        wk_all: *const c_void,
        wv_all: *const c_void,
        k_cache: *mut f32,
        v_cache: *mut c_void,
        out: *mut c_void,
        d: i32,
        hd: i32,
        h: i32,
        pos: i32,
        kv_stride: i32,
    );
    fn cuda_concat_heads(
        head_ptrs: *mut *mut c_void,
        out: *mut c_void,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
    );
    fn cuda_matmul(
        a: *const c_void,
        b: *const c_void,
        result: *mut c_void,
        rows_a: i32,
        cols_a: i32,
        cols_b: i32,
    );
    fn cuda_add_inplace_f16(dst: *mut c_void, src: *const c_void, n: i32);
    fn cuda_feed_forward(
        input: *const c_void,
        w1: *const c_void,
        w2: *const c_void,
        b1: *const c_void,
        b2: *const c_void,
        out: *mut c_void,
        seq_len: i32,
        d_model: i32,
        d_ff: i32,
    );
}

pub struct InferSession {
    // ── Weights, f16, uploaded once in `new` and never touched again ──
    // Per-head QKV are contiguous `[h, d, hd]`, the layout
    // `cuda_attention_forward_batched` strides over. Packed by
    // `flatten_qkv_heads`, shared with `Workspace::upload_weights` so the packing
    // has one definition.
    d_wq_all: Vec<DeviceBufferF16>, // [layers][h * d * hd]
    d_wk_all: Vec<DeviceBufferF16>,
    d_wv_all: Vec<DeviceBufferF16>,
    d_wo: Vec<DeviceBufferF16>, // [layers][d * d]
    d_w1: Vec<DeviceBufferF16>, // [layers][d * ff]
    d_w2: Vec<DeviceBufferF16>, // [layers][ff * d]
    d_b1: Vec<DeviceBufferF16>,
    d_b2: Vec<DeviceBufferF16>,
    d_ln_gamma: Vec<DeviceBufferF16>,
    d_ln_beta: Vec<DeviceBufferF16>,
    d_ln_final_gamma: DeviceBufferF16,
    d_ln_final_beta: DeviceBufferF16,
    d_w_output: DeviceBufferF16,  // [d * vocab]
    d_embedding: DeviceBufferF16, // [vocab * d]

    // ── Activation scratch, allocated at `max_s` once ──
    // Every buffer a kernel fills with an `[s, d]` block is sized at `max_s`, for
    // the reason `Workspace::assert_scratch_fits` exists: an undersized device
    // write does not fault, it silently corrupts whatever sits next to it.
    d_x: DeviceBufferF16,           // [max_s, d] the residual stream
    d_norm: DeviceBufferF16,        // [max_s, d] layer-norm output
    d_attn_concat: DeviceBufferF16, // [max_s, d] heads interleaved by token
    d_sub_out: DeviceBufferF16,     // [max_s, d] sublayer output before the residual
    d_final: DeviceBufferF16,       // [max_s, d] final-norm output
    d_head_out: DeviceBufferF16,    // [h, max_s, hd] per-head attention output
    d_p: DeviceBufferF16,           // [h, max_s, max_s] softmax scores
    d_head_ptrs: Buffer,            // [h] device pointers for cuda_concat_heads
    d_token_ids: Buffer,            // [max_s] i32
    d_logits: DeviceBufferF16,      // [max_s, vocab]

    // ── The KV cache ──
    //
    // One per layer, `[h, max_s, hd]`, holding the projections for every position
    // already forwarded. K is f32 and V is f16 because that is what the prefill's
    // GEMMs natively produce — see `cuda_attention_decode` on why caching K as f16
    // would make decode disagree with prefill.
    //
    // The cached K is post-RoPE, so a cached row is only valid at the position it
    // was written for. `cached_tokens` is what makes that safe.
    d_k_cache: Vec<Buffer>,          // [layers][h * max_s * hd] f32
    d_v_cache: Vec<DeviceBufferF16>, // [layers][h * max_s * hd] f16

    /// The windowed token sequence the cache currently describes, position for
    /// position. Reuse is decided by comparing a new sequence's prefix against this.
    ///
    /// Holding the *windowed* sequence rather than the caller's full one is what
    /// makes a moving context window safe without tracking an offset. K and V at
    /// position `i` are a function of the windowed tokens `0..=i` and of `i` itself
    /// (through RoPE), and nothing else — so if two windowed sequences agree on a
    /// prefix, the cached rows for that prefix are exactly right for both, however
    /// the window moved to produce them. If the window shifted, the comparison
    /// simply fails early and the sequence is re-prefilled.
    cached_tokens: Vec<u32>,

    /// Every token this session has actually pushed through the stack: `s` per
    /// prefill, one per decode step. It is the work done, not the work asked for —
    /// a prompt served from the KV cache costs nothing here, and a sequence scored
    /// in two overlapping windows costs both windows. That is the honest reading,
    /// and it is what the benchmark reports as the cost of a run.
    tokens_forwarded: u64,

    // ── Dimensions ──
    max_s: usize,
    d: usize,
    h: usize,
    hd: usize,
    ff: usize,
    vocab: usize,
    layers: usize,
    has_final_ln: bool,
}

// SAFETY: every field is either a plain integer or a device allocation addressed
// by a raw pointer. A CUDA device pointer belongs to the process's primary
// context, not to the thread that allocated it: any thread may use or free it, so
// moving a session between threads is sound. What is *not* sound is two threads
// using one concurrently — and neither is two sessions running concurrently at
// all, because the `attn`/`ff` scratch in shaders.cu is file-static and the cuBLAS
// handle is shared (see `backend/CLAUDE.md`).
//
// That serialization is not provided by this impl; it comes from the GPU queue,
// which is the only thing that starts device work and runs one job at a time, and
// from `infer::MODEL_CACHE`'s mutex, which is what needs the `Send` bound in the
// first place. `Sync` is deliberately NOT implemented: a shared `&InferSession`
// handed to two threads would be exactly the unserialized case.
unsafe impl Send for InferSession {}

/// Pack a layer's per-head QKV weights into the contiguous `[h, d, hd]` layout
/// the batched attention kernels stride over.
///
/// Shared with `Workspace::upload_weights`: both forwards read the same packing
/// from the same `Model`, so it is defined once here rather than transcribed
/// twice. A transposition here is not a crash, it is a model that serves
/// differently than it trained.
pub fn flatten_qkv_heads(per_head: &[Vec<Vec<f32>>], h: usize, d: usize, hd: usize) -> Vec<f32> {
    let mut flat = Vec::with_capacity(h * d * hd);
    for hi in 0..h {
        flat.extend(per_head[hi].iter().flatten());
    }
    debug_assert_eq!(flat.len(), h * d * hd);
    flat
}

/// Row-major flatten of a 2-D weight.
fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

impl InferSession {
    /// Allocate device state for `t` at a context window of `ctx_window` and
    /// upload every weight. Cost is paid once per loaded model, not per token.
    pub fn new(t: &Model, ctx_window: usize) -> Self {
        let max_s = ctx_window;
        let d = t.wo[0].len();
        let h = t.heads as usize;
        let hd = t.head_dim as usize;
        let ff = t.w1[0][0].len();
        let vocab = t.w_output[0].len();
        let layers = t.layers as usize;

        // SAFETY: no arguments; initializes the process-wide cuBLAS handle. The
        // GPU queue serializes every user of the device (one GPU, one runner), so
        // no two sessions or workspaces are constructed concurrently.
        unsafe { cublas_init() };

        let session = InferSession {
            d_wq_all: (0..layers)
                .map(|_| DeviceBufferF16::alloc(h * d * hd))
                .collect(),
            d_wk_all: (0..layers)
                .map(|_| DeviceBufferF16::alloc(h * d * hd))
                .collect(),
            d_wv_all: (0..layers)
                .map(|_| DeviceBufferF16::alloc(h * d * hd))
                .collect(),
            d_wo: (0..layers).map(|_| DeviceBufferF16::alloc(d * d)).collect(),
            d_w1: (0..layers).map(|_| DeviceBufferF16::alloc(d * ff)).collect(),
            d_w2: (0..layers).map(|_| DeviceBufferF16::alloc(ff * d)).collect(),
            d_b1: (0..layers).map(|_| DeviceBufferF16::alloc(ff)).collect(),
            d_b2: (0..layers).map(|_| DeviceBufferF16::alloc(d)).collect(),
            d_ln_gamma: (0..layers).map(|_| DeviceBufferF16::alloc(d)).collect(),
            d_ln_beta: (0..layers).map(|_| DeviceBufferF16::alloc(d)).collect(),
            d_ln_final_gamma: DeviceBufferF16::alloc(d),
            d_ln_final_beta: DeviceBufferF16::alloc(d),
            d_w_output: DeviceBufferF16::alloc(d * vocab),
            d_embedding: DeviceBufferF16::alloc(vocab * d),

            d_x: DeviceBufferF16::alloc(max_s * d),
            d_norm: DeviceBufferF16::alloc(max_s * d),
            d_attn_concat: DeviceBufferF16::alloc(max_s * d),
            d_sub_out: DeviceBufferF16::alloc(max_s * d),
            d_final: DeviceBufferF16::alloc(max_s * d),
            d_head_out: DeviceBufferF16::alloc(h * max_s * hd),
            d_p: DeviceBufferF16::alloc(h * max_s * max_s),
            d_head_ptrs: Buffer::alloc(h * 2),
            d_token_ids: Buffer::alloc(max_s),
            d_logits: DeviceBufferF16::alloc(max_s * vocab),

            d_k_cache: (0..layers).map(|_| Buffer::alloc(h * max_s * hd)).collect(),
            d_v_cache: (0..layers)
                .map(|_| DeviceBufferF16::alloc(h * max_s * hd))
                .collect(),
            cached_tokens: Vec::new(),
            tokens_forwarded: 0,

            max_s,
            d,
            h,
            hd,
            ff,
            vocab,
            layers,
            has_final_ln: t.has_final_ln(),
        };

        // Size the shared attention/feed-forward scratch for the longest sequence
        // this session can ever run, once. `ensure_attn` only ever grows, and the
        // old path let it grow one token at a time: a generating sequence gets one
        // longer every step, so the guard missed every time and all 11 buffers
        // were freed and re-malloc'd per token — ~9 ms/token of pure allocator
        // churn at a short prompt, measured.
        //
        // SAFETY: plain integer arguments; allocates the file-static scratch in
        // shaders.cu, which is idempotent and grow-only.
        unsafe { cuda_infer_reserve(max_s as i32, hd as i32, d as i32, h as i32, ff as i32) };

        session.upload_weights(t);
        session
    }

    fn upload_weights(&self, t: &Model) {
        let (d, h, hd) = (self.d, self.h, self.hd);
        for l in 0..self.layers {
            self.d_wq_all[l].upload_f32_slice(&flatten_qkv_heads(&t.wq[l], h, d, hd));
            self.d_wk_all[l].upload_f32_slice(&flatten_qkv_heads(&t.wk[l], h, d, hd));
            self.d_wv_all[l].upload_f32_slice(&flatten_qkv_heads(&t.wv[l], h, d, hd));
            self.d_wo[l].upload_f32_slice(&flatten(&t.wo[l]));
            self.d_w1[l].upload_f32_slice(&flatten(&t.w1[l]));
            self.d_w2[l].upload_f32_slice(&flatten(&t.w2[l]));
            self.d_b1[l].upload_f32_slice(&t.b1[l]);
            self.d_b2[l].upload_f32_slice(&t.b2[l]);
            self.d_ln_gamma[l].upload_f32_slice(&t.ln_gamma[l]);
            self.d_ln_beta[l].upload_f32_slice(&t.ln_beta[l]);
        }
        // Skipped for a pre-v5 model exactly as `Workspace::upload_weights` skips
        // it: `has_final_ln` gates every read, and synthesizing a gamma=1 norm
        // would change what such a model outputs (a LayerNorm at gamma=1/beta=0
        // is not the identity).
        if self.has_final_ln {
            self.d_ln_final_gamma.upload_f32_slice(&t.ln_final_gamma);
            self.d_ln_final_beta.upload_f32_slice(&t.ln_final_beta);
        }
        self.d_w_output.upload_f32_slice(&flatten(&t.w_output));
        self.d_embedding
            .upload_f32_slice(&flatten(&t.embedding_table));
    }

    /// The longest sequence this session can forward — the context window it was
    /// built at.
    pub fn ctx_window(&self) -> usize {
        self.max_s
    }


    /// Vocabulary size — the row stride of what `logits_all` returns.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// How many tokens have gone through the stack since this session was built.
    ///
    /// One session per benchmark run, so this reads as the cost of the whole run
    /// — see the field's note for what it does and does not count.
    pub fn tokens_forwarded(&self) -> u64 {
        self.tokens_forwarded
    }

    /// Forward `tokens` and return the logits for the **last** position only.
    ///
    /// What generation wants. The vocab GEMM runs at `1 × d × vocab` rather than
    /// `s × d × vocab`, and only one row comes back to the host.
    pub fn logits_last(&mut self, tokens: &[u32]) -> Vec<f32> {
        let Some(row) = self.forward_incremental(tokens) else {
            return Vec::new();
        };
        let hidden = self.final_hidden_ptr();
        // SAFETY: `hidden` addresses at least row+1 rows of d f16 values (every
        // activation buffer is allocated at max_s, and `row` indexes a row the
        // forward just wrote), so offsetting to it stays in bounds.
        let last_row = unsafe { (hidden as *const u8).add(row * self.d * 2) as *const c_void };
        self.project_logits(last_row, 1);
        self.read_logits(1)
    }

    /// Forward `tokens` and return the logits for **every** position, as a flat
    /// `[s * vocab]` buffer where row `j` is `[j * vocab .. (j + 1) * vocab]`.
    ///
    /// What teacher-forced scoring wants. Flat rather than `Vec<Vec<f32>>` on
    /// purpose: a `Vec` per position was a measurable part of the old path's cost
    /// and every caller either borrows rows or reduces them.
    /// Always a full prefill: teacher-forced scoring wants the hidden state at every
    /// position, and only a prefill leaves all of them on the device. There is
    /// nothing for the KV cache to save here — a scoring window is one forward over
    /// a fixed span, not an incremental walk.
    pub fn logits_all(&mut self, tokens: &[u32]) -> Vec<f32> {
        let seq = self.window(tokens).to_vec();
        let s = self.prefill(&seq);
        if s == 0 {
            return Vec::new();
        }
        let hidden = self.final_hidden_ptr();
        self.project_logits(hidden as *const c_void, s);
        self.read_logits(s)
    }

    // Only the last `max_s` tokens are used, so a growing prompt is fine — RoPE is
    // relative, so the window sliding forward simply re-bases the positions.
    fn window<'a>(&self, tokens: &'a [u32]) -> &'a [u32] {
        if tokens.len() > self.max_s {
            &tokens[tokens.len() - self.max_s..]
        } else {
            tokens
        }
    }

    // Forward the whole sequence, filling the KV cache as a side effect. Leaves the
    // final hidden state for every position on the device.
    fn prefill(&mut self, seq: &[u32]) -> usize {
        let s = seq.len();
        if s == 0 {
            return 0;
        }
        self.embed(seq);
        for l in 0..self.layers {
            self.attn_sublayer(s, l);
            self.ffn_sublayer(s, l);
        }
        self.final_ln(s);
        self.cached_tokens = seq.to_vec();
        self.tokens_forwarded += s as u64;
        s
    }

    // How many leading positions of `seq` the cache already holds correctly.
    fn shared_prefix(&self, seq: &[u32]) -> usize {
        self.cached_tokens
            .iter()
            .zip(seq.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    // Forward `tokens`, reusing the KV cache where it applies, and leave the hidden
    // state for the last position on the device. Returns the **row index within the
    // activation buffers** holding it — `s - 1` after a prefill, but `0` after a
    // decode step, which only ever puts one row there — or None if there was
    // nothing to run.
    //
    // Three cases, and the split is deliberately coarse:
    //
    //   * **one new token on the end** — the generation hot path. One decode step:
    //     project one row, append it to the cache, score it against the history.
    //     O(s) instead of the prefill's O(s²).
    //   * **the cache already covers everything** — re-decode the final position.
    //     This is what makes N candidates from one prompt share a prefill: the
    //     second candidate arrives with the prompt fully cached and pays one decode
    //     step rather than a whole forward.
    //   * **anything else** — re-prefill. A new chat turn adds many tokens at once,
    //     and a batched prefill is the right tool for that; decoding them one at a
    //     time would be many small launches instead of a few big GEMMs. A shifted
    //     context window also lands here, because its windowed sequence no longer
    //     shares a prefix.
    fn forward_incremental(&mut self, tokens: &[u32]) -> Option<usize> {
        let seq = self.window(tokens).to_vec();
        let n = seq.len();
        if n == 0 {
            return None;
        }

        let shared = self.shared_prefix(&seq);
        // A decode step needs the cache to hold exactly the positions before the one
        // it is about to write, so reuse is capped at `n - 1`: the last position has
        // to be (re)computed to produce logits for it.
        let reusable = shared.min(n - 1);

        if reusable == n - 1 && reusable > 0 {
            self.cached_tokens.truncate(reusable);
            self.decode(&seq, n - 1);
            Some(0)
        } else {
            self.prefill(&seq);
            Some(n - 1)
        }
    }

    // One token through the stack against the cache. `pos` is its absolute position
    // in the windowed sequence, and the cache must already hold positions `0..pos`.
    fn decode(&mut self, seq: &[u32], pos: usize) {
        debug_assert_eq!(self.cached_tokens.len(), pos);
        self.embed(&seq[pos..=pos]);
        for l in 0..self.layers {
            self.attn_sublayer_decode(pos, l);
            self.ffn_sublayer(1, l);
        }
        self.final_ln(1);
        self.cached_tokens.push(seq[pos]);
        self.tokens_forwarded += 1;
    }

    // Token ids → the residual stream, on the device.
    //
    // The clamp is not optional: it is what the training forward does
    // (`forward.rs`'s `embed_tokens`), and an f16 residual stream that starts
    // unclamped diverges from it.
    fn embed(&mut self, seq: &[u32]) {
        let timer = infer_start();
        let ids: Vec<i32> = seq.iter().map(|&t| t as i32).collect();
        let s = seq.len();
        // SAFETY: `d_token_ids` holds max_s slots of 4 bytes and s <= max_s;
        // `ids` is a live slice of s i32s. `d_embedding` holds vocab*d f16 and
        // `d_x` holds max_s*d f16, both matching the dimensions passed.
        unsafe {
            cuda_memcpy_h2d(
                self.d_token_ids.ptr as *mut c_void,
                ids.as_ptr() as *const c_void,
                s * size_of::<i32>(),
            );
            cuda_embedding_forward(
                self.d_token_ids.ptr as *const i32,
                self.d_embedding.ptr as *const c_void,
                self.d_x.ptr as *mut c_void,
                s as i32,
                self.d as i32,
                self.vocab as i32,
            );
            cuda_clamp_f16(self.d_x.ptr as *mut c_void, (s * self.d) as i32);
        }
        infer_end("embed", timer);
    }

    // LN1 → batched multi-head attention → concat → wo → residual.
    //
    // Mirrors `attn_sublayer_forward` minus the activation caching. The residual
    // add is in place on `d_x` because, unlike the training forward, there is no
    // backward that needs the pre-residual value.
    fn attn_sublayer(&mut self, s: usize, l: usize) {
        let (d, h, hd) = (self.d, self.h, self.hd);

        let t = infer_start();
        // SAFETY (whole block): every buffer is allocated at max_s >= s in the
        // element type its kernel writes, and the dimensions passed match the
        // allocations made in `new`. `d_head_ptrs` holds h pointers.
        unsafe {
            cuda_layer_norm(
                self.d_x.ptr as *const c_void,
                self.d_norm.ptr as *mut c_void,
                self.d_ln_gamma[l].ptr as *const c_void,
                self.d_ln_beta[l].ptr as *const c_void,
                s as i32,
                d as i32,
                LAYER_NORM_EPS,
            );

            // The f16 Q/K caches are null: `rope_qk_kernel` null-checks them, and
            // there is no backward here to feed. The *inference* caches are the last
            // two arguments — V is written straight through the `V_all` parameter
            // (which is the V cache, at the cache's pitch), and K arrives in
            // `d_k_cache` post-RoPE. So a prefill leaves the cache warm for free.
            cuda_attention_forward_batched(
                self.d_norm.ptr as *const c_void,
                self.d_wq_all[l].ptr as *const c_void,
                self.d_wk_all[l].ptr as *const c_void,
                self.d_wv_all[l].ptr as *const c_void,
                self.d_head_out.ptr as *mut c_void,
                self.d_p.ptr as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                self.d_v_cache[l].ptr as *mut c_void,
                self.d_k_cache[l].ptr,
                self.max_s as i32,
                s as i32,
                d as i32,
                hd as i32,
                h as i32,
            );

            // Heads are laid out [h, s, hd] and the wo GEMM wants [s, h*hd], so
            // they have to be interleaved by token. Same device-side concat the
            // training forward uses; the old inference path did this on the CPU.
            let base = self.d_head_out.ptr as *mut u8;
            let head_ptrs: Vec<*mut c_void> = (0..h)
                .map(|hi| base.add(hi * s * hd * 2) as *mut c_void)
                .collect();
            cuda_memcpy_h2d(
                self.d_head_ptrs.ptr as *mut c_void,
                head_ptrs.as_ptr() as *const c_void,
                head_ptrs.len() * size_of::<*mut c_void>(),
            );
            cuda_concat_heads(
                self.d_head_ptrs.ptr as *mut *mut c_void,
                self.d_attn_concat.ptr as *mut c_void,
                s as i32,
                h as i32,
                hd as i32,
            );

            cuda_matmul(
                self.d_attn_concat.ptr as *const c_void,
                self.d_wo[l].ptr as *const c_void,
                self.d_sub_out.ptr as *mut c_void,
                s as i32,
                (h * hd) as i32,
                d as i32,
            );

            cuda_add_inplace_f16(
                self.d_x.ptr as *mut c_void,
                self.d_sub_out.ptr as *const c_void,
                (s * d) as i32,
            );
            cuda_clamp_f16(self.d_x.ptr as *mut c_void, (s * d) as i32);
        }
        infer_end("attn sublayer", t);
    }

    // The decode counterpart of `attn_sublayer`: one token at absolute position
    // `pos`, attending to the cache.
    //
    // Structurally the same LN → attention → wo → residual, with two differences.
    // The attention is `cuda_attention_decode` rather than the batched prefill, and
    // there is **no head concat**: the decode kernel writes `[h, 1, hd]`
    // contiguously, and that is bit-for-bit the same buffer as the `[1, d]` row the
    // wo GEMM wants. The prefill needs a concat only because its `[h, s, hd]` has to
    // be interleaved by token.
    fn attn_sublayer_decode(&mut self, pos: usize, l: usize) {
        let (d, h, hd) = (self.d, self.h, self.hd);
        let timer = infer_start();
        // SAFETY: as `attn_sublayer`, with one row instead of s. `pos < max_s`
        // because the windowed sequence is at most max_s long, so the cache row the
        // kernel writes is in bounds.
        unsafe {
            cuda_layer_norm(
                self.d_x.ptr as *const c_void,
                self.d_norm.ptr as *mut c_void,
                self.d_ln_gamma[l].ptr as *const c_void,
                self.d_ln_beta[l].ptr as *const c_void,
                1,
                d as i32,
                LAYER_NORM_EPS,
            );

            cuda_attention_decode(
                self.d_norm.ptr as *const c_void,
                self.d_wq_all[l].ptr as *const c_void,
                self.d_wk_all[l].ptr as *const c_void,
                self.d_wv_all[l].ptr as *const c_void,
                self.d_k_cache[l].ptr,
                self.d_v_cache[l].ptr as *mut c_void,
                self.d_attn_concat.ptr as *mut c_void,
                d as i32,
                hd as i32,
                h as i32,
                pos as i32,
                self.max_s as i32,
            );

            cuda_matmul(
                self.d_attn_concat.ptr as *const c_void,
                self.d_wo[l].ptr as *const c_void,
                self.d_sub_out.ptr as *mut c_void,
                1,
                (h * hd) as i32,
                d as i32,
            );

            cuda_add_inplace_f16(
                self.d_x.ptr as *mut c_void,
                self.d_sub_out.ptr as *const c_void,
                d as i32,
            );
            cuda_clamp_f16(self.d_x.ptr as *mut c_void, d as i32);
        }
        infer_end("attn decode", timer);
    }

    // LN2 → feed-forward → residual. Mirrors `ffn_sublayer_forward`.
    //
    // `cuda_feed_forward` is the inference variant (fused bias+GELU, no pre-GELU
    // activation saved); the training path calls `cuda_feed_forward_forward`,
    // which differs only in saving that activation for the backward.
    fn ffn_sublayer(&mut self, s: usize, l: usize) {
        let d = self.d;
        let t = infer_start();
        // SAFETY: as `attn_sublayer`.
        unsafe {
            cuda_layer_norm(
                self.d_x.ptr as *const c_void,
                self.d_norm.ptr as *mut c_void,
                self.d_ln_gamma[l].ptr as *const c_void,
                self.d_ln_beta[l].ptr as *const c_void,
                s as i32,
                d as i32,
                LAYER_NORM_EPS,
            );
            cuda_feed_forward(
                self.d_norm.ptr as *const c_void,
                self.d_w1[l].ptr as *const c_void,
                self.d_w2[l].ptr as *const c_void,
                self.d_b1[l].ptr as *const c_void,
                self.d_b2[l].ptr as *const c_void,
                self.d_sub_out.ptr as *mut c_void,
                s as i32,
                d as i32,
                self.ff as i32,
            );
            cuda_add_inplace_f16(
                self.d_x.ptr as *mut c_void,
                self.d_sub_out.ptr as *const c_void,
                (s * d) as i32,
            );
            cuda_clamp_f16(self.d_x.ptr as *mut c_void, (s * d) as i32);
        }
        infer_end("ff sublayer", t);
    }

    // Final LayerNorm over the residual stream, when the model carries one.
    // Mirrors `final_ln_forward`.
    fn final_ln(&mut self, s: usize) {
        if !self.has_final_ln {
            return;
        }
        // SAFETY: as above.
        unsafe {
            cuda_layer_norm(
                self.d_x.ptr as *const c_void,
                self.d_final.ptr as *mut c_void,
                self.d_ln_final_gamma.ptr as *const c_void,
                self.d_ln_final_beta.ptr as *const c_void,
                s as i32,
                self.d as i32,
                LAYER_NORM_EPS,
            );
        }
    }

    // Where the logit projection reads from: the final norm's output, or the raw
    // residual stream for a pre-v5 model that has no final norm.
    fn final_hidden_ptr(&self) -> *mut c_void {
        if self.has_final_ln {
            self.d_final.ptr as *mut c_void
        } else {
            self.d_x.ptr as *mut c_void
        }
    }

    fn project_logits(&self, hidden: *const c_void, rows: usize) {
        let t = infer_start();
        // SAFETY: `hidden` addresses rows*d f16 values; `d_logits` is allocated at
        // max_s*vocab and rows <= max_s.
        unsafe {
            cuda_matmul(
                hidden,
                self.d_w_output.ptr as *const c_void,
                self.d_logits.ptr as *mut c_void,
                rows as i32,
                self.d as i32,
                self.vocab as i32,
            );
        }
        infer_end("logits gemm", t);
    }

    // The one device→host copy on the whole path. Reads exactly the rows written
    // and no more — `d_logits` is allocated at max_s*vocab, and copying all of it
    // back to hand out one row would be the old path's mistake again.
    fn read_logits(&self, rows: usize) -> Vec<f32> {
        let t = infer_start();
        let out = self.d_logits.to_f32_vec_n(rows * self.vocab);
        infer_end("logits d2h", t);
        out
    }
}
