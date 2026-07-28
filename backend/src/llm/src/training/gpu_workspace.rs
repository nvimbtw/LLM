use std::ffi::c_void;

use crate::training::device_buffer::{Buffer, DeviceBufferF16};
use crate::training::transformer::Model;

unsafe extern "C" {
    fn cublas_init();
    fn cublas_shutdown();
    fn cuda_malloc(ptr: *mut *mut c_void, size: usize);
    fn cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize);
}

pub struct Workspace {
    // Per-layer weight mirrors (f16) — per-head QKV are now contiguous
    pub d_wq_all: Vec<DeviceBufferF16>,   // [layers][h * d * hd]
    pub d_wk_all: Vec<DeviceBufferF16>,
    pub d_wv_all: Vec<DeviceBufferF16>,
    pub d_wo: Vec<DeviceBufferF16>,       // [layers][d * d]
    pub d_w1: Vec<DeviceBufferF16>,
    pub d_w2: Vec<DeviceBufferF16>,
    pub d_b1: Vec<DeviceBufferF16>,
    pub d_b2: Vec<DeviceBufferF16>,
    pub d_ln_gamma: Vec<DeviceBufferF16>,
    pub d_ln_beta: Vec<DeviceBufferF16>,
    pub d_w_output: DeviceBufferF16,      // shared
    pub d_embedding: DeviceBufferF16,      // [vocab, d]
    pub d_positional: DeviceBufferF16,     // [max_s, d]

    // f32 master weights (source of truth) — one per *trained* weight. The
    // matching `d_*` f16 buffer above is a derived view refreshed from the master
    // every optimizer step; the master is what Adam accumulates into (0052 epic).
    // `d_positional` is never stepped, so it has no master.
    pub m_wq_all: Vec<Buffer>,      // [layers][h * d * hd]
    pub m_wk_all: Vec<Buffer>,
    pub m_wv_all: Vec<Buffer>,
    pub m_wo: Vec<Buffer>,          // [layers][d * d]
    pub m_w1: Vec<Buffer>,
    pub m_w2: Vec<Buffer>,
    pub m_b1: Vec<Buffer>,
    pub m_b2: Vec<Buffer>,
    pub m_ln_gamma: Vec<Buffer>,
    pub m_ln_beta: Vec<Buffer>,
    pub m_w_output: Buffer,         // shared
    pub m_embedding: Buffer,         // [vocab, d]

    // Per-layer gradient accumulators (f32) — contiguous for QKV
    pub g_wq_all: Vec<Buffer>,      // [layers][h * d * hd]
    pub g_wk_all: Vec<Buffer>,
    pub g_wv_all: Vec<Buffer>,
    pub g_dw_output: Buffer,        // shared
    pub g_d_embedding: Buffer,       // [vocab, d]
    pub d_token_ids: Buffer,         // [max_s] (i32, stored as f32-sized slots)
    pub g_dwo: Vec<Buffer>,         // [layers]
    pub g_dw1: Vec<Buffer>,
    pub g_dw2: Vec<Buffer>,
    pub g_db1: Vec<Buffer>,
    pub g_db2: Vec<Buffer>,
    pub g_dln_gamma: Vec<Buffer>,
    pub g_dln_beta: Vec<Buffer>,

    // Adam first-moment buffers (f32) — one per gradient buffer.
    pub v_wq_all: Vec<Buffer>,
    pub v_wk_all: Vec<Buffer>,
    pub v_wv_all: Vec<Buffer>,
    pub v_dw_output: Buffer,
    pub v_d_embedding: Buffer,
    pub v_dwo: Vec<Buffer>,
    pub v_dw1: Vec<Buffer>,
    pub v_dw2: Vec<Buffer>,
    pub v_db1: Vec<Buffer>,
    pub v_db2: Vec<Buffer>,
    pub v_dln_gamma: Vec<Buffer>,
    pub v_dln_beta: Vec<Buffer>,

    // Adam second-moment buffers (f32) — EMA of gradient², one per gradient.
    pub s_wq_all: Vec<Buffer>,
    pub s_wk_all: Vec<Buffer>,
    pub s_wv_all: Vec<Buffer>,
    pub s_dw_output: Buffer,
    pub s_d_embedding: Buffer,
    pub s_dwo: Vec<Buffer>,
    pub s_dw1: Vec<Buffer>,
    pub s_dw2: Vec<Buffer>,
    pub s_db1: Vec<Buffer>,
    pub s_db2: Vec<Buffer>,
    pub s_dln_gamma: Vec<Buffer>,
    pub s_dln_beta: Vec<Buffer>,

    // Logits + loss
    pub d_logits: DeviceBufferF16,
    pub d_dlogits: DeviceBufferF16,
    pub d_targets: Buffer,
    // Per-position loss mask (f32, 1=train / 0=ignore) for SFT. Unused by the
    // pretraining path, which passes a null mask to the cross-entropy kernels.
    pub d_mask: Buffer,
    pub d_dw_output: Buffer,
    pub d_dhidden: DeviceBufferF16,
    // Gradient-clip state, 3 f32s kept on the device so the clip never
    // synchronizes: [0] sum-of-squares accumulator, [1] the global gradient
    // norm, [2] the scale applied to every gradient buffer. Allocated zeroed
    // because `cuda_clip_finalize` consumes [0] and re-zeros it each step.
    pub d_clip_state: Buffer,
    // Device-resident table of the gradient buffers `Model::clip_gradients`
    // clips, so the whole clip is two launches instead of three per tensor.
    // Pointers and sizes are built once in `new` — workspace allocations never
    // move — and `d_clip_norms` holds one accumulator per buffer.
    pub d_clip_ptrs: *mut c_void,   // [n_clip] device pointers (8 bytes each)
    pub d_clip_sizes: Buffer,       // [n_clip] i32 element counts
    pub d_clip_norms: Buffer,       // [n_clip] f32 sum-of-squares accumulators
    pub n_clip: usize,

    // Per-layer forward caches (f16)
    pub layer_inputs: Vec<DeviceBufferF16>,
    pub post_attn_residuals: Vec<DeviceBufferF16>,
    pub ff_inputs: Vec<DeviceBufferF16>,
    pub pre_gelus: Vec<DeviceBufferF16>,
    pub post_ff_residuals: Vec<DeviceBufferF16>,
    pub concat_outputs: Vec<DeviceBufferF16>,

    // Contiguous per-head buffers for batched attention
    pub head_out_all: Vec<DeviceBufferF16>,   // [layers][h * max_s * hd]
    pub head_p_all: Vec<DeviceBufferF16>,     // [layers][h * max_s * max_s]

    pub d_head_ptrs: Buffer,

    // Backward scratch
    pub b_sd: Vec<DeviceBufferF16>,
    pub b_sd_head: DeviceBufferF16,
    pub d_wo_x: DeviceBufferF16,
    pub b_d: Vec<Buffer>,
    pub b_dd: Buffer,
    pub b_dff: Vec<Buffer>,
    pub b_ff: Buffer,
    pub max_s: usize,
    pub d: usize,
    pub h: usize,
    pub hd: usize,
    pub ff: usize,
    pub vocab: usize,
}

impl Workspace {
    pub fn new(t: &Model, ctx_window: usize) -> Self {
        let max_s = ctx_window;
        let d = t.wo[0].len();
        let h = t.heads as usize;
        let hd = t.head_dim as usize;
        let ff = t.w1[0][0].len();
        let vocab = t.w_output[0].len();
        let layers = t.layers as usize;

        // Contiguous per-head QKV weights [h * d * hd]
        let d_wq_all = (0..layers).map(|_| DeviceBufferF16::alloc(h * d * hd)).collect();
        let d_wk_all = (0..layers).map(|_| DeviceBufferF16::alloc(h * d * hd)).collect();
        let d_wv_all = (0..layers).map(|_| DeviceBufferF16::alloc(h * d * hd)).collect();
        let d_wo = (0..layers).map(|_| DeviceBufferF16::alloc(d * d)).collect();
        let d_w1 = (0..layers).map(|_| DeviceBufferF16::alloc(d * ff)).collect();
        let d_w2 = (0..layers).map(|_| DeviceBufferF16::alloc(ff * d)).collect();
        let d_b1 = (0..layers).map(|_| DeviceBufferF16::alloc(ff)).collect();
        let d_b2 = (0..layers).map(|_| DeviceBufferF16::alloc(d)).collect();
        let d_ln_gamma = (0..layers).map(|_| DeviceBufferF16::alloc(d)).collect();
        let d_ln_beta = (0..layers).map(|_| DeviceBufferF16::alloc(d)).collect();
        let d_w_output = DeviceBufferF16::alloc(d * vocab);
        let d_embedding = DeviceBufferF16::alloc(vocab * d);
        let d_positional = DeviceBufferF16::alloc(max_s * d);

        // f32 master weights (same element counts as the f16 views above). Start
        // zeroed; 0056 seeds them from the f32 CPU weights on `upload_weights`.
        let m_wq_all = (0..layers).map(|_| Buffer::alloc_zeroed(h * d * hd)).collect();
        let m_wk_all = (0..layers).map(|_| Buffer::alloc_zeroed(h * d * hd)).collect();
        let m_wv_all = (0..layers).map(|_| Buffer::alloc_zeroed(h * d * hd)).collect();
        let m_wo = (0..layers).map(|_| Buffer::alloc_zeroed(d * d)).collect();
        let m_w1 = (0..layers).map(|_| Buffer::alloc_zeroed(d * ff)).collect();
        let m_w2 = (0..layers).map(|_| Buffer::alloc_zeroed(ff * d)).collect();
        let m_b1 = (0..layers).map(|_| Buffer::alloc_zeroed(ff)).collect();
        let m_b2 = (0..layers).map(|_| Buffer::alloc_zeroed(d)).collect();
        let m_ln_gamma = (0..layers).map(|_| Buffer::alloc_zeroed(d)).collect();
        let m_ln_beta = (0..layers).map(|_| Buffer::alloc_zeroed(d)).collect();
        let m_w_output = Buffer::alloc_zeroed(d * vocab);
        let m_embedding = Buffer::alloc_zeroed(vocab * d);

        // Gradient accumulators
        let g_wq_all = (0..layers).map(|_| Buffer::alloc(h * d * hd)).collect();
        let g_wk_all = (0..layers).map(|_| Buffer::alloc(h * d * hd)).collect();
        let g_wv_all = (0..layers).map(|_| Buffer::alloc(h * d * hd)).collect();
        let g_dw_output = Buffer::alloc(d * vocab);
        let g_d_embedding = Buffer::alloc(vocab * d);
        let d_token_ids = Buffer::alloc(max_s);
        let g_dwo = (0..layers).map(|_| Buffer::alloc(d * d)).collect();
        let g_dw1 = (0..layers).map(|_| Buffer::alloc(d * ff)).collect();
        let g_dw2 = (0..layers).map(|_| Buffer::alloc(ff * d)).collect();
        let g_db1 = (0..layers).map(|_| Buffer::alloc(ff)).collect();
        let g_db2 = (0..layers).map(|_| Buffer::alloc(d)).collect();
        let g_dln_gamma = (0..layers).map(|_| Buffer::alloc(d)).collect();
        let g_dln_beta = (0..layers).map(|_| Buffer::alloc(d)).collect();

        // Momentum velocity buffers (same size as gradient buffers, zero-initialized)
        let v_wq_all = (0..layers).map(|_| Buffer::alloc_zeroed(h * d * hd)).collect();
        let v_wk_all = (0..layers).map(|_| Buffer::alloc_zeroed(h * d * hd)).collect();
        let v_wv_all = (0..layers).map(|_| Buffer::alloc_zeroed(h * d * hd)).collect();
        let v_dw_output = Buffer::alloc_zeroed(d * vocab);
        let v_d_embedding = Buffer::alloc_zeroed(vocab * d);
        let v_dwo = (0..layers).map(|_| Buffer::alloc_zeroed(d * d)).collect();
        let v_dw1 = (0..layers).map(|_| Buffer::alloc_zeroed(d * ff)).collect();
        let v_dw2 = (0..layers).map(|_| Buffer::alloc_zeroed(ff * d)).collect();
        let v_db1 = (0..layers).map(|_| Buffer::alloc_zeroed(ff)).collect();
        let v_db2 = (0..layers).map(|_| Buffer::alloc_zeroed(d)).collect();
        let v_dln_gamma = (0..layers).map(|_| Buffer::alloc_zeroed(d)).collect();
        let v_dln_beta = (0..layers).map(|_| Buffer::alloc_zeroed(d)).collect();

        // Adam second-moment buffers (same shapes, zero-initialized).
        let s_wq_all = (0..layers).map(|_| Buffer::alloc_zeroed(h * d * hd)).collect();
        let s_wk_all = (0..layers).map(|_| Buffer::alloc_zeroed(h * d * hd)).collect();
        let s_wv_all = (0..layers).map(|_| Buffer::alloc_zeroed(h * d * hd)).collect();
        let s_dw_output = Buffer::alloc_zeroed(d * vocab);
        let s_d_embedding = Buffer::alloc_zeroed(vocab * d);
        let s_dwo = (0..layers).map(|_| Buffer::alloc_zeroed(d * d)).collect();
        let s_dw1 = (0..layers).map(|_| Buffer::alloc_zeroed(d * ff)).collect();
        let s_dw2 = (0..layers).map(|_| Buffer::alloc_zeroed(ff * d)).collect();
        let s_db1 = (0..layers).map(|_| Buffer::alloc_zeroed(ff)).collect();
        let s_db2 = (0..layers).map(|_| Buffer::alloc_zeroed(d)).collect();
        let s_dln_gamma = (0..layers).map(|_| Buffer::alloc_zeroed(d)).collect();
        let s_dln_beta = (0..layers).map(|_| Buffer::alloc_zeroed(d)).collect();

        let d_logits = DeviceBufferF16::alloc(max_s * vocab);
        let d_dlogits = DeviceBufferF16::alloc(max_s * vocab);
        let d_targets = Buffer::alloc(max_s);
        let d_mask = Buffer::alloc(max_s);
        let d_dw_output = Buffer::alloc(d * vocab);
        let d_dhidden = DeviceBufferF16::alloc(max_s * d);
        let d_clip_state = Buffer::alloc_zeroed(3);

        // The buffers `Model::clip_gradients` walks, in the same order: the
        // shared output projection, then each layer's ten trained tensors.
        let n_clip = 1 + 10 * layers;
        let d_clip_sizes = Buffer::alloc(n_clip);
        let d_clip_norms = Buffer::alloc_zeroed(n_clip);
        let mut d_clip_ptrs: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-pointer to a local; allocates n_clip device pointers.
        unsafe { cuda_malloc(&mut d_clip_ptrs, n_clip * size_of::<*mut f32>()) };

        let d_head_ptrs = Buffer::alloc(h * 2);

        let b_sd = (0..4).map(|_| DeviceBufferF16::alloc(max_s * d)).collect();
        let b_sd_head = DeviceBufferF16::alloc(max_s * d);
        let d_wo_x = DeviceBufferF16::alloc(max_s * d);
        let b_d = (0..2).map(|_| Buffer::alloc(d)).collect();
        let b_dd = Buffer::alloc(d * d);
        let b_dff = vec![Buffer::alloc(d * ff), Buffer::alloc(ff * d)];
        let b_ff = Buffer::alloc(ff);
        let mut ws = Workspace {
            d_wq_all, d_wk_all, d_wv_all, d_wo, d_w1, d_w2, d_b1, d_b2, d_ln_gamma, d_ln_beta, d_w_output,
            d_embedding, d_positional,
            m_wq_all, m_wk_all, m_wv_all, m_wo, m_w1, m_w2, m_b1, m_b2, m_ln_gamma, m_ln_beta,
            m_w_output, m_embedding,
            g_wq_all, g_wk_all, g_wv_all,
            g_dw_output, g_d_embedding, d_token_ids, g_dwo, g_dw1, g_dw2, g_db1, g_db2, g_dln_gamma, g_dln_beta,
            v_wq_all, v_wk_all, v_wv_all, v_dw_output, v_d_embedding,
            v_dwo, v_dw1, v_dw2, v_db1, v_db2, v_dln_gamma, v_dln_beta,
            s_wq_all, s_wk_all, s_wv_all, s_dw_output, s_d_embedding,
            s_dwo, s_dw1, s_dw2, s_db1, s_db2, s_dln_gamma, s_dln_beta,
            d_logits, d_dlogits, d_targets, d_mask, d_dw_output, d_dhidden,
            d_clip_state,
            d_clip_ptrs, d_clip_sizes, d_clip_norms, n_clip,
            layer_inputs: Vec::new(),
            post_attn_residuals: Vec::new(),
            ff_inputs: Vec::new(),
            pre_gelus: Vec::new(),
            post_ff_residuals: Vec::new(),
            concat_outputs: Vec::new(),
            head_out_all: Vec::new(),
            head_p_all: Vec::new(),
            d_head_ptrs,
            b_sd, b_d, b_dd, b_dff, b_ff, b_sd_head, d_wo_x,
            max_s, d, h, hd, ff, vocab,
        };
        ws.fill_clip_table(layers);
        ws.upload_weights(t);
        // SAFETY: cublas_init() takes no arguments and initializes the process-wide
        // cuBLAS handle used by the CUDA kernels; calling it once per `Workspace`
        // construction (paired with `cublas_shutdown` in `Drop`) is sound as long as
        // workspaces aren't constructed concurrently on multiple threads, which this
        // single-threaded training loop guarantees.
        unsafe { cublas_init(); }
        ws.zero_accumulators();
        ws
    }

    pub fn upload_weights(&self, t: &Model) {
        let d = self.d;
        let h = self.h;
        let hd = self.hd;
        let _ff = self.ff;
        let _vocab = self.vocab;

        for layer_idx in 0..t.layers as usize {
            // Upload contiguous per-head QKV weights
            let mut wq_flat = Vec::with_capacity(h * d * hd);
            let mut wk_flat = Vec::with_capacity(h * d * hd);
            let mut wv_flat = Vec::with_capacity(h * d * hd);
            for hi in 0..h {
                wq_flat.extend(t.wq[layer_idx][hi].iter().flatten());
                wk_flat.extend(t.wk[layer_idx][hi].iter().flatten());
                wv_flat.extend(t.wv[layer_idx][hi].iter().flatten());
            }
            // Seed both the f16 view (first forward pass reads it before any Adam
            // step) and the f32 master (source of truth) from the full-precision
            // CPU weights — the master gets the exact f32, not the f16 round-trip.
            self.d_wq_all[layer_idx].upload_f32_slice(&wq_flat);
            self.d_wk_all[layer_idx].upload_f32_slice(&wk_flat);
            self.d_wv_all[layer_idx].upload_f32_slice(&wv_flat);
            self.m_wq_all[layer_idx].upload(&wq_flat);
            self.m_wk_all[layer_idx].upload(&wk_flat);
            self.m_wv_all[layer_idx].upload(&wv_flat);

            let flat_wo: Vec<f32> = t.wo[layer_idx].iter().flatten().copied().collect();
            self.d_wo[layer_idx].upload_f32_slice(&flat_wo);
            self.m_wo[layer_idx].upload(&flat_wo);
            let flat_w1: Vec<f32> = t.w1[layer_idx].iter().flatten().copied().collect();
            self.d_w1[layer_idx].upload_f32_slice(&flat_w1);
            self.m_w1[layer_idx].upload(&flat_w1);
            let flat_w2: Vec<f32> = t.w2[layer_idx].iter().flatten().copied().collect();
            self.d_w2[layer_idx].upload_f32_slice(&flat_w2);
            self.m_w2[layer_idx].upload(&flat_w2);
            self.d_b1[layer_idx].upload_f32_slice(&t.b1[layer_idx]);
            self.m_b1[layer_idx].upload(&t.b1[layer_idx]);
            self.d_b2[layer_idx].upload_f32_slice(&t.b2[layer_idx]);
            self.m_b2[layer_idx].upload(&t.b2[layer_idx]);
            self.d_ln_gamma[layer_idx].upload_f32_slice(&t.ln_gamma[layer_idx]);
            self.m_ln_gamma[layer_idx].upload(&t.ln_gamma[layer_idx]);
            self.d_ln_beta[layer_idx].upload_f32_slice(&t.ln_beta[layer_idx]);
            self.m_ln_beta[layer_idx].upload(&t.ln_beta[layer_idx]);
        }
        let flat_w_output: Vec<f32> = t.w_output.iter().flatten().copied().collect();
        self.d_w_output.upload_f32_slice(&flat_w_output);
        self.m_w_output.upload(&flat_w_output);
        let flat_embedding: Vec<f32> = t.embedding_table.iter().flatten().copied().collect();
        self.d_embedding.upload_f32_slice(&flat_embedding);
        self.m_embedding.upload(&flat_embedding);
        // `d_positional` is never stepped by Adam, so it has no master.
        let flat_positional: Vec<f32> = t.positional_table.iter().flatten().copied().collect();
        self.d_positional.upload_f32_slice(&flat_positional);
    }

    // Upload the pointer/size table the batched gradient clip reads. Must list
    // the buffers in exactly the order `Model::clip_gradients` clips them; the
    // device allocations are stable for the workspace's life, so this runs once.
    fn fill_clip_table(&mut self, layers: usize) {
        let mut ptrs: Vec<*mut f32> = Vec::with_capacity(self.n_clip);
        let mut sizes: Vec<f32> = Vec::with_capacity(self.n_clip);
        let (d, h, hd, ff, vocab) = (self.d, self.h, self.hd, self.ff, self.vocab);

        ptrs.push(self.g_dw_output.ptr);
        sizes.push((d * vocab) as f32);
        for l in 0..layers {
            for (p, n) in [
                (self.g_wq_all[l].ptr, h * d * hd),
                (self.g_wk_all[l].ptr, h * d * hd),
                (self.g_wv_all[l].ptr, h * d * hd),
                (self.g_dwo[l].ptr, d * d),
                (self.g_dw1[l].ptr, d * ff),
                (self.g_dw2[l].ptr, ff * d),
                (self.g_db1[l].ptr, ff),
                (self.g_db2[l].ptr, d),
                (self.g_dln_gamma[l].ptr, d),
                (self.g_dln_beta[l].ptr, d),
            ] {
                ptrs.push(p);
                sizes.push(n as f32);
            }
        }
        debug_assert_eq!(ptrs.len(), self.n_clip);

        // Sizes ride in an f32 buffer (4 bytes, same as the i32 the kernel
        // reads) so they can reuse `Buffer`; every count here is far below f32's
        // exact-integer range, so the round trip is lossless.
        let sizes_i32: Vec<i32> = sizes.iter().map(|&n| n as i32).collect();
        // SAFETY: both copies are sized from the same `n_clip` the device
        // buffers were allocated with, and both sources are live local slices.
        unsafe {
            cuda_memcpy_h2d(
                self.d_clip_ptrs,
                ptrs.as_ptr() as *const c_void,
                ptrs.len() * size_of::<*mut f32>(),
            );
            cuda_memcpy_h2d(
                self.d_clip_sizes.ptr as *mut c_void,
                sizes_i32.as_ptr() as *const c_void,
                sizes_i32.len() * size_of::<i32>(),
            );
        }
    }

    pub fn zero_accumulators(&self) {
        // SAFETY: every pointer below is a `Buffer`/`DeviceBufferF32`
        // device allocation owned by `self` (or `self.d_token_ids`), and each
        // `cudaMemset` count matches the element count that pointer was
        // allocated with in `Workspace::new` (`self.d`, `self.vocab`,
        // `self.h`, `self.hd`, `self.ff`, `self.max_s` — the same dims used at
        // allocation time), scaled by 4 bytes per f32 element.
        unsafe {
            extern "C" { fn cudaMemset(ptr: *mut std::ffi::c_void, value: i32, count: usize) -> i32; }

            cudaMemset(self.g_dw_output.ptr as *mut std::ffi::c_void, 0, (self.d * self.vocab * 4) as usize);
            cudaMemset(self.g_d_embedding.ptr as *mut std::ffi::c_void, 0, (self.vocab * self.d * 4) as usize);
            cudaMemset(self.d_token_ids.ptr as *mut std::ffi::c_void, 0, (self.max_s * 4) as usize);

            for layer_idx in 0..self.d_wo.len() {
                cudaMemset(self.g_wq_all[layer_idx].ptr as *mut std::ffi::c_void, 0, (self.h * self.d * self.hd * 4) as usize);
                cudaMemset(self.g_wk_all[layer_idx].ptr as *mut std::ffi::c_void, 0, (self.h * self.d * self.hd * 4) as usize);
                cudaMemset(self.g_wv_all[layer_idx].ptr as *mut std::ffi::c_void, 0, (self.h * self.d * self.hd * 4) as usize);
                cudaMemset(self.g_dwo[layer_idx].ptr as *mut std::ffi::c_void, 0, (self.d * self.d * 4) as usize);
                cudaMemset(self.g_dw1[layer_idx].ptr as *mut std::ffi::c_void, 0, (self.d * self.ff * 4) as usize);
                cudaMemset(self.g_dw2[layer_idx].ptr as *mut std::ffi::c_void, 0, (self.ff * self.d * 4) as usize);
                cudaMemset(self.g_db1[layer_idx].ptr as *mut std::ffi::c_void, 0, (self.ff * 4) as usize);
                cudaMemset(self.g_db2[layer_idx].ptr as *mut std::ffi::c_void, 0, (self.d * 4) as usize);
                cudaMemset(self.g_dln_gamma[layer_idx].ptr as *mut std::ffi::c_void, 0, (self.d * 4) as usize);
                cudaMemset(self.g_dln_beta[layer_idx].ptr as *mut std::ffi::c_void, 0, (self.d * 4) as usize);
            }
        }
    }

    pub fn ensure_layer_caches(&mut self, layers: usize) {
        while self.layer_inputs.len() < layers {
            self.layer_inputs.push(DeviceBufferF16::alloc(self.max_s * self.d));
            self.post_attn_residuals.push(DeviceBufferF16::alloc(self.max_s * self.d));
            self.ff_inputs.push(DeviceBufferF16::alloc(self.max_s * self.d));
            self.pre_gelus.push(DeviceBufferF16::alloc(self.max_s * self.ff));
            self.post_ff_residuals.push(DeviceBufferF16::alloc(self.max_s * self.d));
            self.concat_outputs.push(DeviceBufferF16::alloc(self.max_s * self.d));
            self.head_out_all.push(DeviceBufferF16::alloc(self.h * self.max_s * self.hd));
            self.head_p_all.push(DeviceBufferF16::alloc(self.h * self.max_s * self.max_s));
        }
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        // SAFETY: cublas_shutdown() tears down the process-wide cuBLAS handle
        // that `cublas_init()` set up in `Workspace::new`; `Drop::drop` runs
        // at most once per value, so the init/shutdown pair stays balanced.
        unsafe { cublas_shutdown(); }
    }
}
