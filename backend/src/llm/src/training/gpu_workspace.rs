use crate::training::device_buffer::{DeviceBuffer, DeviceBufferF16};
use crate::training::transformer::Transformer;

unsafe extern "C" {
    fn cublas_init();
    fn cublas_shutdown();
}

pub struct GpuWorkspace {
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

    // Per-layer gradient accumulators (f32) — contiguous for QKV
    pub g_wq_all: Vec<DeviceBuffer>,      // [layers][h * d * hd]
    pub g_wk_all: Vec<DeviceBuffer>,
    pub g_wv_all: Vec<DeviceBuffer>,
    pub g_dw_output: DeviceBuffer,        // shared
    pub g_d_embedding: DeviceBuffer,       // [vocab, d]
    pub d_token_ids: DeviceBuffer,         // [max_s] (i32, stored as f32-sized slots)
    pub g_dwo: Vec<DeviceBuffer>,         // [layers]
    pub g_dw1: Vec<DeviceBuffer>,
    pub g_dw2: Vec<DeviceBuffer>,
    pub g_db1: Vec<DeviceBuffer>,
    pub g_db2: Vec<DeviceBuffer>,
    pub g_dln_gamma: Vec<DeviceBuffer>,
    pub g_dln_beta: Vec<DeviceBuffer>,

    // Adam first-moment buffers (f32) — one per gradient buffer.
    pub v_wq_all: Vec<DeviceBuffer>,
    pub v_wk_all: Vec<DeviceBuffer>,
    pub v_wv_all: Vec<DeviceBuffer>,
    pub v_dw_output: DeviceBuffer,
    pub v_d_embedding: DeviceBuffer,
    pub v_dwo: Vec<DeviceBuffer>,
    pub v_dw1: Vec<DeviceBuffer>,
    pub v_dw2: Vec<DeviceBuffer>,
    pub v_db1: Vec<DeviceBuffer>,
    pub v_db2: Vec<DeviceBuffer>,
    pub v_dln_gamma: Vec<DeviceBuffer>,
    pub v_dln_beta: Vec<DeviceBuffer>,

    // Adam second-moment buffers (f32) — EMA of gradient², one per gradient.
    pub s_wq_all: Vec<DeviceBuffer>,
    pub s_wk_all: Vec<DeviceBuffer>,
    pub s_wv_all: Vec<DeviceBuffer>,
    pub s_dw_output: DeviceBuffer,
    pub s_d_embedding: DeviceBuffer,
    pub s_dwo: Vec<DeviceBuffer>,
    pub s_dw1: Vec<DeviceBuffer>,
    pub s_dw2: Vec<DeviceBuffer>,
    pub s_db1: Vec<DeviceBuffer>,
    pub s_db2: Vec<DeviceBuffer>,
    pub s_dln_gamma: Vec<DeviceBuffer>,
    pub s_dln_beta: Vec<DeviceBuffer>,

    // Logits + loss
    pub d_logits: DeviceBufferF16,
    pub d_dlogits: DeviceBufferF16,
    pub d_targets: DeviceBuffer,
    // Per-position loss mask (f32, 1=train / 0=ignore) for SFT. Unused by the
    // pretraining path, which passes a null mask to the cross-entropy kernels.
    pub d_mask: DeviceBuffer,
    pub d_dw_output: DeviceBuffer,
    pub d_dhidden: DeviceBufferF16,

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

    pub d_head_ptrs: DeviceBuffer,

    // Backward scratch
    pub b_sd: Vec<DeviceBufferF16>,
    pub b_sd_head: DeviceBufferF16,
    pub d_wo_x: DeviceBufferF16,
    pub b_d: Vec<DeviceBuffer>,
    pub b_dd: DeviceBuffer,
    pub b_dff: Vec<DeviceBuffer>,
    pub b_ff: DeviceBuffer,
    pub max_s: usize,
    pub d: usize,
    pub h: usize,
    pub hd: usize,
    pub ff: usize,
    pub vocab: usize,
}

impl GpuWorkspace {
    pub fn new(t: &Transformer, ctx_window: usize) -> Self {
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

        // Gradient accumulators
        let g_wq_all = (0..layers).map(|_| DeviceBuffer::alloc(h * d * hd)).collect();
        let g_wk_all = (0..layers).map(|_| DeviceBuffer::alloc(h * d * hd)).collect();
        let g_wv_all = (0..layers).map(|_| DeviceBuffer::alloc(h * d * hd)).collect();
        let g_dw_output = DeviceBuffer::alloc(d * vocab);
        let g_d_embedding = DeviceBuffer::alloc(vocab * d);
        let d_token_ids = DeviceBuffer::alloc(max_s);
        let g_dwo = (0..layers).map(|_| DeviceBuffer::alloc(d * d)).collect();
        let g_dw1 = (0..layers).map(|_| DeviceBuffer::alloc(d * ff)).collect();
        let g_dw2 = (0..layers).map(|_| DeviceBuffer::alloc(ff * d)).collect();
        let g_db1 = (0..layers).map(|_| DeviceBuffer::alloc(ff)).collect();
        let g_db2 = (0..layers).map(|_| DeviceBuffer::alloc(d)).collect();
        let g_dln_gamma = (0..layers).map(|_| DeviceBuffer::alloc(d)).collect();
        let g_dln_beta = (0..layers).map(|_| DeviceBuffer::alloc(d)).collect();

        // Momentum velocity buffers (same size as gradient buffers, zero-initialized)
        let v_wq_all = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(h * d * hd)).collect();
        let v_wk_all = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(h * d * hd)).collect();
        let v_wv_all = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(h * d * hd)).collect();
        let v_dw_output = DeviceBuffer::alloc_zeroed(d * vocab);
        let v_d_embedding = DeviceBuffer::alloc_zeroed(vocab * d);
        let v_dwo = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(d * d)).collect();
        let v_dw1 = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(d * ff)).collect();
        let v_dw2 = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(ff * d)).collect();
        let v_db1 = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(ff)).collect();
        let v_db2 = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(d)).collect();
        let v_dln_gamma = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(d)).collect();
        let v_dln_beta = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(d)).collect();

        // Adam second-moment buffers (same shapes, zero-initialized).
        let s_wq_all = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(h * d * hd)).collect();
        let s_wk_all = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(h * d * hd)).collect();
        let s_wv_all = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(h * d * hd)).collect();
        let s_dw_output = DeviceBuffer::alloc_zeroed(d * vocab);
        let s_d_embedding = DeviceBuffer::alloc_zeroed(vocab * d);
        let s_dwo = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(d * d)).collect();
        let s_dw1 = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(d * ff)).collect();
        let s_dw2 = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(ff * d)).collect();
        let s_db1 = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(ff)).collect();
        let s_db2 = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(d)).collect();
        let s_dln_gamma = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(d)).collect();
        let s_dln_beta = (0..layers).map(|_| DeviceBuffer::alloc_zeroed(d)).collect();

        let d_logits = DeviceBufferF16::alloc(max_s * vocab);
        let d_dlogits = DeviceBufferF16::alloc(max_s * vocab);
        let d_targets = DeviceBuffer::alloc(max_s);
        let d_mask = DeviceBuffer::alloc(max_s);
        let d_dw_output = DeviceBuffer::alloc(d * vocab);
        let d_dhidden = DeviceBufferF16::alloc(max_s * d);

        let d_head_ptrs = DeviceBuffer::alloc(h * 2);

        let b_sd = (0..4).map(|_| DeviceBufferF16::alloc(max_s * d)).collect();
        let b_sd_head = DeviceBufferF16::alloc(max_s * d);
        let d_wo_x = DeviceBufferF16::alloc(max_s * d);
        let b_d = (0..2).map(|_| DeviceBuffer::alloc(d)).collect();
        let b_dd = DeviceBuffer::alloc(d * d);
        let b_dff = vec![DeviceBuffer::alloc(d * ff), DeviceBuffer::alloc(ff * d)];
        let b_ff = DeviceBuffer::alloc(ff);
        let ws = GpuWorkspace {
            d_wq_all, d_wk_all, d_wv_all, d_wo, d_w1, d_w2, d_b1, d_b2, d_ln_gamma, d_ln_beta, d_w_output,
            d_embedding, d_positional,
            g_wq_all, g_wk_all, g_wv_all,
            g_dw_output, g_d_embedding, d_token_ids, g_dwo, g_dw1, g_dw2, g_db1, g_db2, g_dln_gamma, g_dln_beta,
            v_wq_all, v_wk_all, v_wv_all, v_dw_output, v_d_embedding,
            v_dwo, v_dw1, v_dw2, v_db1, v_db2, v_dln_gamma, v_dln_beta,
            s_wq_all, s_wk_all, s_wv_all, s_dw_output, s_d_embedding,
            s_dwo, s_dw1, s_dw2, s_db1, s_db2, s_dln_gamma, s_dln_beta,
            d_logits, d_dlogits, d_targets, d_mask, d_dw_output, d_dhidden,
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
        ws.upload_weights(t);
        // SAFETY: cublas_init() takes no arguments and initializes the process-wide
        // cuBLAS handle used by the CUDA kernels; calling it once per `GpuWorkspace`
        // construction (paired with `cublas_shutdown` in `Drop`) is sound as long as
        // workspaces aren't constructed concurrently on multiple threads, which this
        // single-threaded training loop guarantees.
        unsafe { cublas_init(); }
        ws.zero_accumulators();
        ws
    }

    pub fn upload_weights(&self, t: &Transformer) {
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
            self.d_wq_all[layer_idx].upload_f32_slice(&wq_flat);
            self.d_wk_all[layer_idx].upload_f32_slice(&wk_flat);
            self.d_wv_all[layer_idx].upload_f32_slice(&wv_flat);

            let flat_wo: Vec<f32> = t.wo[layer_idx].iter().flatten().copied().collect();
            self.d_wo[layer_idx].upload_f32_slice(&flat_wo);
            let flat_w1: Vec<f32> = t.w1[layer_idx].iter().flatten().copied().collect();
            self.d_w1[layer_idx].upload_f32_slice(&flat_w1);
            let flat_w2: Vec<f32> = t.w2[layer_idx].iter().flatten().copied().collect();
            self.d_w2[layer_idx].upload_f32_slice(&flat_w2);
            self.d_b1[layer_idx].upload_f32_slice(&t.b1[layer_idx]);
            self.d_b2[layer_idx].upload_f32_slice(&t.b2[layer_idx]);
            self.d_ln_gamma[layer_idx].upload_f32_slice(&t.ln_gamma[layer_idx]);
            self.d_ln_beta[layer_idx].upload_f32_slice(&t.ln_beta[layer_idx]);
        }
        let flat_w_output: Vec<f32> = t.w_output.iter().flatten().copied().collect();
        self.d_w_output.upload_f32_slice(&flat_w_output);
        let flat_embedding: Vec<f32> = t.embedding_table.iter().flatten().copied().collect();
        self.d_embedding.upload_f32_slice(&flat_embedding);
        let flat_positional: Vec<f32> = t.positional_table.iter().flatten().copied().collect();
        self.d_positional.upload_f32_slice(&flat_positional);
    }

    pub fn zero_accumulators(&self) {
        // SAFETY: every pointer below is a `DeviceBuffer`/`DeviceBufferF32`
        // device allocation owned by `self` (or `self.d_token_ids`), and each
        // `cudaMemset` count matches the element count that pointer was
        // allocated with in `GpuWorkspace::new` (`self.d`, `self.vocab`,
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

impl Drop for GpuWorkspace {
    fn drop(&mut self) {
        // SAFETY: cublas_shutdown() tears down the process-wide cuBLAS handle
        // that `cublas_init()` set up in `GpuWorkspace::new`; `Drop::drop` runs
        // at most once per value, so the init/shutdown pair stays balanced.
        unsafe { cublas_shutdown(); }
    }
}
