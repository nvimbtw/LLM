//! Adam optimizer step and the global gradient-norm clip. These operate on the
//! GPU `Workspace` (not the model), so they live at `training/` level rather than
//! under `transformer/`. FFI decls are duplicated (not shared) so each module
//! declares exactly the kernels it calls.

use std::ffi::c_void;

use crate::training::gpu_workspace::Workspace;

// AdamW optimizer hyperparameters (standard defaults).
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;
// Decoupled weight decay. Applied to the 2-D weight matrices only; 1-D params
// (biases, LayerNorm scale/shift) pass 0.0 — decaying those is the standard
// AdamW carve-out (it hurts more than it helps). 0.1 is the common GPT default;
// tune here if a run over- or under-regularizes.
const ADAM_WEIGHT_DECAY: f32 = 0.1;

unsafe extern "C" {
    fn cuda_adam(
        g: *const f32,
        w_master: *mut f32,
        w_view: *mut c_void,
        m: *mut f32,
        v: *mut f32,
        grad_scale: f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
        n: i32,
    );
    fn cuda_l2norm(v: *const f32, n: i32) -> f32;
    // Device-side global gradient clip — see `gpu_clip_gradients`. All three are
    // asynchronous; none of them reads back to the host.
    fn cuda_grad_sumsq_accum(v: *const f32, n: i32, accum: *mut f32);
    fn cuda_clip_finalize(state: *mut f32, max_norm: f32);
    fn cuda_scale_by_dev(v: *mut f32, scale: *const f32, n: i32);
    fn cuda_memcpy_d2h(dst: *mut c_void, src: *const c_void, size: usize);
    // Sum of squares of the AdamW step vector `lr * m_hat/(sqrt(v_hat)+eps)` over
    // the current moment buffers — the applied-update magnitude for the
    // update:weight telemetry (decoupled weight decay excluded). Read-only.
    fn cuda_adam_step_sumsq(
        m: *const f32,
        v: *const f32,
        lr: f32,
        eps: f32,
        bc1: f32,
        bc2: f32,
        n: i32,
    ) -> f32;
}

// One trained tensor's device buffers, as (grad, master, moment1, moment2, n).
// The tuple bundles exactly the buffers the grad-norm and update:weight
// diagnostics read, in one place so both walk parameters identically.
type ParamBufs = (*mut f32, *mut f32, *mut f32, *mut f32, usize);

// The 10 trained tensors of layer `l`, in the same order `gpu_adam_update`
// steps them (moment1 = `v_*`, moment2 = `s_*`, master = `m_*`; see the
// mixed-precision note in backend/CLAUDE.md). Shared by both diagnostics below.
fn layer_param_bufs(ws: &Workspace, l: usize) -> [ParamBufs; 10] {
    let d = ws.d;
    let h = ws.h;
    let hd = ws.hd;
    let ff = ws.ff;
    [
        (ws.g_wq_all[l].ptr, ws.m_wq_all[l].ptr, ws.v_wq_all[l].ptr, ws.s_wq_all[l].ptr, h * d * hd),
        (ws.g_wk_all[l].ptr, ws.m_wk_all[l].ptr, ws.v_wk_all[l].ptr, ws.s_wk_all[l].ptr, h * d * hd),
        (ws.g_wv_all[l].ptr, ws.m_wv_all[l].ptr, ws.v_wv_all[l].ptr, ws.s_wv_all[l].ptr, h * d * hd),
        (ws.g_dwo[l].ptr, ws.m_wo[l].ptr, ws.v_dwo[l].ptr, ws.s_dwo[l].ptr, d * d),
        (ws.g_dw1[l].ptr, ws.m_w1[l].ptr, ws.v_dw1[l].ptr, ws.s_dw1[l].ptr, d * ff),
        (ws.g_dw2[l].ptr, ws.m_w2[l].ptr, ws.v_dw2[l].ptr, ws.s_dw2[l].ptr, ff * d),
        (ws.g_db1[l].ptr, ws.m_b1[l].ptr, ws.v_db1[l].ptr, ws.s_db1[l].ptr, ff),
        (ws.g_db2[l].ptr, ws.m_b2[l].ptr, ws.v_db2[l].ptr, ws.s_db2[l].ptr, d),
        (ws.g_dln_gamma[l].ptr, ws.m_ln_gamma[l].ptr, ws.v_dln_gamma[l].ptr, ws.s_dln_gamma[l].ptr, d),
        (ws.g_dln_beta[l].ptr, ws.m_ln_beta[l].ptr, ws.v_dln_beta[l].ptr, ws.s_dln_beta[l].ptr, d),
    ]
}

// Gradient L2 norms for the training dashboard: one per transformer layer (over
// that layer's 10 trained tensors), the two shared tables (token embedding and
// output projection), and the global norm over everything. Reads the accumulated
// gradient buffers, so it must be sampled *before* the optimizer step zeroes them.
pub struct GradNorms {
    pub global: f32,
    pub layers: Vec<f32>,
    pub embedding: f32,
    pub unembed: f32,
}

pub fn gpu_grad_norms(ws: &Workspace, layers: usize) -> GradNorms {
    let mut layer_norms = Vec::with_capacity(layers);
    let mut global_sq = 0.0f64;
    for l in 0..layers {
        let mut layer_sq = 0.0f64;
        for (g, _m, _v, _s, n) in layer_param_bufs(ws, l) {
            let norm = unsafe { cuda_l2norm(g, n as i32) } as f64;
            layer_sq += norm * norm;
        }
        global_sq += layer_sq;
        layer_norms.push(layer_sq.sqrt() as f32);
    }
    let embedding = unsafe { cuda_l2norm(ws.g_d_embedding.ptr, (ws.vocab * ws.d) as i32) };
    let unembed = unsafe { cuda_l2norm(ws.g_dw_output.ptr, (ws.d * ws.vocab) as i32) };
    global_sq += (embedding as f64).powi(2) + (unembed as f64).powi(2);
    GradNorms {
        global: global_sq.sqrt() as f32,
        layers: layer_norms,
        embedding,
        unembed,
    }
}

// Per-layer update:weight ratio ‖Δw‖ / ‖w‖ — the classic "is the learning rate in
// the right ballpark" diagnostic (healthy is ~1e-3). ‖Δw‖ is the AdamW adaptive
// step magnitude (from the current moments), ‖w‖ the master-weight norm; both are
// pooled over a layer's 10 tensors. Sampled *after* the optimizer step so the
// moments are the ones that produced the just-applied update. `step` is the 1-based
// Adam step (for bias correction), `lr` the scheduled learning rate of that step.
pub fn gpu_update_ratios(ws: &Workspace, layers: usize, lr: f32, step: i32) -> Vec<f32> {
    // Before the first step the bias-correction denominators are 0 (1 - beta^0);
    // there is no update to report yet, so return zeros rather than divide by zero.
    if step < 1 {
        return vec![0.0; layers];
    }
    let bc1 = 1.0 - ADAM_BETA1.powi(step);
    let bc2 = 1.0 - ADAM_BETA2.powi(step);
    let mut ratios = Vec::with_capacity(layers);
    for l in 0..layers {
        let mut update_sq = 0.0f64;
        let mut weight_sq = 0.0f64;
        for (_g, m, v, s, n) in layer_param_bufs(ws, l) {
            // cuda_adam_step_sumsq already returns Σ(step²); accumulate directly.
            update_sq += unsafe { cuda_adam_step_sumsq(v, s, lr, ADAM_EPS, bc1, bc2, n as i32) } as f64;
            let wn = unsafe { cuda_l2norm(m, n as i32) } as f64;
            weight_sq += wn * wn;
        }
        let ratio = if weight_sq > 0.0 {
            (update_sq.sqrt() / weight_sq.sqrt()) as f32
        } else {
            0.0
        };
        ratios.push(ratio);
    }
    ratios
}

fn grad_buffers(ws: &Workspace, layers: usize) -> Vec<(*mut f32, usize)> {
    let d = ws.d;
    let h = ws.h;
    let hd = ws.hd;
    let ff = ws.ff;
    let vocab = ws.vocab;
    let mut bufs = vec![
        (ws.g_dw_output.ptr, d * vocab),
        (ws.g_d_embedding.ptr, vocab * d),
    ];
    for l in 0..layers {
        bufs.push((ws.g_wq_all[l].ptr, h * d * hd));
        bufs.push((ws.g_wk_all[l].ptr, h * d * hd));
        bufs.push((ws.g_wv_all[l].ptr, h * d * hd));
        bufs.push((ws.g_dwo[l].ptr, d * d));
        bufs.push((ws.g_dw1[l].ptr, d * ff));
        bufs.push((ws.g_dw2[l].ptr, ff * d));
        bufs.push((ws.g_db1[l].ptr, ff));
        bufs.push((ws.g_db2[l].ptr, d));
        bufs.push((ws.g_dln_gamma[l].ptr, d));
        bufs.push((ws.g_dln_beta[l].ptr, d));
    }
    bufs
}

// Clips the global gradient norm to `max_norm` (over all trainable buffers),
// scaling every gradient down by the same factor when it's exceeded. This is the
// standard guard against loss spikes: no single step can be large enough to blow
// the weights up. A non-finite norm (already-overflowed gradient) is zeroed so
// the step is effectively skipped. Returns the pre-clip norm for logging.
pub fn gpu_clip_gradients(ws: &Workspace, layers: usize, max_norm: f32) {
    let bufs = grad_buffers(ws, layers);

    // Sum of squares over every gradient buffer, accumulated into
    // `d_clip_state[0]`. One async kernel per buffer, no readback.
    for &(ptr, n) in &bufs {
        // SAFETY: each `ptr` is a live f32 device allocation of `n` elements
        // (from `grad_buffers`), and `d_clip_state` is a 3-f32 device buffer.
        unsafe { cuda_grad_sumsq_accum(ptr, n as i32, ws.d_clip_state.ptr) };
    }

    // norm = sqrt(accum) into [1], clip scale into [2], and [0] re-zeroed for
    // the next optimizer step. Applies the same policy the host-side version
    // did: zero on a non-finite norm, scale down when over `max_norm`.
    // SAFETY: `d_clip_state` is a 3-f32 device buffer.
    unsafe { cuda_clip_finalize(ws.d_clip_state.ptr, max_norm) };

    for &(ptr, n) in &bufs {
        // SAFETY: as above; `.add(2)` is the scale slot inside the same 3-f32
        // allocation, and the kernel only reads it.
        unsafe { cuda_scale_by_dev(ptr, ws.d_clip_state.ptr.add(2), n as i32) };
    }
}

/// The global gradient norm computed by the last `gpu_clip_gradients`.
///
/// **This synchronizes** — it is the one readback the clip still needs, so call
/// it only when the value is actually going to be used (the periodic log line),
/// never once per step. That per-step readback is what the device-side clip
/// exists to avoid.
pub fn read_grad_norm(ws: &Workspace) -> f32 {
    let mut out = 0.0f32;
    // SAFETY: copying 4 bytes out of slot [1] of the 3-f32 `d_clip_state`
    // allocation into a live local f32.
    unsafe {
        cuda_memcpy_d2h(
            &mut out as *mut f32 as *mut c_void,
            ws.d_clip_state.ptr.add(1) as *const c_void,
            4,
        );
    }
    out
}

// AdamW optimizer step. `step` is the 1-based global update count (for bias
// correction); `accum_steps` is the gradient-accumulation factor so the summed
// gradient is averaged. `v_*` hold the first moment (m), `s_*` the second (v).
// Decoupled weight decay (`ADAM_WEIGHT_DECAY`) is applied to the 2-D weight
// matrices; 1-D params (biases `b1`/`b2`, LayerNorm `gamma`/`beta`) pass 0.0.
pub fn gpu_adam_update(ws: &Workspace, lr: f32, step: i32, accum_steps: usize, layers: usize) {
    let beta1 = ADAM_BETA1;
    let beta2 = ADAM_BETA2;
    let eps = ADAM_EPS;
    let wd = ADAM_WEIGHT_DECAY;
    let bc1 = 1.0 - beta1.powi(step);
    let bc2 = 1.0 - beta2.powi(step);
    let gs = 1.0 / accum_steps as f32;
    let d = ws.d;
    let h = ws.h;
    let hd = ws.hd;
    let ff = ws.ff;
    let vocab = ws.vocab;

    // Cuts the repeated argument list down to (grad, master, view, m, v, wd, n).
    // `$wm` is the f32 master (canonical); `$wv` the f16 view refreshed from it;
    // `$wd` is this tensor's decoupled weight decay (0.0 disables it).
    macro_rules! adam {
        ($g:expr, $wm:expr, $wv:expr, $m:expr, $v:expr, $wd:expr, $n:expr) => {
            cuda_adam($g, $wm, $wv as *mut c_void, $m, $v, gs, lr, beta1, beta2, eps, $wd, bc1, bc2, $n as i32)
        };
    }

    unsafe {
        // 2-D weight matrices — decayed.
        adam!(ws.g_dw_output.ptr, ws.m_w_output.ptr, ws.d_w_output.ptr, ws.v_dw_output.ptr, ws.s_dw_output.ptr, wd, d * vocab);
        adam!(ws.g_d_embedding.ptr, ws.m_embedding.ptr, ws.d_embedding.ptr, ws.v_d_embedding.ptr, ws.s_d_embedding.ptr, wd, vocab * d);
        for l in 0..layers {
            adam!(ws.g_wq_all[l].ptr, ws.m_wq_all[l].ptr, ws.d_wq_all[l].ptr, ws.v_wq_all[l].ptr, ws.s_wq_all[l].ptr, wd, h * d * hd);
            adam!(ws.g_wk_all[l].ptr, ws.m_wk_all[l].ptr, ws.d_wk_all[l].ptr, ws.v_wk_all[l].ptr, ws.s_wk_all[l].ptr, wd, h * d * hd);
            adam!(ws.g_wv_all[l].ptr, ws.m_wv_all[l].ptr, ws.d_wv_all[l].ptr, ws.v_wv_all[l].ptr, ws.s_wv_all[l].ptr, wd, h * d * hd);
            adam!(ws.g_dwo[l].ptr, ws.m_wo[l].ptr, ws.d_wo[l].ptr, ws.v_dwo[l].ptr, ws.s_dwo[l].ptr, wd, d * d);
            adam!(ws.g_dw1[l].ptr, ws.m_w1[l].ptr, ws.d_w1[l].ptr, ws.v_dw1[l].ptr, ws.s_dw1[l].ptr, wd, d * ff);
            adam!(ws.g_dw2[l].ptr, ws.m_w2[l].ptr, ws.d_w2[l].ptr, ws.v_dw2[l].ptr, ws.s_dw2[l].ptr, wd, ff * d);
            // 1-D params (biases, LayerNorm scale/shift) — NOT decayed (pass 0.0).
            adam!(ws.g_db1[l].ptr, ws.m_b1[l].ptr, ws.d_b1[l].ptr, ws.v_db1[l].ptr, ws.s_db1[l].ptr, 0.0, ff);
            adam!(ws.g_db2[l].ptr, ws.m_b2[l].ptr, ws.d_b2[l].ptr, ws.v_db2[l].ptr, ws.s_db2[l].ptr, 0.0, d);
            adam!(ws.g_dln_gamma[l].ptr, ws.m_ln_gamma[l].ptr, ws.d_ln_gamma[l].ptr, ws.v_dln_gamma[l].ptr, ws.s_dln_gamma[l].ptr, 0.0, d);
            adam!(ws.g_dln_beta[l].ptr, ws.m_ln_beta[l].ptr, ws.d_ln_beta[l].ptr, ws.v_dln_beta[l].ptr, ws.s_dln_beta[l].ptr, 0.0, d);
        }
    }
}

