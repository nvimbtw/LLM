//! Adam optimizer step and the global gradient-norm clip. These operate on the
//! GPU `Workspace` (not the model), so they live at `training/` level rather than
//! under `transformer/`. FFI decls are duplicated (not shared) so each module
//! declares exactly the kernels it calls.

use std::ffi::c_void;

use crate::training::gpu_workspace::Workspace;

// Adam optimizer hyperparameters (standard defaults).
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

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
        bc1: f32,
        bc2: f32,
        n: i32,
    );
    fn cuda_l2norm(v: *const f32, n: i32) -> f32;
    fn cuda_scale(v: *mut f32, s: f32, n: i32);
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
pub fn gpu_clip_gradients(ws: &Workspace, layers: usize, max_norm: f32) -> f32 {
    let bufs = grad_buffers(ws, layers);

    let mut sumsq = 0.0f64;
    for &(ptr, n) in &bufs {
        let norm = unsafe { cuda_l2norm(ptr, n as i32) } as f64;
        sumsq += norm * norm;
    }
    let total = sumsq.sqrt() as f32;

    if !total.is_finite() {
        for &(ptr, n) in &bufs {
            unsafe { cuda_scale(ptr, 0.0, n as i32) };
        }
    } else if total > max_norm {
        let scale = max_norm / total;
        for &(ptr, n) in &bufs {
            unsafe { cuda_scale(ptr, scale, n as i32) };
        }
    }
    total
}

// Adam optimizer step. `step` is the 1-based global update count (for bias
// correction); `accum_steps` is the gradient-accumulation factor so the summed
// gradient is averaged. `v_*` hold the first moment (m), `s_*` the second (v).
pub fn gpu_adam_update(ws: &Workspace, lr: f32, step: i32, accum_steps: usize, layers: usize) {
    let beta1 = ADAM_BETA1;
    let beta2 = ADAM_BETA2;
    let eps = ADAM_EPS;
    let bc1 = 1.0 - beta1.powi(step);
    let bc2 = 1.0 - beta2.powi(step);
    let gs = 1.0 / accum_steps as f32;
    let d = ws.d;
    let h = ws.h;
    let hd = ws.hd;
    let ff = ws.ff;
    let vocab = ws.vocab;

    // Cuts the repeated argument list down to (grad, master, view, m, v, n).
    // `$wm` is the f32 master (canonical); `$wv` the f16 view refreshed from it.
    macro_rules! adam {
        ($g:expr, $wm:expr, $wv:expr, $m:expr, $v:expr, $n:expr) => {
            cuda_adam($g, $wm, $wv as *mut c_void, $m, $v, gs, lr, beta1, beta2, eps, bc1, bc2, $n as i32)
        };
    }

    unsafe {
        adam!(ws.g_dw_output.ptr, ws.m_w_output.ptr, ws.d_w_output.ptr, ws.v_dw_output.ptr, ws.s_dw_output.ptr, d * vocab);
        adam!(ws.g_d_embedding.ptr, ws.m_embedding.ptr, ws.d_embedding.ptr, ws.v_d_embedding.ptr, ws.s_d_embedding.ptr, vocab * d);
        for l in 0..layers {
            adam!(ws.g_wq_all[l].ptr, ws.m_wq_all[l].ptr, ws.d_wq_all[l].ptr, ws.v_wq_all[l].ptr, ws.s_wq_all[l].ptr, h * d * hd);
            adam!(ws.g_wk_all[l].ptr, ws.m_wk_all[l].ptr, ws.d_wk_all[l].ptr, ws.v_wk_all[l].ptr, ws.s_wk_all[l].ptr, h * d * hd);
            adam!(ws.g_wv_all[l].ptr, ws.m_wv_all[l].ptr, ws.d_wv_all[l].ptr, ws.v_wv_all[l].ptr, ws.s_wv_all[l].ptr, h * d * hd);
            adam!(ws.g_dwo[l].ptr, ws.m_wo[l].ptr, ws.d_wo[l].ptr, ws.v_dwo[l].ptr, ws.s_dwo[l].ptr, d * d);
            adam!(ws.g_dw1[l].ptr, ws.m_w1[l].ptr, ws.d_w1[l].ptr, ws.v_dw1[l].ptr, ws.s_dw1[l].ptr, d * ff);
            adam!(ws.g_dw2[l].ptr, ws.m_w2[l].ptr, ws.d_w2[l].ptr, ws.v_dw2[l].ptr, ws.s_dw2[l].ptr, ff * d);
            adam!(ws.g_db1[l].ptr, ws.m_b1[l].ptr, ws.d_b1[l].ptr, ws.v_db1[l].ptr, ws.s_db1[l].ptr, ff);
            adam!(ws.g_db2[l].ptr, ws.m_b2[l].ptr, ws.d_b2[l].ptr, ws.v_db2[l].ptr, ws.s_db2[l].ptr, d);
            adam!(ws.g_dln_gamma[l].ptr, ws.m_ln_gamma[l].ptr, ws.d_ln_gamma[l].ptr, ws.v_dln_gamma[l].ptr, ws.s_dln_gamma[l].ptr, d);
            adam!(ws.g_dln_beta[l].ptr, ws.m_ln_beta[l].ptr, ws.d_ln_beta[l].ptr, ws.v_dln_beta[l].ptr, ws.s_dln_beta[l].ptr, d);
        }
    }
}

