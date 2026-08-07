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
    fn cuda_add_inplace_f32(dst: *mut f32, src: *const f32, n: i32);
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

/// Fold a tied model's output-projection gradient into its embedding gradient,
/// leaving one accumulator for the one parameter they share. A no-op for an
/// untied model.
///
/// **Must run after the per-step clip and before the global clip and the Adam
/// step.** Both buffers are already clipped to their own unit norm by then, so
/// the sum is exactly what an untied model would have applied to its two
/// matrices — which is the point: tying changes the parameter count, not how
/// much gradient the vocabulary tables receive. Doing the merge earlier (in the
/// backward) would put both paths under a single clip and quietly change the
/// effective learning rate on the largest tensor in the model.
///
/// The layouts agree because a tied model's `g_dw_output` holds `[vocab, d]` —
/// see `backward_unembed` — not the `[d, vocab]` an untied one holds.
pub fn merge_tied_gradients(ws: &Workspace) {
    if !ws.tied {
        return;
    }
    // SAFETY: both are `Workspace` device allocations of `vocab * d` f32s.
    unsafe {
        cuda_add_inplace_f32(ws.g_d_embedding.ptr, ws.g_dw_output.ptr, (ws.vocab * ws.d) as i32);
    }
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
    // Both are live whether or not the model ties: tying merges them at the
    // optimizer step, not in the backward, so up to that point there really are
    // two gradient tensors to report.
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
    if ws.has_final_ln {
        bufs.push((ws.g_dln_final_gamma.ptr, d));
        bufs.push((ws.g_dln_final_beta.ptr, d));
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
        // A tied model has no output-projection parameter to step: its gradient
        // was folded into the embedding's by `merge_tied_gradients`, and the one
        // matrix is stepped once through the embedding entry below.
        if !ws.tied {
            adam!(ws.g_dw_output.ptr, ws.m_w_output.ptr, ws.d_w_output.ptr, ws.v_dw_output.ptr, ws.s_dw_output.ptr, wd, d * vocab);
        }
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
        // The final norm's scale/shift — 1-D, so undecayed like the per-block
        // ones. Stepped only when the model has one; a pre-v5 model's buffers
        // hold nothing meaningful and must not be written back over its weights.
        if ws.has_final_ln {
            adam!(ws.g_dln_final_gamma.ptr, ws.m_ln_final_gamma.ptr, ws.d_ln_final_gamma.ptr, ws.v_dln_final_gamma.ptr, ws.s_dln_final_gamma.ptr, 0.0, d);
            adam!(ws.g_dln_final_beta.ptr, ws.m_ln_final_beta.ptr, ws.d_ln_final_beta.ptr, ws.v_dln_final_beta.ptr, ws.s_dln_final_beta.ptr, 0.0, d);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::transformer::Model;

    // Every trained tensor must receive a gradient. A tensor whose accumulated
    // gradient is exactly zero after a real forward+backward is not being trained
    // at all — it only ever shrinks under decoupled weight decay, which looks like
    // "training happened" in the weight file but means the model is running on
    // frozen random projections.
    #[test]
    fn every_attention_tensor_receives_a_gradient() {
        let (vocab, dim, ctx, heads, ff_mult, layers) = (32, 16, 8, 2, 2, 3);
        let t = Model::init(dim, vocab, ctx, heads, ff_mult, layers);
        let mut ws = Workspace::new(&t, ctx as usize);
        ws.upload_weights(&t);
        let chunk: Vec<u32> = vec![3, 7, 1, 14, 2, 9, 5, 11];

        pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));

        let names = [
            "wq", "wk", "wv", "wo", "w1", "w2", "b1", "b2", "ln_gamma", "ln_beta",
        ];
        let mut dead = Vec::new();
        for l in 0..layers as usize {
            for (i, (g, _m, _v, _s, n)) in layer_param_bufs(&ws, l).iter().enumerate() {
                let norm = unsafe { cuda_l2norm(*g, *n as i32) };
                println!("L{l} {:<9} grad_l2 = {norm:.6e}", names[i]);
                if !(norm > 0.0) {
                    dead.push(format!("L{l}.{}", names[i]));
                }
            }
        }
        assert!(dead.is_empty(), "tensors receiving no gradient: {dead:?}");
    }

    // The final layer-norm is the third parameter this project has added between
    // the residual stream and the logits; the previous two (the absolute
    // positional table, the K projection) both shipped wired into the forward
    // pass but receiving no gradient, and in both cases the loss curve still went
    // down, so nothing caught it. A non-zero gradient on both ends is the cheapest
    // thing that would have.
    // Expand a tied model's single matrix into the explicit `[d, vocab]` output
    // projection it stands in for, giving a model that must compute exactly the
    // same thing down the untied path.
    fn untie(tied: &Model) -> Model {
        let vocab = tied.vocab();
        let d = tied.embedding_table[0].len();
        let mut untied = tied.clone();
        untied.w_output = vec![vec![0.0f32; vocab]; d];
        for v in 0..vocab {
            for j in 0..d {
                untied.w_output[j][v] = tied.embedding_table[v][j];
            }
        }
        untied
    }

    // Tying must be an identity on the arithmetic, not merely "trains without
    // crashing". An untied model whose `w_output` *is* the embedding transposed
    // has to produce the same logits and the same total gradient, so the tied
    // model's single accumulator must equal the untied model's two, added.
    //
    // This is the check that catches the failures a non-zero-gradient assert
    // cannot see: a transposed operand, a `_bt` where a plain matmul belongs, or
    // an accumulation that lands in the wrong buffer. All three would still
    // descend a loss curve, which is exactly how the K projection and the
    // absolute positional table both shipped untrained.
    #[test]
    fn a_tied_model_computes_what_the_untied_one_does() {
        let (vocab, dim, ctx, heads, ff_mult, layers) = (32, 16, 8, 2, 2, 3);
        let chunk: Vec<u32> = vec![3, 7, 1, 14, 2, 9, 5, 11];

        let tied = Model::init(dim, vocab, ctx, heads, ff_mult, layers);
        assert!(tied.ties_embeddings(), "a fresh model must tie");
        let untied = untie(&tied);
        assert!(!untied.ties_embeddings());

        let mut ws_tied = Workspace::new(&tied, ctx as usize);
        let mut ws_untied = Workspace::new(&untied, ctx as usize);

        let loss_tied = pollster::block_on(tied.train_step(&mut ws_tied, &chunk, layers as usize));
        let loss_untied =
            pollster::block_on(untied.train_step(&mut ws_untied, &chunk, layers as usize));

        assert!(
            (loss_tied - loss_untied).abs() < 1e-3,
            "same arithmetic must give the same loss: tied {loss_tied}, untied {loss_untied}"
        );

        // The tied accumulator against the untied pair, summed. `g_dw_output` is
        // `[d, vocab]` and `g_d_embedding` is `[vocab, d]`, so one is transposed
        // onto the other before comparing.
        // Merge first: tying is applied at the optimizer step, so before it the
        // tied workspace still holds the two halves separately — the same two an
        // untied model holds, in the same clipped state.
        merge_tied_gradients(&ws_tied);

        let (v, d) = (vocab as usize, dim as usize);
        let tied_grad = ws_tied.g_d_embedding.to_vec();
        let untied_emb = ws_untied.g_d_embedding.to_vec();
        let untied_out = ws_untied.g_dw_output.to_vec();

        let mut worst = 0.0f32;
        let mut scale = 0.0f32;
        for i in 0..v {
            for j in 0..d {
                let expect = untied_emb[i * d + j] + untied_out[j * v + i];
                let got = tied_grad[i * d + j];
                worst = worst.max((expect - got).abs());
                scale = scale.max(expect.abs());
            }
        }
        // Both sides run the same GEMMs on the same values, so this is tight —
        // it is not a "roughly similar" check. What little slack there is comes
        // from f16 accumulation order in the two GEMM wirings.
        assert!(
            worst <= 1e-3 * scale.max(1e-6),
            "tied gradient differs from the untied pair summed: worst {worst:e} against a \
             largest element of {scale:e}"
        );
        assert!(scale > 0.0, "the untied pair produced no gradient at all");
    }

    // Both halves of the tie must be live, and they are separable: the
    // embedding lookup writes only the rows of tokens that appear in the chunk,
    // while the output projection writes *every* row (the softmax pushes down
    // every non-target logit). So a token absent from the chunk isolates the
    // projection path, and the lookup path is read off the sparse buffer.
    #[test]
    fn both_gradient_paths_reach_the_tied_matrix() {
        let (vocab, dim, ctx, heads, ff_mult, layers) = (32, 16, 8, 2, 2, 3);
        // Token 3 appears (as the first input); token 30 never does.
        let chunk: Vec<u32> = vec![3, 7, 1, 14, 2, 9, 5, 21];

        let tied = Model::init(dim, vocab, ctx, heads, ff_mult, layers);
        let mut ws = Workspace::new(&tied, ctx as usize);
        pollster::block_on(tied.train_step(&mut ws, &chunk, layers as usize));

        let d = dim as usize;
        let row = |buf: &Vec<f32>, tok: usize| -> f32 {
            buf[tok * d..(tok + 1) * d].iter().map(|v| v * v).sum::<f32>().sqrt()
        };

        // The lookup half: sparse, and only over tokens the chunk contains.
        let lookup = ws.g_d_embedding.to_vec();
        assert!(row(&lookup, 3) > 0.0, "the embedding-lookup path is dead");
        assert_eq!(
            row(&lookup, 30),
            0.0,
            "the lookup path must not touch a token the chunk never uses"
        );

        // The projection half: dense, so it reaches a token the lookup cannot.
        let projection = ws.g_dw_output.to_vec();
        assert!(
            row(&projection, 30) > 0.0,
            "the output-projection path is dead — nothing wrote a row the lookup cannot reach"
        );

        // And the merge actually combines them into the one accumulator.
        merge_tied_gradients(&ws);
        let merged = ws.g_d_embedding.to_vec();
        assert!(row(&merged, 30) > 0.0, "the merge dropped the projection path");
        assert!(row(&merged, 3) > 0.0, "the merge dropped the lookup path");
    }

    #[test]
    fn the_final_layer_norm_receives_a_gradient() {
        let (vocab, dim, ctx, heads, ff_mult, layers) = (32, 16, 8, 2, 2, 3);
        let t = Model::init(dim, vocab, ctx, heads, ff_mult, layers);
        assert!(t.has_final_ln(), "a freshly initialized model must carry one");
        let mut ws = Workspace::new(&t, ctx as usize);
        let chunk: Vec<u32> = vec![3, 7, 1, 14, 2, 9, 5, 11];

        pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));

        for (name, ptr) in [
            ("ln_final_gamma", ws.g_dln_final_gamma.ptr),
            ("ln_final_beta", ws.g_dln_final_beta.ptr),
        ] {
            let norm = unsafe { cuda_l2norm(ptr, dim as i32) };
            assert!(norm > 0.0, "{name} received no gradient (l2 = {norm:e})");
        }
    }

    // A model loaded from a pre-v5 file has no final norm, and the step must
    // leave that alone rather than training a norm the weights never saw. The
    // empty vectors are the whole representation of "off", so this pins that the
    // GPU path agrees with them.
    #[test]
    fn a_model_without_a_final_layer_norm_trains_without_one() {
        let (vocab, dim, ctx, heads, ff_mult, layers) = (32, 16, 8, 2, 2, 2);
        let mut t = Model::init(dim, vocab, ctx, heads, ff_mult, layers);
        t.ln_final_gamma = Vec::new();
        t.ln_final_beta = Vec::new();
        assert!(!t.has_final_ln());

        let mut ws = Workspace::new(&t, ctx as usize);
        assert!(!ws.has_final_ln);
        assert_eq!(ws.n_clip, 1 + 10 * layers as usize);
        let chunk: Vec<u32> = vec![3, 7, 1, 14, 2, 9, 5, 11];

        let loss = pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));
        assert!(loss.is_finite(), "loss must stay finite without a final norm");

        for (name, ptr) in [
            ("ln_final_gamma", ws.g_dln_final_gamma.ptr),
            ("ln_final_beta", ws.g_dln_final_beta.ptr),
        ] {
            let norm = unsafe { cuda_l2norm(ptr, dim as i32) };
            assert_eq!(norm, 0.0, "{name} was written for a model that has none");
        }
    }
}
