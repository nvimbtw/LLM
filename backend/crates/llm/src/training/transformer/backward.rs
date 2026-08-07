//! GPU backward pass for one training step — gradients through the output
//! projection, the feed-forward and attention sublayers, and accumulation of the
//! embedding gradient, plus the per-step gradient-norm clip. These are
//! `pub(super)` stage helpers driven by `run_step` in the parent module.
//!
//! SAFETY CONTRACT (shared by every `unsafe` block below): all `d_*`, `b_*`,
//! `g_*`, `head_*` pointers are Workspace device allocations sized for the
//! configured maximum dims, and `s <= ctx_window`, so every kernel stays within
//! its buffers; each CUDA kernel is bounded by the size arguments passed
//! alongside the pointers. Individual blocks add a note only where they do
//! something beyond "call a kernel over ws buffers with valid dims".

use super::*;

impl Model {
    pub(super) fn backward_unembed(&self, ws: &mut Workspace, dims: StepDims, d_x: *mut c_void) -> *mut c_void {
        let StepDims { s, d, vocab, .. } = dims;
        // SAFETY: see contract.
        unsafe {
            if self.ties_embeddings() {
                // dE[vocab, d] = dlogits[s, vocab]^T @ x[s, d] — the same GEMM as
                // the untied case with the two operands swapped, because the tied
                // matrix is the transpose of the projection it stands in for.
                cuda_d_weight(
                    ws.d_dlogits.ptr as *const c_void,
                    d_x,
                    ws.d_dw_output.ptr,
                    s as i32,
                    vocab as i32,
                    d as i32,
                );
                // dhidden[s, d] = dlogits[s, vocab] @ E[vocab, d]. Untied this is
                // a `_bt` against `[d, vocab]`; tied the matrix already has the
                // orientation the plain matmul wants.
                cuda_matmul(
                    ws.d_dlogits.ptr as *const c_void,
                    ws.d_embedding.ptr as *const c_void,
                    ws.d_dhidden.ptr as *mut c_void,
                    s as i32,
                    vocab as i32,
                    d as i32,
                );
                // Into the same accumulator the untied path uses, holding the
                // transposed shape. It is clipped with every other gradient and
                // folded into `g_d_embedding` at the optimizer step; see
                // `Workspace::new` for why it is not summed in here.
                cuda_add_inplace_f32(
                    ws.g_dw_output.ptr,
                    ws.d_dw_output.ptr,
                    (vocab * d) as i32,
                );
            } else {
                cuda_d_weight(
                    d_x,
                    ws.d_dlogits.ptr as *const c_void,
                    ws.d_dw_output.ptr,
                    s as i32,
                    d as i32,
                    vocab as i32,
                );
                cuda_matmul_bt(
                    ws.d_dlogits.ptr as *const c_void,
                    ws.d_w_output.ptr as *const c_void,
                    ws.d_dhidden.ptr as *mut c_void,
                    s as i32,
                    vocab as i32,
                    d as i32,
                );
                cuda_add_inplace_f32(ws.g_dw_output.ptr, ws.d_dw_output.ptr, (d * vocab) as i32);
            }
        }
        ws.d_dhidden.ptr as *mut c_void
    }

    // Final LayerNorm backward: turns the gradient w.r.t. the normed hidden
    // state (what `backward_unembed` produced) into the gradient w.r.t. the last
    // block's residual stream, accumulating the norm's own gamma/beta gradients.
    // Passes the gradient straight through for a pre-v5 model that has no final
    // norm, so the block loop below sees the same thing it always did.
    pub(super) fn final_ln_backward(
        &self,
        ws: &mut Workspace,
        dims: StepDims,
        d_dh: *mut c_void,
    ) -> *mut c_void {
        if !self.has_final_ln() {
            return d_dh;
        }
        let StepDims { s, d, .. } = dims;
        // SAFETY: see contract.
        unsafe {
            cuda_layer_norm_backward(
                ws.final_ln_input.ptr as *const c_void,
                d_dh as *const c_void,
                ws.d_ln_final_gamma.ptr as *const c_void,
                ws.final_ln_dgrad.ptr as *mut c_void,
                ws.b_d[0].ptr,
                ws.b_d[1].ptr,
                s as i32,
                d as i32,
                LAYER_NORM_EPS,
            );
            cuda_add_inplace_f32(ws.g_dln_final_gamma.ptr, ws.b_d[0].ptr, d as i32);
            cuda_add_inplace_f32(ws.g_dln_final_beta.ptr, ws.b_d[1].ptr, d as i32);
            cuda_clamp_f16(ws.final_ln_dgrad.ptr as *mut c_void, (s * d) as i32);
        }
        ws.final_ln_dgrad.ptr as *mut c_void
    }

    // Feed-forward sublayer backward: FF backward → LN2 backward → residual.
    // Takes the incoming gradient and returns the gradient out of the FF sublayer
    // (which the attention sublayer backward then consumes).
    pub(super) fn ffn_sublayer_backward(
        &self,
        ws: &mut Workspace,
        dims: StepDims,
        layer_idx: usize,
        d_dx: *mut c_void,
    ) -> *mut c_void {
        let StepDims { s, d, ff, .. } = dims;

        // Feed-forward backward, accumulate its weight/bias gradients.
        // SAFETY: see contract.
        unsafe {
            cuda_feed_forward_backward(
                ws.ff_inputs[layer_idx].ptr as *const c_void,
                ws.pre_gelus[layer_idx].ptr as *const c_void,
                d_dx,
                ws.d_w1[layer_idx].ptr as *const c_void,
                ws.d_w2[layer_idx].ptr as *const c_void,
                ws.b_sd[0].ptr as *mut c_void,
                ws.b_dff[0].ptr,
                ws.b_dff[1].ptr,
                ws.b_ff.ptr,
                ws.b_d[0].ptr,
                s as i32,
                d as i32,
                ff as i32,
            );
            cuda_add_inplace_f32(ws.g_dw1[layer_idx].ptr, ws.b_dff[0].ptr, (d * ff) as i32);
            cuda_add_inplace_f32(ws.g_dw2[layer_idx].ptr, ws.b_dff[1].ptr, (ff * d) as i32);
            cuda_add_inplace_f32(ws.g_db1[layer_idx].ptr, ws.b_ff.ptr, ff as i32);
            cuda_add_inplace_f32(ws.g_db2[layer_idx].ptr, ws.b_d[0].ptr, d as i32);
            cuda_clamp_f16(ws.b_sd[0].ptr as *mut c_void, (s * d) as i32);
        }

        // Layer norm 2 backward, accumulate its gamma/beta gradients.
        // SAFETY: see contract.
        unsafe {
            cuda_layer_norm_backward(
                ws.post_attn_residuals[layer_idx].ptr as *const c_void,
                ws.b_sd[0].ptr as *const c_void,
                ws.d_ln_gamma[layer_idx].ptr as *const c_void,
                ws.ln2_dgrad.ptr as *mut c_void,
                ws.b_d[0].ptr,
                ws.b_d[1].ptr,
                s as i32,
                d as i32,
                LAYER_NORM_EPS,
            );
            cuda_add_inplace_f32(ws.g_dln_gamma[layer_idx].ptr, ws.b_d[0].ptr, d as i32);
            cuda_add_inplace_f32(ws.g_dln_beta[layer_idx].ptr, ws.b_d[1].ptr, d as i32);
            cuda_clamp_f16(ws.ln2_dgrad.ptr as *mut c_void, (s * d) as i32);
        }

        // Residual: d_dx + LN2 backprop → b_sd[2].
        // SAFETY: see contract.
        unsafe {
            cuda_memcpy_d2d(ws.b_sd[2].ptr as *mut c_void, d_dx, (s * d * 2) as usize);
            cuda_add_inplace_f16(
                ws.b_sd[2].ptr as *mut c_void,
                ws.ln2_dgrad.ptr as *const c_void,
                (s * d) as i32,
            );
            cuda_clamp_f16(ws.b_sd[2].ptr as *mut c_void, (s * d) as i32);
        }
        ws.b_sd[2].ptr as *mut c_void
    }

    // Attention sublayer backward: wo → multi-head attention → LN1, accumulating
    // weight gradients and returning the block's input gradient (for the next,
    // earlier block). Takes the gradient out of the FF sublayer.
    pub(super) fn attn_sublayer_backward(
        &self,
        ws: &mut Workspace,
        dims: StepDims,
        layer_idx: usize,
        d_dx: *mut c_void,
    ) -> *mut c_void {
        let StepDims { s, d, h, hd, .. } = dims;

        // Wo backward (accumulate its gradient), then d_wo_x = d_post_attn @ wo^T.
        // SAFETY: see contract.
        unsafe {
            cuda_d_weight(
                ws.concat_outputs[layer_idx].ptr as *const c_void,
                d_dx as *const c_void,
                ws.b_dwo.ptr,
                s as i32,
                d as i32,
                d as i32,
            );
            cuda_add_inplace_f32(ws.g_dwo[layer_idx].ptr, ws.b_dwo.ptr, (d * d) as i32);
            cuda_matmul(
                d_dx as *const c_void,
                ws.d_wo[layer_idx].ptr as *const c_void,
                ws.d_wo_x.ptr as *mut c_void,
                s as i32,
                d as i32,
                d as i32,
            );
        }

        // Batched multi-head attention backward. Weight gradients accumulate into
        // g_wq_all/g_wk_all/g_wv_all (β=1 inside the kernel); dX goes to b_sd_head.
        // SAFETY: see contract.
        unsafe {
            cuda_attention_backward_batched(
                ws.ln1_outputs[layer_idx].ptr as *const c_void,
                ws.d_wq_all[layer_idx].ptr as *const c_void,
                ws.d_wk_all[layer_idx].ptr as *const c_void,
                ws.d_wv_all[layer_idx].ptr as *const c_void,
                ws.head_p_all[layer_idx].ptr as *const c_void,
                ws.q_all[layer_idx].ptr as *const c_void,
                ws.k_all[layer_idx].ptr as *const c_void,
                ws.v_all[layer_idx].ptr as *const c_void,
                ws.d_wo_x.ptr as *const c_void,
                ws.b_sd_head.ptr as *mut c_void,
                ws.g_wq_all[layer_idx].ptr,
                ws.g_wk_all[layer_idx].ptr,
                ws.g_wv_all[layer_idx].ptr,
                s as i32, d as i32, hd as i32, h as i32,
            );
            cuda_clamp_f16(ws.b_sd_head.ptr as *mut c_void, (s * d) as i32);
        }

        // Layer norm 1 backward, accumulate its gamma/beta gradients.
        // SAFETY: see contract.
        unsafe {
            cuda_layer_norm_backward(
                ws.layer_inputs[layer_idx].ptr as *const c_void,
                ws.b_sd_head.ptr as *const c_void,
                ws.d_ln_gamma[layer_idx].ptr as *const c_void,
                ws.b_sd[1].ptr as *mut c_void,
                ws.b_d[0].ptr,
                ws.b_d[1].ptr,
                s as i32,
                d as i32,
                LAYER_NORM_EPS,
            );
            cuda_add_inplace_f32(ws.g_dln_gamma[layer_idx].ptr, ws.b_d[0].ptr, d as i32);
            cuda_add_inplace_f32(ws.g_dln_beta[layer_idx].ptr, ws.b_d[1].ptr, d as i32);
        }

        // Residual: layer_input gradient = incoming FF-sublayer gradient (d_dx)
        // + LN1 backprop (b_sd[1]).
        // SAFETY: see contract.
        unsafe {
            cuda_add_inplace_f16(
                ws.b_sd[1].ptr as *mut c_void,
                d_dx as *const c_void,
                (s * d) as i32,
            );
            cuda_clamp_f16(ws.b_sd[1].ptr as *mut c_void, (s * d) as i32);
        }
        ws.b_sd[1].ptr as *mut c_void
    }

    // Clip every accumulated gradient buffer to unit norm before the optimizer
    // step (the per-step loss-spike guard).
    pub(super) fn clip_gradients(&self, ws: &mut Workspace, _dims: StepDims, _layers: usize) {
        // One batched call over the whole gradient-buffer table instead of one
        // clip per tensor. The table (pointers + sizes, in this exact order) is
        // uploaded once by `Workspace::fill_clip_table`; the semantics are
        // unchanged, each tensor still clipped to its own norm.
        //
        // The per-tensor form cost 61 x 3 launches per step and measured as
        // almost pure launch overhead — the kernels had nearly nothing to do.
        // SAFETY: the table describes live device allocations owned by `ws`, and
        // `n_clip` is the length all three device arrays were sized with.
        unsafe {
            cuda_clip_norm_batched(
                ws.d_clip_ptrs as *mut *mut f32,
                ws.d_clip_sizes.ptr as *const i32,
                ws.d_clip_norms.ptr,
                ws.n_clip as i32,
                TRAIN_STEP_CLIP_NORM,
            );
        }
    }

    // Scatter-add the input gradient into the embedding gradient (by token id)
    // and clip it, closing out the backward pass.
    pub(super) fn accumulate_embedding_grad(&self, ws: &mut Workspace, dims: StepDims, d_dx: *mut c_void) {
        let StepDims { s, d, vocab, .. } = dims;
        // SAFETY: see contract; d_token_ids holds the s token ids scattered into.
        unsafe {
            cuda_scatter_add_embedding_grad(
                d_dx as *const c_void,
                ws.d_token_ids.ptr as *const i32,
                ws.g_d_embedding.ptr,
                s as i32,
                d as i32,
                vocab as i32,
            );
            cuda_clip_norm(ws.g_d_embedding.ptr, (vocab * d) as i32, TRAIN_STEP_CLIP_NORM);
        }
    }
}
