//! GPU forward pass for one training step — embedding lookup, the attention and
//! feed-forward sublayers, logit projection, and the cross-entropy loss. These
//! are `pub(super)` stage helpers driven by `run_step` in the parent module.
//!
//! SAFETY CONTRACT (shared by every `unsafe` block below): all `d_*`, `b_*`,
//! `g_*`, `head_*` pointers are Workspace device allocations sized for the
//! configured maximum dims, and `s <= ctx_window`, so every kernel stays within
//! its buffers; each CUDA kernel is bounded by the size arguments passed
//! alongside the pointers. Individual blocks add a note only where they do
//! something beyond "call a kernel over ws buffers with valid dims".

use super::*;

impl Model {
    // Embedding lookup + positional add for the input tokens, producing the
    // initial hidden state in b_sd[0]; returns the pointer to it.
    pub(super) fn embed_forward(&self, ws: &mut Workspace, dims: StepDims, chunk: &[u32]) -> *mut c_void {
        let StepDims { s, d, vocab, .. } = dims;
        let token_ids: Vec<i32> = chunk.iter().map(|&id| id as i32).collect();
        // SAFETY: see contract; token_ids holds exactly s i32s.
        unsafe {
            cuda_memcpy_h2d(
                ws.d_token_ids.ptr as *mut c_void,
                token_ids.as_ptr() as *const c_void,
                (s * 4) as usize,
            );
            cuda_embedding_forward(
                ws.d_token_ids.ptr as *const i32,
                ws.d_embedding.ptr as *const c_void,
                ws.d_positional.ptr as *const c_void,
                ws.b_sd[0].ptr as *mut c_void,
                s as i32,
                d as i32,
                vocab as i32,
            );
            cuda_clamp_f16(ws.b_sd[0].ptr as *mut c_void, (s * d) as i32);
        }
        ws.b_sd[0].ptr as *mut c_void
    }

    // Attention sublayer forward: LN1 → multi-head attention → residual, caching
    // the activations the backward pass needs. Takes the block input and returns
    // the post-attention residual.
    pub(super) fn attn_sublayer_forward(
        &self,
        ws: &mut Workspace,
        dims: StepDims,
        layer_idx: usize,
        d_x: *mut c_void,
    ) -> *mut c_void {
        let StepDims { s, d, h, hd, .. } = dims;

        // Save layer input (for LN1 backward + residual).
        // SAFETY: see contract.
        unsafe {
            cuda_memcpy_d2d(
                ws.layer_inputs[layer_idx].ptr as *mut c_void,
                d_x as *const c_void,
                (s * d * 2) as usize,
            );
        }

        // Layer norm 1 (pre-attention), then cache its output (= attention input).
        // SAFETY: see contract.
        unsafe {
            cuda_layer_norm(
                d_x,
                ws.b_sd[0].ptr as *mut c_void,
                ws.d_ln_gamma[layer_idx].ptr as *const c_void,
                ws.d_ln_beta[layer_idx].ptr as *const c_void,
                s as i32,
                d as i32,
                LAYER_NORM_EPS,
            );
            cuda_memcpy_d2d(
                ws.post_ff_residuals[layer_idx].ptr as *mut c_void,
                ws.b_sd[0].ptr as *const c_void,
                (s * d * 2) as usize,
            );
        }

        // Batched multi-head self-attention forward.
        // SAFETY: see contract.
        unsafe {
            cuda_attention_forward_batched(
                ws.b_sd[0].ptr as *const c_void,
                ws.d_wq_all[layer_idx].ptr as *const c_void,
                ws.d_wk_all[layer_idx].ptr as *const c_void,
                ws.d_wv_all[layer_idx].ptr as *const c_void,
                ws.head_out_all[layer_idx].ptr as *mut c_void,
                ws.head_p_all[layer_idx].ptr as *mut c_void,
                s as i32, d as i32, hd as i32, h as i32,
            );
        }

        // Concat heads: interleave by token on GPU.
        let head_out_base = ws.head_out_all[layer_idx].ptr as *mut u8;
        let head_ptrs: Vec<*mut c_void> = (0..h)
            .map(|hi| {
                // SAFETY: head hi's slice starts at byte offset hi*s*hd*2 within
                // head_out_all[layer_idx], which holds h*s*hd f16 values.
                unsafe { head_out_base.add(hi * s * hd * 2) as *mut c_void }
            })
            .collect();
        // SAFETY: see contract; head_ptrs holds exactly h device pointers.
        unsafe {
            cuda_memcpy_h2d(
                ws.d_head_ptrs.ptr as *mut c_void,
                head_ptrs.as_ptr() as *const c_void,
                head_ptrs.len() * std::mem::size_of::<*mut c_void>(),
            );
            cuda_concat_heads(
                ws.d_head_ptrs.ptr as *mut *mut c_void,
                ws.b_sd[1].ptr as *mut c_void,
                s as i32,
                h as i32,
                hd as i32,
            );
        }

        // Save concat output (for wo backward), then project through wo.
        // SAFETY: see contract.
        unsafe {
            cuda_memcpy_d2d(
                ws.concat_outputs[layer_idx].ptr as *mut c_void,
                ws.b_sd[1].ptr as *const c_void,
                (s * h * hd * 2) as usize,
            );
            cuda_matmul(
                ws.b_sd[1].ptr as *const c_void,
                ws.d_wo[layer_idx].ptr as *const c_void,
                ws.b_sd[2].ptr as *mut c_void,
                s as i32,
                (h * hd) as i32,
                d as i32,
            );
        }

        // Residual add: x + attn_out (restore x from cache), then save the
        // post-attention residual (for LN2 backward + FF residual).
        // SAFETY: see contract.
        unsafe {
            cuda_memcpy_d2d(
                ws.b_sd[3].ptr as *mut c_void,
                ws.layer_inputs[layer_idx].ptr as *const c_void,
                (s * d * 2) as usize,
            );
            cuda_add_inplace_f16(
                ws.b_sd[3].ptr as *mut c_void,
                ws.b_sd[2].ptr as *const c_void,
                (s * d) as i32,
            );
            cuda_clamp_f16(ws.b_sd[3].ptr as *mut c_void, (s * d) as i32);
            cuda_memcpy_d2d(
                ws.post_attn_residuals[layer_idx].ptr as *mut c_void,
                ws.b_sd[3].ptr as *const c_void,
                (s * d * 2) as usize,
            );
        }
        ws.b_sd[3].ptr as *mut c_void
    }

    // Feed-forward sublayer forward: LN2 → feed-forward (saving pre-gelu) →
    // residual. Takes the post-attention residual and returns the block output.
    pub(super) fn ffn_sublayer_forward(
        &self,
        ws: &mut Workspace,
        dims: StepDims,
        layer_idx: usize,
        d_x: *mut c_void,
    ) -> *mut c_void {
        let StepDims { s, d, ff, .. } = dims;

        // Layer norm 2 (pre-FF), save its output (= FF input), then FF (with the
        // pre-gelu activation saved for backward).
        // SAFETY: see contract.
        unsafe {
            cuda_layer_norm(
                d_x,
                ws.b_sd[0].ptr as *mut c_void,
                ws.d_ln_gamma[layer_idx].ptr as *const c_void,
                ws.d_ln_beta[layer_idx].ptr as *const c_void,
                s as i32,
                d as i32,
                LAYER_NORM_EPS,
            );
            cuda_memcpy_d2d(
                ws.ff_inputs[layer_idx].ptr as *mut c_void,
                ws.b_sd[0].ptr as *const c_void,
                (s * d * 2) as usize,
            );
            cuda_feed_forward_forward(
                ws.b_sd[0].ptr as *const c_void,
                ws.d_w1[layer_idx].ptr as *const c_void,
                ws.d_w2[layer_idx].ptr as *const c_void,
                ws.d_b1[layer_idx].ptr as *const c_void,
                ws.d_b2[layer_idx].ptr as *const c_void,
                ws.b_sd[1].ptr as *mut c_void,
                ws.pre_gelus[layer_idx].ptr as *mut c_void,
                s as i32,
                d as i32,
                ff as i32,
            );
        }

        // Residual add: post_attn + ff_out.
        // SAFETY: see contract.
        unsafe {
            cuda_memcpy_d2d(
                ws.b_sd[2].ptr as *mut c_void,
                d_x as *const c_void,
                (s * d * 2) as usize,
            );
            cuda_add_inplace_f16(
                ws.b_sd[2].ptr as *mut c_void,
                ws.b_sd[1].ptr as *const c_void,
                (s * d) as i32,
            );
            cuda_clamp_f16(ws.b_sd[2].ptr as *mut c_void, (s * d) as i32);
        }
        ws.b_sd[2].ptr as *mut c_void
    }

    // Project the final hidden state to logits in d_logits.
    pub(super) fn project_logits(&self, ws: &mut Workspace, dims: StepDims, d_x: *mut c_void) {
        let StepDims { s, d, vocab, .. } = dims;
        // SAFETY: see contract.
        unsafe {
            cuda_matmul(
                d_x,
                ws.d_w_output.ptr as *const c_void,
                ws.d_logits.ptr as *mut c_void,
                s as i32,
                d as i32,
                vocab as i32,
            );
            cuda_clamp_f16(ws.d_logits.ptr as *mut c_void, (s * vocab) as i32);
        }
    }

    // Cross-entropy loss over positions 0..s-1 (position i predicts token i+1),
    // plus its gradient into d_dlogits (final row zeroed — no target). Returns
    // the scalar loss.
    pub(super) fn compute_loss(
        &self,
        ws: &mut Workspace,
        dims: StepDims,
        chunk: &[u32],
        mask: Option<&[u8]>,
    ) -> f32 {
        let StepDims { s, vocab, .. } = dims;
        let flat_targets_f32: Vec<f32> = chunk[1..].iter().map(|&t| t as f32).collect();
        // SAFETY: see contract; flat_targets_f32 holds s-1 f32 targets.
        unsafe {
            cuda_memcpy_h2d(
                ws.d_targets.ptr as *mut c_void,
                flat_targets_f32.as_ptr() as *const c_void,
                (flat_targets_f32.len() * 4) as usize,
            );
        }

        // Resolve the mask + active-row count for the loss over positions 0..s-1.
        // - Pretraining (mask None): null mask, active == seq_len (s-1) — the
        //   normalizer is identical to the pre-mask behaviour.
        // - SFT (mask Some): the per-token mask's first s-1 entries are the
        //   prediction mask for these positions (entry i governs position i, which
        //   predicts chunk[i+1]); the final token has no successor and is unused
        //   here. Upload it to ws.d_mask and count the trained rows.
        let (mask_ptr, active): (*const f32, i32) = match mask {
            None => (std::ptr::null(), (s - 1) as i32),
            Some(m) => {
                let mask_f32: Vec<f32> = m[..s - 1].iter().map(|&b| b as f32).collect();
                let active = mask_f32.iter().filter(|&&x| x != 0.0).count() as i32;
                // SAFETY: see contract; mask_f32 holds s-1 f32s, ws.d_mask is sized max_s.
                unsafe {
                    cuda_memcpy_h2d(
                        ws.d_mask.ptr as *mut c_void,
                        mask_f32.as_ptr() as *const c_void,
                        (mask_f32.len() * 4) as usize,
                    );
                }
                (ws.d_mask.ptr as *const f32, active)
            }
        };

        // No trained positions in this chunk: produce no loss and a zero gradient
        // (guards the kernels' divide-by-`active`).
        if active == 0 {
            // SAFETY: see contract; zeroing the whole s*vocab f16 dlogits buffer.
            unsafe {
                cudaMemset(ws.d_dlogits.ptr as *mut c_void, 0, s * vocab * 2);
            }
            return 0.0;
        }

        // SAFETY: see contract.
        let loss = unsafe {
            cuda_cross_entropy_loss(
                ws.d_logits.ptr as *const c_void,
                ws.d_targets.ptr as *const f32,
                mask_ptr,
                (s - 1) as i32,
                vocab as i32,
                active,
            )
        };
        unsafe {
            cuda_cross_entropy_backward(
                ws.d_logits.ptr as *const c_void,
                ws.d_targets.ptr as *const f32,
                mask_ptr,
                ws.d_dlogits.ptr as *mut c_void,
                (s - 1) as i32,
                vocab as i32,
                active,
            );
            // SAFETY: offset (s-1)*vocab*2 bytes lands on the final (no-target)
            // logit row inside d_dlogits (s*vocab f16 = s*vocab*2 bytes); zero it.
            cudaMemset(
                (ws.d_dlogits.ptr as *mut c_void).add((s - 1) * vocab * 2),
                0,
                vocab * 2,
            );
        }
        loss
    }

    // Backprop through the output projection: accumulate the d_w_output gradient
    // and produce the hidden-state gradient d_dhidden; returns its pointer.
}
