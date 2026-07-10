use std::ffi::c_void;
use std::io::Read;
use crate::training::device_buffer::DeviceBufferF16;
use crate::training::embed::embed;
use crate::training::gpu_workspace::GpuWorkspace;

// Layer-norm numerical-stability epsilon, shared by every layer-norm forward and
// backward call.
const LAYER_NORM_EPS: f32 = 1e-5;
// Per-step gradient L2-norm clip applied inside train_step before accumulation.
const TRAIN_STEP_CLIP_NORM: f32 = 1.0;
// Adam optimizer hyperparameters (standard defaults).
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

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
pub struct LayerRow {
    pub name: String,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}

#[derive(serde::Serialize, Clone)]
pub struct ModelHeader {
    pub dimensions: i64,
    pub vocab_size: i64,
    pub max_seq_len: i64,
    pub heads: i64,
    pub ff_multiplier: i64,
    pub layers: i64,
}

/// Read just the fixed-size header of a saved model (`model.bin` in `dir`)
/// without loading any weights — used to report a model's architecture.
pub fn read_model_header(dir: &str) -> Result<ModelHeader, std::io::Error> {
    let path = format!("{}/model.bin", dir);
    let mut f = std::fs::File::open(&path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"GORK" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
    }
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    let _version = u32::from_le_bytes(buf);
    let read_i64 = |f: &mut std::fs::File| -> Result<i64, std::io::Error> {
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    };
    let dimensions = read_i64(&mut f)?;
    let vocab_size = read_i64(&mut f)?;
    let max_seq_len = read_i64(&mut f)?;
    let heads = read_i64(&mut f)?;
    let ff_multiplier = read_i64(&mut f)?;
    let layers = read_i64(&mut f)?;
    validate_header_fields(dimensions, vocab_size, max_seq_len, heads, ff_multiplier, layers)?;
    Ok(ModelHeader { dimensions, vocab_size, max_seq_len, heads, ff_multiplier, layers })
}

// A model header comes from a file (external input), so its fields are validated
// at the boundary rather than trusted. A non-positive field would produce a
// negative-sized allocation, and heads not dividing dimensions (or being zero)
// would make `dimensions / heads` panic or compute a wrong head_dim.
fn validate_header_fields(
    dimensions: i64,
    vocab: i64,
    ctx: i64,
    heads: i64,
    ff_multiplier: i64,
    layers: i64,
) -> Result<(), std::io::Error> {
    let invalid = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    for (name, value) in [
        ("dimensions", dimensions),
        ("vocab_size", vocab),
        ("max_seq_len", ctx),
        ("heads", heads),
        ("ff_multiplier", ff_multiplier),
        ("layers", layers),
    ] {
        if value <= 0 {
            return Err(invalid(format!(
                "model header field `{}` must be positive, got {}",
                name, value
            )));
        }
    }
    if dimensions % heads != 0 {
        return Err(invalid(format!(
            "model header: dimensions ({}) must be divisible by heads ({})",
            dimensions, heads
        )));
    }
    Ok(())
}

/// A decoder-only transformer held as plain nested `Vec`s on the host. This is
/// the canonical CPU-side copy that gets (de)serialized and used for inference;
/// during training the weights are mirrored to GPU buffers in `GpuWorkspace`.
pub struct Transformer {
    pub embedding_table: Vec<Vec<f32>>,
    pub positional_table: Vec<Vec<f32>>,
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
    pub w_output: Vec<Vec<f32>>,
}

impl Transformer {
    /// Build a fresh model with randomly-initialized weights. `dimensions` must
    /// be divisible by `heads`; the feed-forward width is `dimensions * ff_multiplier`.
    pub fn init(dimensions: i64, vocab: i64, ctx_window: i64, heads: i64, ff_multiplier: i64, layers: i64) -> Transformer {
        let head_dim: i64 = dimensions / heads;
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

        Transformer {
            embedding_table: embed(dimensions, vocab),
            positional_table: embed(dimensions, ctx_window),
            wq, wk, wv, wo, w1, w2, b1, b2, ln_gamma, ln_beta,
            heads, head_dim, layers,
            w_output: embed(vocab, dimensions),
        }
    }

    /// Run the forward pass over a token sequence and return the final hidden
    /// state for every position. Only the last `ctx_window` tokens are used (the
    /// model has no positions beyond its table), so a growing prompt is fine.
    pub async fn forward_seq(
        &self,
        tokens: &[u32],
        layers: usize,
        ctx_window: usize,
    ) -> Vec<Vec<f32>> {
        let seq = if tokens.len() > ctx_window {
            &tokens[tokens.len() - ctx_window..]
        } else {
            tokens
        };

        let input: Vec<Vec<f32>> = seq
            .iter()
            .map(|&token_id| self.embedding_table[token_id as usize].clone())
            .collect();

        let positioned: Vec<Vec<f32>> = input
            .iter()
            .enumerate()
            .map(|(pos, token_embed)| {
                token_embed
                    .iter()
                    .zip(self.positional_table[pos].iter())
                    .map(|(t, p)| t + p)
                    .collect()
            })
            .collect();

        let mut x = positioned;

        for layer_idx in 0..layers {
            let x_norm = self.layer_norm(&x, &self.ln_gamma[layer_idx], &self.ln_beta[layer_idx]).await;
            let attn = self.self_attention(&x_norm, &self.wq[layer_idx], &self.wk[layer_idx], &self.wv[layer_idx], &self.wo[layer_idx]).await;
            for i in 0..x.len() {
                for j in 0..x[i].len() {
                    x[i][j] += attn[i][j];
                }
            }

            let x_norm = self.layer_norm(&x, &self.ln_gamma[layer_idx], &self.ln_beta[layer_idx]).await;
            let ff = self.feed_forward(&x_norm, &self.w1[layer_idx], &self.w2[layer_idx], &self.b1[layer_idx], &self.b2[layer_idx]).await;
            for i in 0..x.len() {
                for j in 0..x[i].len() {
                    x[i][j] += ff[i][j];
                }
            }
        }

        x
    }

    async fn layer_norm(&self, input: &[Vec<f32>], gamma: &[f32], beta: &[f32]) -> Vec<Vec<f32>> {
        let rows = input.len();
        let cols = input[0].len();
        let flat: Vec<f32> = input.iter().flatten().cloned().collect();
        let d_in = DeviceBufferF16::from_f32_slice(&flat);
        let d_gamma = DeviceBufferF16::from_f32_slice(gamma);
        let d_beta = DeviceBufferF16::from_f32_slice(beta);
        let d_out = DeviceBufferF16::alloc(rows * cols);

        unsafe {
            cuda_layer_norm(
                d_in.ptr as *const c_void,
                d_out.ptr as *mut c_void,
                d_gamma.ptr as *const c_void,
                d_beta.ptr as *const c_void,
                rows as i32,
                cols as i32,
                LAYER_NORM_EPS,
            );
        }

        d_out
            .to_f32_vec()
            .chunks(cols)
            .map(|row: &[f32]| row.to_vec())
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    async fn self_attention(
        &self, input: &[Vec<f32>],
        wq: &[Vec<Vec<f32>>], wk: &[Vec<Vec<f32>>], wv: &[Vec<Vec<f32>>],
        wo: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let seq_len = input.len();
        let d_model = input[0].len();
        let flat_embeddings: Vec<f32> = input.iter().flatten().cloned().collect();
        let d_embeddings = DeviceBufferF16::from_f32_slice(&flat_embeddings);

        let mut head_outputs: Vec<Vec<f32>> = Vec::new();

        for i in 0..self.heads {
            let h = i as usize;

            let d_wq = DeviceBufferF16::from_f32_slice(
                &wq[h].iter().flatten().cloned().collect::<Vec<f32>>(),
            );
            let d_wk = DeviceBufferF16::from_f32_slice(
                &wk[h].iter().flatten().cloned().collect::<Vec<f32>>(),
            );
            let d_wv = DeviceBufferF16::from_f32_slice(
                &wv[h].iter().flatten().cloned().collect::<Vec<f32>>(),
            );
            let d_out = DeviceBufferF16::alloc(seq_len * self.head_dim as usize);

            unsafe {
                cuda_attention_head(
                    d_embeddings.ptr as *const c_void,
                    d_wq.ptr as *const c_void,
                    d_wk.ptr as *const c_void,
                    d_wv.ptr as *const c_void,
                    d_out.ptr as *mut c_void,
                    seq_len as i32,
                    d_model as i32,
                    self.head_dim as i32,
                );
            }

            head_outputs.push(d_out.to_f32_vec());
        }

        let head_dim = self.head_dim as usize;
        let heads = self.heads as usize;

        let mut concatenated: Vec<f32> = vec![0.0f32; seq_len * heads * head_dim];
        for token in 0..seq_len {
            for h in 0..heads {
                let src_start = token * head_dim;
                let dst_start = token * (heads * head_dim) + h * head_dim;
                concatenated[dst_start..dst_start + head_dim]
                    .copy_from_slice(&head_outputs[h][src_start..src_start + head_dim]);
            }
        }

        let flat_wo: Vec<f32> = wo.iter().flatten().cloned().collect();
        let d_concat = DeviceBufferF16::from_f32_slice(&concatenated);
        let d_wo = DeviceBufferF16::from_f32_slice(&flat_wo);
        let d_result = DeviceBufferF16::alloc(seq_len * d_model);

        unsafe {
            cuda_matmul(
                d_concat.ptr as *const c_void,
                d_wo.ptr as *const c_void,
                d_result.ptr as *mut c_void,
                seq_len as i32,
                (heads * head_dim) as i32,
                d_model as i32,
            );
        }

        d_result
            .to_f32_vec()
            .chunks(d_model)
            .map(|row: &[f32]| row.to_vec())
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    async fn feed_forward(
        &self, input: &[Vec<f32>],
        w1: &[Vec<f32>], w2: &[Vec<f32>], b1: &[f32], b2: &[f32],
    ) -> Vec<Vec<f32>> {
        let seq_len = input.len();
        let d_model = input[0].len();
        // FF width comes from the weight shape (w1 is [d_model][d_ff]), not a
        // fixed multiple: ff_multiplier is configurable, so assuming ×4 would
        // read w1 with the wrong stride for any other setting.
        let d_ff = w1[0].len();

        let flat_input: Vec<f32> = input.iter().flatten().cloned().collect();
        let flat_w1: Vec<f32> = w1.iter().flatten().cloned().collect();
        let flat_w2: Vec<f32> = w2.iter().flatten().cloned().collect();

        let d_input = DeviceBufferF16::from_f32_slice(&flat_input);
        let d_w1 = DeviceBufferF16::from_f32_slice(&flat_w1);
        let d_w2 = DeviceBufferF16::from_f32_slice(&flat_w2);
        let d_b1 = DeviceBufferF16::from_f32_slice(b1);
        let d_b2 = DeviceBufferF16::from_f32_slice(b2);
        let d_out = DeviceBufferF16::alloc(seq_len * d_model);

        unsafe {
            cuda_feed_forward(
                d_input.ptr as *const c_void,
                d_w1.ptr as *const c_void,
                d_w2.ptr as *const c_void,
                d_b1.ptr as *const c_void,
                d_b2.ptr as *const c_void,
                d_out.ptr as *mut c_void,
                seq_len as i32,
                d_model as i32,
                d_ff as i32,
            );
        }

        d_out
            .to_f32_vec()
            .chunks(d_model)
            .map(|row: &[f32]| row.to_vec())
            .collect()
    }

    /// Project hidden states to vocab-sized logits via the output embedding.
    pub async fn unembed(&self, hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = hidden.len();
        let d_model = hidden[0].len();
        let vocab = self.w_output[0].len();

        let flat_hidden: Vec<f32> = hidden.iter().flatten().cloned().collect();
        let flat_w_output: Vec<f32> = self.w_output.iter().flatten().cloned().collect();

        let d_hidden = DeviceBufferF16::from_f32_slice(&flat_hidden);
        let d_w_output = DeviceBufferF16::from_f32_slice(&flat_w_output);
        let d_logits = DeviceBufferF16::alloc(seq_len * vocab);

        unsafe {
            cuda_matmul(
                d_hidden.ptr as *const c_void,
                d_w_output.ptr as *const c_void,
                d_logits.ptr as *mut c_void,
                seq_len as i32,
                d_model as i32,
                vocab as i32,
            );
        }

        d_logits
            .to_f32_vec()
            .chunks(vocab)
            .map(|row: &[f32]| row.to_vec())
            .collect()
    }

    /// Run one forward+backward pass over a single `chunk`, accumulating gradients
    /// into `ws` (which it mutates) and returning the scalar cross-entropy loss.
    /// `chunk` must hold at least 2 tokens (position i is trained to predict token
    /// i+1). Weights are not updated here — the caller drives the optimizer step.
    pub async fn train_step(&self, ws: &mut GpuWorkspace, chunk: &[u32], layers: usize) -> f32 {
        let dims = StepDims {
            s: chunk.len(),
            d: self.wo[0].len(),
            h: self.heads as usize,
            hd: self.head_dim as usize,
            ff: self.w1[0][0].len(),
            vocab: self.w_output[0].len(),
        };
        ws.ensure_layer_caches(layers);

        let mut d_x = self.embed_forward(ws, dims, chunk);
        for layer_idx in 0..layers {
            let x = self.attn_sublayer_forward(ws, dims, layer_idx, d_x);
            d_x = self.ffn_sublayer_forward(ws, dims, layer_idx, x);
        }
        self.project_logits(ws, dims, d_x);
        let loss = self.compute_loss(ws, dims, chunk);

        let mut d_dx = self.backward_unembed(ws, dims, d_x);
        for layer_idx in (0..layers).rev() {
            let g = self.ffn_sublayer_backward(ws, dims, layer_idx, d_dx);
            d_dx = self.attn_sublayer_backward(ws, dims, layer_idx, g);
        }
        self.clip_gradients(ws, dims, layers);
        self.accumulate_embedding_grad(ws, dims, d_dx);

        loss
    }

    // ── train_step stage helpers ─────────────────────────────────────────────
    //
    // SAFETY CONTRACT (shared by every `unsafe` block in the helpers below):
    // all `d_*`, `b_*`, `g_*`, `head_*` pointers are GpuWorkspace device
    // allocations sized for the configured maximum dims, and `s <= ctx_window`,
    // so every kernel stays within its buffers. Each CUDA kernel is bounded by
    // the size arguments (`s`, `d`, `vocab`, …) passed alongside the pointers.
    // Individual blocks add a note only where they do something beyond "call a
    // kernel over ws buffers with valid dims" (raw pointer arithmetic).

    // Embedding lookup + positional add for the input tokens, producing the
    // initial hidden state in b_sd[0]; returns the pointer to it.
    fn embed_forward(&self, ws: &mut GpuWorkspace, dims: StepDims, chunk: &[u32]) -> *mut c_void {
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
    fn attn_sublayer_forward(
        &self,
        ws: &mut GpuWorkspace,
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
    fn ffn_sublayer_forward(
        &self,
        ws: &mut GpuWorkspace,
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
    fn project_logits(&self, ws: &mut GpuWorkspace, dims: StepDims, d_x: *mut c_void) {
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
    fn compute_loss(&self, ws: &mut GpuWorkspace, dims: StepDims, chunk: &[u32]) -> f32 {
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
        // Pretraining trains on every position: null mask, and active == seq_len
        // (s-1) so the normalizer is identical to the pre-mask behaviour. The SFT
        // path (ISSUE-0033) passes a real mask + trained-row count instead.
        // SAFETY: see contract.
        let loss = unsafe {
            cuda_cross_entropy_loss(
                ws.d_logits.ptr as *const c_void,
                ws.d_targets.ptr as *const f32,
                std::ptr::null(),
                (s - 1) as i32,
                vocab as i32,
                (s - 1) as i32,
            )
        };
        unsafe {
            cuda_cross_entropy_backward(
                ws.d_logits.ptr as *const c_void,
                ws.d_targets.ptr as *const f32,
                std::ptr::null(),
                ws.d_dlogits.ptr as *mut c_void,
                (s - 1) as i32,
                vocab as i32,
                (s - 1) as i32,
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
    fn backward_unembed(&self, ws: &mut GpuWorkspace, dims: StepDims, d_x: *mut c_void) -> *mut c_void {
        let StepDims { s, d, vocab, .. } = dims;
        // SAFETY: see contract.
        unsafe {
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
        ws.d_dhidden.ptr as *mut c_void
    }

    // Feed-forward sublayer backward: FF backward → LN2 backward → residual.
    // Takes the incoming gradient and returns the gradient out of the FF sublayer
    // (which the attention sublayer backward then consumes).
    fn ffn_sublayer_backward(
        &self,
        ws: &mut GpuWorkspace,
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
                ws.b_dd.ptr as *mut c_void,
                ws.b_d[0].ptr,
                ws.b_d[1].ptr,
                s as i32,
                d as i32,
                LAYER_NORM_EPS,
            );
            cuda_add_inplace_f32(ws.g_dln_gamma[layer_idx].ptr, ws.b_d[0].ptr, d as i32);
            cuda_add_inplace_f32(ws.g_dln_beta[layer_idx].ptr, ws.b_d[1].ptr, d as i32);
            cuda_clamp_f16(ws.b_dd.ptr as *mut c_void, (s * d) as i32);
        }

        // Residual: d_dx + LN2 backprop → b_sd[2].
        // SAFETY: see contract.
        unsafe {
            cuda_memcpy_d2d(ws.b_sd[2].ptr as *mut c_void, d_dx, (s * d * 2) as usize);
            cuda_add_inplace_f16(
                ws.b_sd[2].ptr as *mut c_void,
                ws.b_dd.ptr as *const c_void,
                (s * d) as i32,
            );
            cuda_clamp_f16(ws.b_sd[2].ptr as *mut c_void, (s * d) as i32);
        }
        ws.b_sd[2].ptr as *mut c_void
    }

    // Attention sublayer backward: wo → multi-head attention → LN1, accumulating
    // weight gradients and returning the block's input gradient (for the next,
    // earlier block). Takes the gradient out of the FF sublayer.
    fn attn_sublayer_backward(
        &self,
        ws: &mut GpuWorkspace,
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
                ws.b_dd.ptr,
                s as i32,
                d as i32,
                d as i32,
            );
            cuda_add_inplace_f32(ws.g_dwo[layer_idx].ptr, ws.b_dd.ptr, (d * d) as i32);
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
                ws.post_ff_residuals[layer_idx].ptr as *const c_void,
                ws.d_wq_all[layer_idx].ptr as *const c_void,
                ws.d_wk_all[layer_idx].ptr as *const c_void,
                ws.d_wv_all[layer_idx].ptr as *const c_void,
                ws.head_p_all[layer_idx].ptr as *const c_void,
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
    fn clip_gradients(&self, ws: &mut GpuWorkspace, dims: StepDims, layers: usize) {
        let StepDims { d, h, hd, ff, vocab, .. } = dims;
        let max_norm = TRAIN_STEP_CLIP_NORM;
        // SAFETY: see contract; each buffer's element count is passed alongside.
        unsafe {
            cuda_clip_norm(ws.g_dw_output.ptr, (d * vocab) as i32, max_norm);
            for layer_idx in 0..layers {
                cuda_clip_norm(ws.g_wq_all[layer_idx].ptr, (h * d * hd) as i32, max_norm);
                cuda_clip_norm(ws.g_wk_all[layer_idx].ptr, (h * d * hd) as i32, max_norm);
                cuda_clip_norm(ws.g_wv_all[layer_idx].ptr, (h * d * hd) as i32, max_norm);
                cuda_clip_norm(ws.g_dwo[layer_idx].ptr, (d * d) as i32, max_norm);
                cuda_clip_norm(ws.g_dw1[layer_idx].ptr, (d * ff) as i32, max_norm);
                cuda_clip_norm(ws.g_dw2[layer_idx].ptr, (ff * d) as i32, max_norm);
                cuda_clip_norm(ws.g_db1[layer_idx].ptr, ff as i32, max_norm);
                cuda_clip_norm(ws.g_db2[layer_idx].ptr, d as i32, max_norm);
                cuda_clip_norm(ws.g_dln_gamma[layer_idx].ptr, d as i32, max_norm);
                cuda_clip_norm(ws.g_dln_beta[layer_idx].ptr, d as i32, max_norm);
            }
        }
    }

    // Scatter-add the input gradient into the embedding gradient (by token id)
    // and clip it, closing out the backward pass.
    fn accumulate_embedding_grad(&self, ws: &mut GpuWorkspace, dims: StepDims, d_dx: *mut c_void) {
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
    /// Serialize the model to `path` in the versioned `GORK` binary format.
    pub fn save_to_path(&self, path: &str) -> Result<(), std::io::Error> {
        let mut f = std::fs::File::create(path)?;
        write_model(self, &mut f)
    }

    /// Serialize the model to `model.bin` inside `dir`.
    pub fn save(&self, dir: &str) -> Result<(), std::io::Error> {
        self.save_to_path(&format!("{}/model.bin", dir))
    }

    /// Load a model from `model.bin` inside `dir`, accepting any supported format
    /// version. Errors on bad magic, truncation, or an invalid header.
    pub fn load(dir: &str) -> Result<Self, std::io::Error> {
        let mut f = std::fs::File::open(format!("{}/model.bin", dir))?;
        read_model(&mut f)
    }
}

// ── Model (de)serialization ─────────────────────────────────────────────────
// One definition of the on-disk format, shared by the file-backed (`save`/
// `load`) and in-memory (`to_bytes`/`from_bytes`) paths by being generic over
// the writer/reader — so the byte layout is described exactly once.

const MODEL_MAGIC: &[u8; 4] = b"GORK";
const MODEL_VERSION: u32 = 3;

fn write_2d<W: std::io::Write>(w: &mut W, m: &[Vec<f32>]) -> Result<(), std::io::Error> {
    w.write_all(&(m.len() as u64).to_le_bytes())?;
    w.write_all(&(m[0].len() as u64).to_le_bytes())?;
    for row in m {
        for &v in row {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_1d<W: std::io::Write>(w: &mut W, v: &[f32]) -> Result<(), std::io::Error> {
    w.write_all(&(v.len() as u64).to_le_bytes())?;
    for &x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn write_model<W: std::io::Write>(t: &Transformer, w: &mut W) -> Result<(), std::io::Error> {
    w.write_all(MODEL_MAGIC)?;
    w.write_all(&MODEL_VERSION.to_le_bytes())?;
    let d = t.wo[0].len() as i64;
    let vocab = t.w_output[0].len() as i64;
    let ctx = t.positional_table.len() as i64;
    let ff = t.w1[0][0].len() as i64;
    for field in [d, vocab, ctx, t.heads, ff, t.layers] {
        w.write_all(&field.to_le_bytes())?;
    }
    write_2d(w, &t.embedding_table)?;
    write_2d(w, &t.positional_table)?;
    for layer in 0..t.layers as usize {
        for head in 0..t.heads as usize {
            write_2d(w, &t.wq[layer][head])?;
            write_2d(w, &t.wk[layer][head])?;
            write_2d(w, &t.wv[layer][head])?;
        }
    }
    for mat in &t.wo { write_2d(w, mat)?; }
    for mat in &t.w1 { write_2d(w, mat)?; }
    for mat in &t.w2 { write_2d(w, mat)?; }
    for v in &t.b1 { write_1d(w, v)?; }
    for v in &t.b2 { write_1d(w, v)?; }
    for v in &t.ln_gamma { write_1d(w, v)?; }
    for v in &t.ln_beta { write_1d(w, v)?; }
    write_2d(w, &t.w_output)?;
    Ok(())
}

fn read_u32<R: std::io::Read>(r: &mut R) -> Result<u32, std::io::Error> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i64<R: std::io::Read>(r: &mut R) -> Result<i64, std::io::Error> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}

fn read_u64<R: std::io::Read>(r: &mut R) -> Result<u64, std::io::Error> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_2d<R: std::io::Read>(r: &mut R) -> Result<Vec<Vec<f32>>, std::io::Error> {
    let rows = read_u64(r)? as usize;
    let cols = read_u64(r)? as usize;
    let mut buf = vec![0u8; rows * cols * 4];
    r.read_exact(&mut buf)?;
    // chunks_exact(4) yields exactly-4-byte slices, so try_into can't fail here.
    let data: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(data.chunks(cols).map(|c| c.to_vec()).collect())
}

fn read_1d<R: std::io::Read>(r: &mut R) -> Result<Vec<f32>, std::io::Error> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len * 4];
    r.read_exact(&mut buf)?;
    // chunks_exact(4) yields exactly-4-byte slices, so try_into can't fail here.
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

struct ModelFileHeader {
    version: u32,
    heads: i64,
    head_dim: i64,
    layers: i64,
}

// Parse and validate the fixed-size header: magic, version, and the six i64
// dimension fields. `layers` was added in v2; v1 files are assumed to be 6-layer.
fn read_header<R: std::io::Read>(r: &mut R) -> Result<ModelFileHeader, std::io::Error> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MODEL_MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
    }
    let version = read_u32(r)?;
    let dimensions = read_i64(r)?;
    let vocab = read_i64(r)?;
    let ctx = read_i64(r)?;
    let heads = read_i64(r)?;
    let ff_multiplier = read_i64(r)?;
    let layers = if version >= 2 { read_i64(r)? } else { 6 };
    validate_header_fields(dimensions, vocab, ctx, heads, ff_multiplier, layers)?;
    Ok(ModelFileHeader {
        version,
        heads,
        head_dim: dimensions / heads,
        layers,
    })
}

// Per-head Q/K/V weights. v3 stores them per layer ([layers][heads][d][hd]);
// legacy v1/v2 stored a single layer's weights, broadcast to every layer.
#[allow(clippy::type_complexity)]
fn read_qkv_weights<R: std::io::Read>(
    r: &mut R,
    version: u32,
    layers: usize,
    heads: usize,
) -> Result<(Vec<Vec<Vec<Vec<f32>>>>, Vec<Vec<Vec<Vec<f32>>>>, Vec<Vec<Vec<Vec<f32>>>>), std::io::Error>
{
    let read_layer = |r: &mut R| -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>), std::io::Error> {
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        for _ in 0..heads {
            wq.push(read_2d(r)?);
            wk.push(read_2d(r)?);
            wv.push(read_2d(r)?);
        }
        Ok((wq, wk, wv))
    };

    let mut wq = Vec::new();
    let mut wk = Vec::new();
    let mut wv = Vec::new();
    if version >= 3 {
        for _ in 0..layers {
            let (lq, lk, lv) = read_layer(r)?;
            wq.push(lq);
            wk.push(lk);
            wv.push(lv);
        }
    } else {
        let (sq, sk, sv) = read_layer(r)?;
        for _ in 0..layers {
            wq.push(sq.clone());
            wk.push(sk.clone());
            wv.push(sv.clone());
        }
    }
    Ok((wq, wk, wv))
}

struct LayerBlockWeights {
    wo: Vec<Vec<Vec<f32>>>,
    w1: Vec<Vec<Vec<f32>>>,
    w2: Vec<Vec<Vec<f32>>>,
    b1: Vec<Vec<f32>>,
    b2: Vec<Vec<f32>>,
    ln_gamma: Vec<Vec<f32>>,
    ln_beta: Vec<Vec<f32>>,
}

// The per-layer block tensors (wo, w1, w2, biases, layer-norm scale/shift). Same
// v3-per-layer vs legacy-broadcast rule as the QKV weights.
fn read_block_weights<R: std::io::Read>(
    r: &mut R,
    version: u32,
    layers: usize,
) -> Result<LayerBlockWeights, std::io::Error> {
    let mut b = LayerBlockWeights {
        wo: Vec::new(), w1: Vec::new(), w2: Vec::new(),
        b1: Vec::new(), b2: Vec::new(), ln_gamma: Vec::new(), ln_beta: Vec::new(),
    };
    if version >= 3 {
        for _ in 0..layers { b.wo.push(read_2d(r)?); }
        for _ in 0..layers { b.w1.push(read_2d(r)?); }
        for _ in 0..layers { b.w2.push(read_2d(r)?); }
        for _ in 0..layers { b.b1.push(read_1d(r)?); }
        for _ in 0..layers { b.b2.push(read_1d(r)?); }
        for _ in 0..layers { b.ln_gamma.push(read_1d(r)?); }
        for _ in 0..layers { b.ln_beta.push(read_1d(r)?); }
    } else {
        let (wo, w1, w2) = (read_2d(r)?, read_2d(r)?, read_2d(r)?);
        let (b1, b2) = (read_1d(r)?, read_1d(r)?);
        let (ln_gamma, ln_beta) = (read_1d(r)?, read_1d(r)?);
        for _ in 0..layers {
            b.wo.push(wo.clone());
            b.w1.push(w1.clone());
            b.w2.push(w2.clone());
            b.b1.push(b1.clone());
            b.b2.push(b2.clone());
            b.ln_gamma.push(ln_gamma.clone());
            b.ln_beta.push(ln_beta.clone());
        }
    }
    Ok(b)
}

fn read_model<R: std::io::Read>(r: &mut R) -> Result<Transformer, std::io::Error> {
    let header = read_header(r)?;
    let embedding_table = read_2d(r)?;
    let positional_table = read_2d(r)?;
    let (wq, wk, wv) = read_qkv_weights(r, header.version, header.layers as usize, header.heads as usize)?;
    let blocks = read_block_weights(r, header.version, header.layers as usize)?;
    let w_output = read_2d(r)?;
    Ok(Transformer {
        embedding_table,
        positional_table,
        wq, wk, wv,
        wo: blocks.wo,
        heads: header.heads,
        head_dim: header.head_dim,
        layers: header.layers,
        w1: blocks.w1,
        w2: blocks.w2,
        b1: blocks.b1,
        b2: blocks.b2,
        ln_gamma: blocks.ln_gamma,
        ln_beta: blocks.ln_beta,
        w_output,
    })
}

unsafe extern "C" {
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
    );

    fn cuda_cross_entropy_loss(
        logits: *const c_void,
        targets: *const f32,
        mask: *const f32,
        seq_len: i32,
        vocab: i32,
        active: i32,
    ) -> f32;

    fn cuda_attention_head(
        embeddings: *const c_void,
        wq: *const c_void,
        wk: *const c_void,
        wv: *const c_void,
        out: *mut c_void,
        seq_len: i32,
        d_model: i32,
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

    fn cuda_clamp_f16(buf: *mut c_void, n: i32);

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
        positional_table: *const c_void,
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
        P_all: *mut c_void,
        seq: i32, d: i32, hd: i32, h: i32,
    );

    fn cuda_attention_backward_batched(
        X: *const c_void,
        wq_all: *const c_void,
        wk_all: *const c_void,
        wv_all: *const c_void,
        P_all: *const c_void,
        dout_full: *const c_void,
        dX: *mut c_void,
        dwq_all: *mut f32,
        dwk_all: *mut f32,
        dwv_all: *mut f32,
        seq: i32, d: i32, hd: i32, h: i32,
    );

    fn cuda_adam(
        g: *const f32,
        w: *mut c_void,
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

// Every trainable gradient buffer as (ptr, element count). Shared by the
// clipping pass and (implicitly) mirrors what the Adam step touches.
fn grad_buffers(ws: &GpuWorkspace, layers: usize) -> Vec<(*mut f32, usize)> {
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
pub fn gpu_clip_gradients(ws: &GpuWorkspace, layers: usize, max_norm: f32) -> f32 {
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
pub fn gpu_adam_update(ws: &GpuWorkspace, lr: f32, step: i32, accum_steps: usize, layers: usize) {
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

    // Cuts the repeated argument list down to (grad, weight, m, v, n).
    macro_rules! adam {
        ($g:expr, $w:expr, $m:expr, $v:expr, $n:expr) => {
            cuda_adam($g, $w as *mut c_void, $m, $v, gs, lr, beta1, beta2, eps, bc1, bc2, $n as i32)
        };
    }

    unsafe {
        adam!(ws.g_dw_output.ptr, ws.d_w_output.ptr, ws.v_dw_output.ptr, ws.s_dw_output.ptr, d * vocab);
        adam!(ws.g_d_embedding.ptr, ws.d_embedding.ptr, ws.v_d_embedding.ptr, ws.s_d_embedding.ptr, vocab * d);
        for l in 0..layers {
            adam!(ws.g_wq_all[l].ptr, ws.d_wq_all[l].ptr, ws.v_wq_all[l].ptr, ws.s_wq_all[l].ptr, h * d * hd);
            adam!(ws.g_wk_all[l].ptr, ws.d_wk_all[l].ptr, ws.v_wk_all[l].ptr, ws.s_wk_all[l].ptr, h * d * hd);
            adam!(ws.g_wv_all[l].ptr, ws.d_wv_all[l].ptr, ws.v_wv_all[l].ptr, ws.s_wv_all[l].ptr, h * d * hd);
            adam!(ws.g_dwo[l].ptr, ws.d_wo[l].ptr, ws.v_dwo[l].ptr, ws.s_dwo[l].ptr, d * d);
            adam!(ws.g_dw1[l].ptr, ws.d_w1[l].ptr, ws.v_dw1[l].ptr, ws.s_dw1[l].ptr, d * ff);
            adam!(ws.g_dw2[l].ptr, ws.d_w2[l].ptr, ws.v_dw2[l].ptr, ws.s_dw2[l].ptr, ff * d);
            adam!(ws.g_db1[l].ptr, ws.d_b1[l].ptr, ws.v_db1[l].ptr, ws.s_db1[l].ptr, ff);
            adam!(ws.g_db2[l].ptr, ws.d_b2[l].ptr, ws.v_db2[l].ptr, ws.s_db2[l].ptr, d);
            adam!(ws.g_dln_gamma[l].ptr, ws.d_ln_gamma[l].ptr, ws.v_dln_gamma[l].ptr, ws.s_dln_gamma[l].ptr, d);
            adam!(ws.g_dln_beta[l].ptr, ws.d_ln_beta[l].ptr, ws.v_dln_beta[l].ptr, ws.s_dln_beta[l].ptr, d);
        }
    }
}

pub fn sync_weights_from_gpu(ws: &GpuWorkspace, t: &mut Transformer) {
    let d = ws.d;
    let h = ws.h;
    let hd = ws.hd;
    let ff = ws.ff;
    let vocab = ws.vocab;
    let layers = t.layers as usize;

    let w = &mut t.w_output;
    let flat = ws.d_w_output.to_f32_vec();
    for i in 0..d {
        for j in 0..vocab {
            w[i][j] = flat[i * vocab + j];
        }
    }
    let emb = &mut t.embedding_table;
    let flat_emb = ws.d_embedding.to_f32_vec();
    for i in 0..vocab {
        for j in 0..d {
            emb[i][j] = flat_emb[i * d + j];
        }
    }
    for layer in 0..layers {
        // Contiguous per-head QKV weights: [h * d * hd]
        let flat_q = ws.d_wq_all[layer].to_f32_vec();
        let flat_k = ws.d_wk_all[layer].to_f32_vec();
        let flat_v = ws.d_wv_all[layer].to_f32_vec();
        for hi in 0..h {
            let off = hi * d * hd;
            for i in 0..d {
                for j in 0..hd {
                    t.wq[layer][hi][i][j] = flat_q[off + i * hd + j];
                    t.wk[layer][hi][i][j] = flat_k[off + i * hd + j];
                    t.wv[layer][hi][i][j] = flat_v[off + i * hd + j];
                }
            }
        }
        let flat = ws.d_wo[layer].to_f32_vec();
        for i in 0..d {
            for j in 0..d {
                t.wo[layer][i][j] = flat[i * d + j];
            }
        }
        let flat = ws.d_w1[layer].to_f32_vec();
        for i in 0..d {
            for j in 0..ff {
                t.w1[layer][i][j] = flat[i * ff + j];
            }
        }
        let flat = ws.d_w2[layer].to_f32_vec();
        for i in 0..ff {
            for j in 0..d {
                t.w2[layer][i][j] = flat[i * d + j];
            }
        }
        let flat = ws.d_b1[layer].to_f32_vec();
        for j in 0..ff {
            t.b1[layer][j] = flat[j];
        }
        let flat = ws.d_b2[layer].to_f32_vec();
        for j in 0..d {
            t.b2[layer][j] = flat[j];
        }
        let flat = ws.d_ln_gamma[layer].to_f32_vec();
        for j in 0..d {
            t.ln_gamma[layer][j] = flat[j];
        }
        let flat = ws.d_ln_beta[layer].to_f32_vec();
        for j in 0..d {
            t.ln_beta[layer][j] = flat[j];
        }
    }
}

impl Transformer {
    /// Serialize the model to an in-memory buffer (same format as `save`).
    pub fn to_bytes(&self) -> Result<Vec<u8>, std::io::Error> {
        let mut buf = Vec::new();
        write_model(self, &mut buf)?;
        Ok(buf)
    }

    /// Deserialize a model from a byte buffer produced by `to_bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, std::io::Error> {
        read_model(&mut std::io::Cursor::new(bytes))
    }

    pub fn to_layer_rows(&self) -> Vec<LayerRow> {
        let mut rows = Vec::new();
        let push_2d = |rows: &mut Vec<LayerRow>, name: &str, m: &[Vec<f32>]| {
            let shape = vec![m.len() as i64, m[0].len() as i64];
            let mut data = Vec::with_capacity(m.len() * m[0].len() * 4);
            for row in m {
                for &v in row {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
            rows.push(LayerRow { name: name.to_string(), shape, data });
        };
        let push_1d = |rows: &mut Vec<LayerRow>, name: &str, v: &[f32]| {
            let shape = vec![v.len() as i64];
            let mut data = Vec::with_capacity(v.len() * 4);
            for &x in v {
                data.extend_from_slice(&x.to_le_bytes());
            }
            rows.push(LayerRow { name: name.to_string(), shape, data });
        };
        push_2d(&mut rows, "embedding", &self.embedding_table);
        push_2d(&mut rows, "positional", &self.positional_table);
        let h = self.heads as usize;
        let d = self.wo[0].len();
        let hd = self.head_dim as usize;
        for l in 0..self.layers as usize {
            let mut flat_wq = Vec::with_capacity(h * d * hd * 4);
            let mut flat_wk = Vec::with_capacity(h * d * hd * 4);
            let mut flat_wv = Vec::with_capacity(h * d * hd * 4);
            for head in 0..h {
                for i in 0..d {
                    for j in 0..hd {
                        flat_wq.extend_from_slice(&self.wq[l][head][i][j].to_le_bytes());
                        flat_wk.extend_from_slice(&self.wk[l][head][i][j].to_le_bytes());
                        flat_wv.extend_from_slice(&self.wv[l][head][i][j].to_le_bytes());
                    }
                }
            }
            rows.push(LayerRow {
                name: format!("h.{}.attn.wq", l),
                shape: vec![h as i64, d as i64, hd as i64],
                data: flat_wq,
            });
            rows.push(LayerRow {
                name: format!("h.{}.attn.wk", l),
                shape: vec![h as i64, d as i64, hd as i64],
                data: flat_wk,
            });
            rows.push(LayerRow {
                name: format!("h.{}.attn.wv", l),
                shape: vec![h as i64, d as i64, hd as i64],
                data: flat_wv,
            });
            push_2d(&mut rows, &format!("h.{}.attn.wo", l), &self.wo[l]);
            push_2d(&mut rows, &format!("h.{}.ffn.w1", l), &self.w1[l]);
            push_2d(&mut rows, &format!("h.{}.ffn.w2", l), &self.w2[l]);
            push_1d(&mut rows, &format!("h.{}.ffn.b1", l), &self.b1[l]);
            push_1d(&mut rows, &format!("h.{}.ffn.b2", l), &self.b2[l]);
            push_1d(&mut rows, &format!("h.{}.ln.gamma", l), &self.ln_gamma[l]);
            push_1d(&mut rows, &format!("h.{}.ln.beta", l), &self.ln_beta[l]);
        }
        push_2d(&mut rows, "unembed", &self.w_output);
        rows
    }

    pub fn from_layer_rows(header: &ModelHeader, rows: &[LayerRow]) -> Result<Self, std::io::Error> {
        use std::collections::HashMap;
        let d = header.dimensions as usize;
        let _vocab = header.vocab_size as usize;
        let _ctx = header.max_seq_len as usize;
        let h = header.heads as usize;
        let hd = d / h;
        let _ff = d * header.ff_multiplier as usize;
        let l = header.layers as usize;

        let map: HashMap<&str, &LayerRow> = rows.iter().map(|r| (r.name.as_str(), r)).collect();

        let get_2d = |map: &HashMap<&str, &LayerRow>, name: &str| -> Result<Vec<Vec<f32>>, std::io::Error> {
            let row = map.get(name).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing layer: {}", name))
            })?;
            let r = row.shape[0] as usize;
            let c = row.shape[1] as usize;
            let flat: Vec<f32> = row.data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            if flat.len() != r * c {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("layer {} data size mismatch", name)));
            }
            Ok(flat.chunks(c).map(|c| c.to_vec()).collect())
        };

        let get_1d = |map: &HashMap<&str, &LayerRow>, name: &str| -> Result<Vec<f32>, std::io::Error> {
            let row = map.get(name).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing layer: {}", name))
            })?;
            Ok(row.data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };

        let _get_3d_heads = |map: &HashMap<&str, &LayerRow>, name_prefix: &str| -> Result<Vec<Vec<Vec<f32>>>, std::io::Error> {
            let mut result = Vec::new();
            for hi in 0..h {
                let key = format!("{}.{}", name_prefix, hi);
                let row = map.get(key.as_str()).unwrap_or_else(|| {
                    // If per-head rows don't exist, the data is stored as a single 3D blob
                    map.get(name_prefix).unwrap()
                });
                // Single 3D blob case: all heads in one row
                if result.is_empty() && h > 1 && row.shape.len() == 3 {
                    let data: Vec<f32> = row.data.chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    let heads = row.shape[0] as usize;
                    let rows_d = row.shape[1] as usize;
                    let cols = row.shape[2] as usize;
                    for hi in 0..heads {
                        let mut mat = vec![vec![0.0f32; cols]; rows_d];
                        let off = hi * rows_d * cols;
                        for i in 0..rows_d {
                            for j in 0..cols {
                                mat[i][j] = data[off + i * cols + j];
                            }
                        }
                        result.push(mat);
                    }
                    return Ok(result);
                }
            }
            // Single head or per-head rows
            for hi in 0..h {
                let key = format!("{}.{}", name_prefix, hi);
                if let Ok(mat) = get_2d(map, &key) {
                    result.push(mat);
                }
            }
            if result.is_empty() {
                // Fallback: try the 3D blob approach
                if let Some(row) = map.get(name_prefix) {
                    if row.shape.len() == 3 {
                        let data: Vec<f32> = row.data.chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        let heads = row.shape[0] as usize;
                        let rows_d = row.shape[1] as usize;
                        let cols = row.shape[2] as usize;
                        for hi in 0..heads {
                            let mut mat = vec![vec![0.0f32; cols]; rows_d];
                            let off = hi * rows_d * cols;
                            for i in 0..rows_d {
                                for j in 0..cols {
                                    mat[i][j] = data[off + i * cols + j];
                                }
                            }
                            result.push(mat);
                        }
                    }
                }
            }
            Ok(result)
        };

        let embedding_table = get_2d(&map, "embedding")?;
        let positional_table = get_2d(&map, "positional")?;

        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let mut wo = Vec::new();
        let mut w1 = Vec::new();
        let mut w2 = Vec::new();
        let mut b1 = Vec::new();
        let mut b2 = Vec::new();
        let mut ln_gamma = Vec::new();
        let mut ln_beta = Vec::new();

        for li in 0..l {
            // wq: single 3D blob [heads][d][hd] or per-head 2D blobs
            let prefix = format!("h.{}.attn", li);

            // Read QKV as flat 3D blob from the combined name
            let read_qkv = |map: &HashMap<&str, &LayerRow>, name: &str| -> Result<Vec<Vec<Vec<f32>>>, std::io::Error> {
                let row = map.get(name).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing layer: {}", name))
                })?;
                if row.shape.len() != 3 {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                        format!("expected 3D shape for {}", name)));
                }
                let heads_d = row.shape[0] as usize;
                let rows_d = row.shape[1] as usize;
                let cols = row.shape[2] as usize;
                let data: Vec<f32> = row.data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let mut result = Vec::new();
                for hi in 0..heads_d {
                    let mut mat = vec![vec![0.0f32; cols]; rows_d];
                    let off = hi * rows_d * cols;
                    for i in 0..rows_d {
                        for j in 0..cols {
                            mat[i][j] = data[off + i * cols + j];
                        }
                    }
                    result.push(mat);
                }
                Ok(result)
            };

            wq.push(read_qkv(&map, &format!("{}.wq", prefix))?);
            wk.push(read_qkv(&map, &format!("{}.wk", prefix))?);
            wv.push(read_qkv(&map, &format!("{}.wv", prefix))?);
            wo.push(get_2d(&map, &format!("{}.wo", prefix))?);
            w1.push(get_2d(&map, &format!("h.{}.ffn.w1", li))?);
            w2.push(get_2d(&map, &format!("h.{}.ffn.w2", li))?);
            b1.push(get_1d(&map, &format!("h.{}.ffn.b1", li))?);
            b2.push(get_1d(&map, &format!("h.{}.ffn.b2", li))?);
            ln_gamma.push(get_1d(&map, &format!("h.{}.ln.gamma", li))?);
            ln_beta.push(get_1d(&map, &format!("h.{}.ln.beta", li))?);
        }

        let w_output = get_2d(&map, "unembed")?;

        Ok(Transformer {
            embedding_table, positional_table,
            wq, wk, wv, wo,
            heads: header.heads,
            head_dim: hd as i64,
            layers: header.layers,
            w1, w2, b1, b2, ln_gamma, ln_beta,
            w_output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::gpu_workspace::GpuWorkspace;

    #[test]
    fn test_embedding_forward_cpu_vs_gpu() {
        let vocab = 10;
        let d = 4;
        let ctx = 8;
        let heads = 2;
        let ff_mult = 2;
        let layers = 1;

        let t = Transformer::init(d, vocab, ctx, heads, ff_mult, layers);
        let ws = GpuWorkspace::new(&t, ctx as usize);

        let chunk: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let s = chunk.len();
        let du = d as usize;

        // CPU embedding forward
        let flat_emb: Vec<f32> = chunk
            .iter()
            .flat_map(|&id| t.embedding_table[id as usize].clone())
            .collect();
        let mut expected = vec![0.0f32; s * du];
        for pos in 0..s {
            for j in 0..du {
                expected[pos * du + j] = flat_emb[pos * du + j] + t.positional_table[pos][j];
            }
        }

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
                ws.d_positional.ptr as *const std::ffi::c_void,
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
        let d = 3;
        let ctx = 8;
        let heads = 2;
        let ff_mult = 2;
        let layers = 1;

        let t = Transformer::init(d, vocab, ctx, heads, ff_mult, layers);
        let ws = GpuWorkspace::new(&t, ctx as usize);

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

        let t = Transformer::init(d, vocab, ctx, heads, ff_mult, layers);
        let mut ws = GpuWorkspace::new(&t, ctx as usize);

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

        let t = Transformer::init(d, vocab, ctx, heads, ff_mult, layers);
        let mut ws = GpuWorkspace::new(&t, ctx as usize);

        // Record pre-SGD embedding weights
        let pre_emb: Vec<f32> = t.embedding_table.iter().flatten().copied().collect();

        let chunk: Vec<u32> = (0..8).map(|i| i as u32 % (vocab as u32)).collect();
        let _loss = pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));

        let _loss = pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));
        gpu_adam_update(&ws, 0.001, 1, 1, layers as usize);
        ws.zero_accumulators();

        let mut t_synced = Transformer {
            embedding_table: t.embedding_table.clone(),
            positional_table: t.positional_table.clone(),
            wq: t.wq.clone(), wk: t.wk.clone(), wv: t.wv.clone(),
            wo: t.wo.clone(), w1: t.w1.clone(), w2: t.w2.clone(),
            b1: t.b1.clone(), b2: t.b2.clone(),
            ln_gamma: t.ln_gamma.clone(), ln_beta: t.ln_beta.clone(),
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

        let t = Transformer::init(d, vocab, ctx, heads, ff_mult, layers);
        let mut ws = GpuWorkspace::new(&t, ctx as usize);

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

    // ── Non-GPU tests: Transformer::init ──

    #[test]
    fn test_init_dimensions() {
        let t = Transformer::init(32, 100, 64, 4, 2, 2);
        assert_eq!(t.embedding_table.len(), 100);
        assert_eq!(t.embedding_table[0].len(), 32);
        assert_eq!(t.positional_table.len(), 64);
        assert_eq!(t.positional_table[0].len(), 32);
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
        assert_eq!(t.w_output.len(), 32);
        assert_eq!(t.w_output[0].len(), 100);
    }

    #[test]
    fn test_init_single_layer() {
        let t = Transformer::init(16, 50, 32, 2, 4, 1);
        assert_eq!(t.layers, 1);
        assert_eq!(t.wq.len(), 1);
        assert_eq!(t.wq[0].len(), 2);
        let ff = 16 * 4;
        assert_eq!(t.w1[0][0].len(), ff);
        assert_eq!(t.b1[0].len(), ff);
    }

    // ── Non-GPU tests: serialization round-trips ──

    #[test]
    fn test_to_bytes_from_bytes_roundtrip() {
        let t1 = Transformer::init(16, 50, 32, 2, 2, 1);
        let bytes = t1.to_bytes().unwrap();
        let t2 = Transformer::from_bytes(&bytes).unwrap();

        assert_eq!(t1.embedding_table, t2.embedding_table);
        assert_eq!(t1.positional_table, t2.positional_table);
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
        let t1 = Transformer::init(32, 100, 64, 4, 2, 3);
        let bytes = t1.to_bytes().unwrap();
        let t2 = Transformer::from_bytes(&bytes).unwrap();

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
        assert_eq!(t1.positional_table, t2.positional_table);
        assert_eq!(t1.w_output, t2.w_output);
    }

    #[test]
    fn test_from_bytes_rejects_bad_magic() {
        let bad = b"\xFF\xFF\xFF\xFF\x03\x00\x00\x00";
        assert!(Transformer::from_bytes(bad).is_err());
    }

    #[test]
    fn test_from_bytes_rejects_truncated() {
        let bad = b"GORK\x03\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00";
        assert!(Transformer::from_bytes(bad).is_err());
    }

    #[test]
    fn test_to_layer_rows_from_layer_rows_roundtrip() {
        let t1 = Transformer::init(16, 50, 32, 2, 2, 1);
        let header = ModelHeader {
            dimensions: 16,
            vocab_size: 50,
            max_seq_len: 32,
            heads: 2,
            ff_multiplier: 2,
            layers: 1,
        };
        let rows = t1.to_layer_rows();
        let t2 = Transformer::from_layer_rows(&header, &rows).unwrap();

        assert_eq!(t1.embedding_table, t2.embedding_table);
        assert_eq!(t1.positional_table, t2.positional_table);
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
        let t1 = Transformer::init(32, 100, 64, 4, 2, 3);
        let header = ModelHeader {
            dimensions: 32,
            vocab_size: 100,
            max_seq_len: 64,
            heads: 4,
            ff_multiplier: 2,
            layers: 3,
        };
        let rows = t1.to_layer_rows();
        let t2 = Transformer::from_layer_rows(&header, &rows).unwrap();

        for l in 0..3 {
            assert_eq!(t1.wq[l], t2.wq[l], "wq layer {} mismatch", l);
            assert_eq!(t1.wk[l], t2.wk[l], "wk layer {} mismatch", l);
            assert_eq!(t1.wv[l], t2.wv[l], "wv layer {} mismatch", l);
            assert_eq!(t1.wo[l], t2.wo[l], "wo layer {} mismatch", l);
            assert_eq!(t1.w1[l], t2.w1[l], "w1 layer {} mismatch", l);
            assert_eq!(t1.w2[l], t2.w2[l], "w2 layer {} mismatch", l);
        }
        assert_eq!(t1.embedding_table, t2.embedding_table);
        assert_eq!(t1.positional_table, t2.positional_table);
        assert_eq!(t1.w_output, t2.w_output);
    }

    #[test]
    fn test_to_layer_rows_correct_names() {
        let t = Transformer::init(8, 20, 16, 2, 2, 2);
        let rows = t.to_layer_rows();
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"embedding"));
        assert!(names.contains(&"positional"));
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
        assert_eq!(names.len(), 23);
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
        assert!(Transformer::from_layer_rows(&header, &rows).is_err());
    }
}
