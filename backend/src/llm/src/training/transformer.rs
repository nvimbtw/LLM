use std::ffi::c_void;
use std::io::Read;
use crate::training::device_buffer::DeviceBufferF16;
use crate::training::embed::embed;
use crate::training::gpu_workspace::GpuWorkspace;

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
    Ok(ModelHeader { dimensions, vocab_size, max_seq_len, heads, ff_multiplier, layers })
}

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

    pub async fn forward(
        &self,
        per_word: &[Vec<u32>],
        layers: usize,
        ctx_window: usize,
    ) -> (Vec<Vec<f32>>, Vec<u32>) {
        // Flatten per-word tokens into a single sequence
        let corpus: Vec<u32> = per_word.iter().flatten().copied().collect();
        let mut all_hidden = Vec::new();
        let mut all_targets = Vec::new();

        for chunk in corpus.chunks(ctx_window) {
            if chunk.len() < 2 {
                continue;
            }

            let input: Vec<Vec<f32>> = chunk
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

        // position i predicts chunk[i+1]; drop the last (no target)
        all_hidden.extend(x[..x.len() - 1].iter().cloned());
        all_targets.extend(chunk[1..].iter().copied());
        }

        (all_hidden, all_targets)
    }

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
                1e-5f32,
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
        let d_ff = d_model * 4;

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

    pub async fn train_step(&self, ws: &mut GpuWorkspace, chunk: &[u32], layers: usize) -> f32 {
        let s = chunk.len();
        let d = self.wo[0].len();
        let h = self.heads as usize;
        let hd = self.head_dim as usize;
        let ff = self.w1[0][0].len();
        let vocab = self.w_output[0].len();

        ws.ensure_layer_caches(layers);

        // ── 1. Embedding forward on GPU: lookup + positional ──
        let token_ids: Vec<i32> = chunk.iter().map(|&id| id as i32).collect();
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
        let mut d_x: *mut c_void = ws.b_sd[0].ptr as *mut c_void;

        for layer_idx in 0..layers {
            // ── Save layer input (for LN1 backward + residual) ──
            unsafe {
                cuda_memcpy_d2d(
                    ws.layer_inputs[layer_idx].ptr as *mut c_void,
                    d_x as *const c_void,
                    (s * d * 2) as usize,
                );
            }

            // ── Layer norm 1 (pre-attention) ──
            unsafe {
                cuda_layer_norm(
                    d_x,
                    ws.b_sd[0].ptr as *mut c_void,
                    ws.d_ln_gamma[layer_idx].ptr as *const c_void,
                    ws.d_ln_beta[layer_idx].ptr as *const c_void,
                    s as i32,
                    d as i32,
                    1e-5f32,
                );
            }
            // Cache LN1 output (= attention input) for backward
            unsafe {
                cuda_memcpy_d2d(
                    ws.post_ff_residuals[layer_idx].ptr as *mut c_void,
                    ws.b_sd[0].ptr as *const c_void,
                    (s * d * 2) as usize,
                );
            }

            // ── Batched multi-head self-attention forward ──
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

            // Concat heads: interleave by token on GPU
            let head_out_base = ws.head_out_all[layer_idx].ptr as *mut u8;
            let head_ptrs: Vec<*mut c_void> = (0..h)
                .map(|hi| unsafe {
                    head_out_base.add(hi * s * hd * 2) as *mut c_void
                })
                .collect();
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

            // Save concat output for wo backward
            unsafe {
                cuda_memcpy_d2d(
                    ws.concat_outputs[layer_idx].ptr as *mut c_void,
                    ws.b_sd[1].ptr as *const c_void,
                    (s * h * hd * 2) as usize,
                );
            }

            // wo projection
            unsafe {
                cuda_matmul(
                    ws.b_sd[1].ptr as *const c_void,
                    ws.d_wo[layer_idx].ptr as *const c_void,
                    ws.b_sd[2].ptr as *mut c_void,
                    s as i32,
                    (h * hd) as i32,
                    d as i32,
                );
            }

            // ── Residual add: x + attn_out (restore x from cache) ──
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
            }
            d_x = ws.b_sd[3].ptr as *mut c_void;

            // ── Save post-attn residual (for LN2 backward + FF residual) ──
            unsafe {
                cuda_memcpy_d2d(
                    ws.post_attn_residuals[layer_idx].ptr as *mut c_void,
                    d_x as *const c_void,
                    (s * d * 2) as usize,
                );
            }

            // ── Layer norm 2 (pre-FF) ──
            unsafe {
                cuda_layer_norm(
                    d_x,
                    ws.b_sd[0].ptr as *mut c_void,
                    ws.d_ln_gamma[layer_idx].ptr as *const c_void,
                    ws.d_ln_beta[layer_idx].ptr as *const c_void,
                    s as i32,
                    d as i32,
                    1e-5f32,
                );
            }
            // Save FF input (= LN2 output)
            unsafe {
                cuda_memcpy_d2d(
                    ws.ff_inputs[layer_idx].ptr as *mut c_void,
                    ws.b_sd[0].ptr as *const c_void,
                    (s * d * 2) as usize,
                );
            }

            // ── Feed-forward with pre-gelu save ──
            unsafe {
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

            // ── Residual add: post_attn + ff_out ──
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
            d_x = ws.b_sd[2].ptr as *mut c_void;
        }

        // ── 2. Compute logits and loss ──
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

        // ── 3. Compute loss and dlogits on GPU ──
        let flat_targets_f32: Vec<f32> = chunk[1..].iter().map(|&t| t as f32).collect();
        unsafe {
            cuda_memcpy_h2d(
                ws.d_targets.ptr as *mut c_void,
                flat_targets_f32.as_ptr() as *const c_void,
                (flat_targets_f32.len() * 4) as usize,
            );
        }
        let loss = unsafe {
            cuda_cross_entropy_loss(
                ws.d_logits.ptr as *const c_void,
                ws.d_targets.ptr as *const f32,
                (s - 1) as i32,
                vocab as i32,
            )
        };
        unsafe {
            cuda_cross_entropy_backward(
                ws.d_logits.ptr as *const c_void,
                ws.d_targets.ptr as *const f32,
                ws.d_dlogits.ptr as *mut c_void,
                (s - 1) as i32,
                vocab as i32,
            );
            cudaMemset(
                (ws.d_dlogits.ptr as *mut c_void).add((s - 1) * vocab as usize * 2),
                0,
                (vocab as usize * 2) as usize,
            );
        }

        // ── 4. Backward through unembed ──
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
        }
        unsafe {
            cuda_add_inplace_f32(ws.g_dw_output.ptr, ws.d_dw_output.ptr, (d * vocab) as i32);
        }

        let mut d_dx: *mut c_void = ws.d_dhidden.ptr as *mut c_void;

        // ── 5. Backward through layers (in reverse) ──
        for layer_idx in (0..layers).rev() {
            // 5a. Feed-forward backward
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
            }
            unsafe {
                cuda_add_inplace_f32(ws.g_dw1[layer_idx].ptr, ws.b_dff[0].ptr, (d * ff) as i32);
                cuda_add_inplace_f32(ws.g_dw2[layer_idx].ptr, ws.b_dff[1].ptr, (ff * d) as i32);
                cuda_add_inplace_f32(ws.g_db1[layer_idx].ptr, ws.b_ff.ptr, ff as i32);
                cuda_add_inplace_f32(ws.g_db2[layer_idx].ptr, ws.b_d[0].ptr, d as i32);
            }
            unsafe { cuda_clamp_f16(ws.b_sd[0].ptr as *mut c_void, (s * d) as i32); }

            // 5b. Layer norm 2 backward
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
                    1e-5f32,
                );
            }
            unsafe {
                cuda_add_inplace_f32(ws.g_dln_gamma[layer_idx].ptr, ws.b_d[0].ptr, d as i32);
                cuda_add_inplace_f32(ws.g_dln_beta[layer_idx].ptr, ws.b_d[1].ptr, d as i32);
            }
            unsafe { cuda_clamp_f16(ws.b_dd.ptr as *mut c_void, (s * d) as i32); }

            // Residual: d_dx + LN2 backprop
            unsafe {
                cuda_memcpy_d2d(
                    ws.b_sd[2].ptr as *mut c_void,
                    d_dx,
                    (s * d * 2) as usize,
                );
                cuda_add_inplace_f16(
                    ws.b_sd[2].ptr as *mut c_void,
                    ws.b_dd.ptr as *const c_void,
                    (s * d) as i32,
                );
            }
            unsafe { cuda_clamp_f16(ws.b_sd[2].ptr as *mut c_void, (s * d) as i32); }

            // 5c. Wo backward
            unsafe {
                cuda_d_weight(
                    ws.concat_outputs[layer_idx].ptr as *const c_void,
                    ws.b_sd[2].ptr as *const c_void,
                    ws.b_dd.ptr,
                    s as i32,
                    d as i32,
                    d as i32,
                );
            }
            unsafe {
                cuda_add_inplace_f32(ws.g_dwo[layer_idx].ptr, ws.b_dd.ptr, (d * d) as i32);
            }

            // d_wo_x = d_post_attn @ wo^T
            unsafe {
                cuda_matmul(
                    ws.b_sd[2].ptr as *const c_void,
                    ws.d_wo[layer_idx].ptr as *const c_void,
                    ws.d_wo_x.ptr as *mut c_void,
                    s as i32,
                    d as i32,
                    d as i32,
                );
            }

            // ── Batched multi-head attention backward ──
            // Weight gradients are accumulated into g_wq_all/g_wk_all/g_wv_all
            // (β=1 inside the kernel). The dX gradient goes to b_sd_head.
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
            }
            unsafe { cuda_clamp_f16(ws.b_sd_head.ptr as *mut c_void, (s * d) as i32); }

            // 5d. Layer norm 1 backward
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
                    1e-5f32,
                );
            }
            unsafe {
                cuda_add_inplace_f32(ws.g_dln_gamma[layer_idx].ptr, ws.b_d[0].ptr, d as i32);
                cuda_add_inplace_f32(ws.g_dln_beta[layer_idx].ptr, ws.b_d[1].ptr, d as i32);
            }

            // Residual: layer_input gradient = residual (b_sd[2]) + LN1 backprop (b_sd[1])
            unsafe {
                cuda_add_inplace_f16(
                    ws.b_sd[1].ptr as *mut c_void,
                    ws.b_sd[2].ptr as *const c_void,
                    (s * d) as i32,
                );
                cuda_clamp_f16(ws.b_sd[1].ptr as *mut c_void, (s * d) as i32);
            }
            d_dx = ws.b_sd[1].ptr as *mut c_void;
        }

        // ── 6. Clip gradient norms on GPU ──
        let max_norm = 1.0f32;
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

        // ── 7. Accumulate embedding gradients on GPU, then clip ──
        unsafe {
            cuda_scatter_add_embedding_grad(
                d_dx as *const c_void,
                ws.d_token_ids.ptr as *const i32,
                ws.g_d_embedding.ptr,
                s as i32,
                d as i32,
                vocab as i32,
            );
            cuda_clip_norm(ws.g_d_embedding.ptr, (vocab * d) as i32, max_norm);
        }

        loss
    }
    pub fn save_to_path(&self, path: &str) -> Result<(), std::io::Error> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"GORK")?;
        f.write_all(&3u32.to_le_bytes())?;
        let d = self.wo[0].len() as i64;
        let vocab = self.w_output[0].len() as i64;
        let ctx = self.positional_table.len() as i64;
        let ff = self.w1[0][0].len() as i64;
        f.write_all(&d.to_le_bytes())?;
        f.write_all(&vocab.to_le_bytes())?;
        f.write_all(&ctx.to_le_bytes())?;
        f.write_all(&self.heads.to_le_bytes())?;
        f.write_all(&ff.to_le_bytes())?;
        f.write_all(&self.layers.to_le_bytes())?;
        let write_2d = |f: &mut std::fs::File, m: &[Vec<f32>]| -> Result<(), std::io::Error> {
            let rows = m.len() as u64;
            let cols = m[0].len() as u64;
            f.write_all(&rows.to_le_bytes())?;
            f.write_all(&cols.to_le_bytes())?;
            for row in m {
                for &v in row {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
            Ok(())
        };
        let write_1d = |f: &mut std::fs::File, v: &[f32]| -> Result<(), std::io::Error> {
            let len = v.len() as u64;
            f.write_all(&len.to_le_bytes())?;
            for &x in v {
                f.write_all(&x.to_le_bytes())?;
            }
            Ok(())
        };
        write_2d(&mut f, &self.embedding_table)?;
        write_2d(&mut f, &self.positional_table)?;
        for layer_idx in 0..self.layers as usize {
            for head_idx in 0..self.heads as usize {
                write_2d(&mut f, &self.wq[layer_idx][head_idx])?;
                write_2d(&mut f, &self.wk[layer_idx][head_idx])?;
                write_2d(&mut f, &self.wv[layer_idx][head_idx])?;
            }
        }
        for mat in &self.wo { write_2d(&mut f, mat)?; }
        for mat in &self.w1 { write_2d(&mut f, mat)?; }
        for mat in &self.w2 { write_2d(&mut f, mat)?; }
        for v in &self.b1 { write_1d(&mut f, v)?; }
        for v in &self.b2 { write_1d(&mut f, v)?; }
        for v in &self.ln_gamma { write_1d(&mut f, v)?; }
        for v in &self.ln_beta { write_1d(&mut f, v)?; }
        write_2d(&mut f, &self.w_output)?;
        Ok(())
    }

    pub fn save(&self, dir: &str) -> Result<(), std::io::Error> {
        self.save_to_path(&format!("{}/model.bin", dir))
    }

    pub fn load(dir: &str) -> Result<Self, std::io::Error> {
        use std::io::Read;
        let path = format!("{}/model.bin", dir);
        let mut f = std::fs::File::open(&path)?;
        let read_u32 = |f: &mut std::fs::File| -> Result<u32, std::io::Error> {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            Ok(u32::from_le_bytes(buf))
        };
        let read_i64 = |f: &mut std::fs::File| -> Result<i64, std::io::Error> {
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf)?;
            Ok(i64::from_le_bytes(buf))
        };
        let read_u64 = |f: &mut std::fs::File| -> Result<u64, std::io::Error> {
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf)?;
            Ok(u64::from_le_bytes(buf))
        };
        let read_2d = |f: &mut std::fs::File| -> Result<Vec<Vec<f32>>, std::io::Error> {
            let rows = read_u64(f)? as usize;
            let cols = read_u64(f)? as usize;
            let mut buf = vec![0u8; rows * cols * 4];
            f.read_exact(&mut buf)?;
            let data: Vec<f32> = buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(data.chunks(cols).map(|c| c.to_vec()).collect())
        };
        let read_1d = |f: &mut std::fs::File| -> Result<Vec<f32>, std::io::Error> {
            let len = read_u64(f)? as usize;
            let mut buf = vec![0u8; len * 4];
            f.read_exact(&mut buf)?;
            Ok(buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };
        // verify magic
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"GORK" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }
        let version = read_u32(&mut f)?;
        let dimensions = read_i64(&mut f)?;
        let _vocab = read_i64(&mut f)?;
        let _ctx_window = read_i64(&mut f)?;
        let heads = read_i64(&mut f)?;
        let _ff_multiplier = read_i64(&mut f)?;
        let layers = if version >= 2 { read_i64(&mut f)? } else { 6 };
        let head_dim = dimensions / heads;
        let embedding_table = read_2d(&mut f)?;
        let positional_table = read_2d(&mut f)?;
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let h = heads as usize;
        let l = layers as usize;
        if version >= 3 {
            // Per-layer format: [layers][heads][d][hd]
            for _ in 0..l {
                let mut layer_wq = Vec::new();
                let mut layer_wk = Vec::new();
                let mut layer_wv = Vec::new();
                for _ in 0..h {
                    layer_wq.push(read_2d(&mut f)?);
                    layer_wk.push(read_2d(&mut f)?);
                    layer_wv.push(read_2d(&mut f)?);
                }
                wq.push(layer_wq);
                wk.push(layer_wk);
                wv.push(layer_wv);
            }
        } else {
            // Legacy v1/v2: single-layer shared weights → broadcast to all layers
            let mut single_wq = Vec::new();
            let mut single_wk = Vec::new();
            let mut single_wv = Vec::new();
            for _ in 0..h {
                single_wq.push(read_2d(&mut f)?);
                single_wk.push(read_2d(&mut f)?);
                single_wv.push(read_2d(&mut f)?);
            }
            for _ in 0..l {
                wq.push(single_wq.clone());
                wk.push(single_wk.clone());
                wv.push(single_wv.clone());
            }
        }
        let mut wo = Vec::new();
        let mut w1 = Vec::new();
        let mut w2 = Vec::new();
        let mut b1 = Vec::new();
        let mut b2 = Vec::new();
        let mut ln_gamma = Vec::new();
        let mut ln_beta = Vec::new();
        if version >= 3 {
            for _ in 0..l {
                wo.push(read_2d(&mut f)?);
            }
            for _ in 0..l {
                w1.push(read_2d(&mut f)?);
            }
            for _ in 0..l {
                w2.push(read_2d(&mut f)?);
            }
            for _ in 0..l {
                b1.push(read_1d(&mut f)?);
            }
            for _ in 0..l {
                b2.push(read_1d(&mut f)?);
            }
            for _ in 0..l {
                ln_gamma.push(read_1d(&mut f)?);
            }
            for _ in 0..l {
                ln_beta.push(read_1d(&mut f)?);
            }
        } else {
            let single_wo = read_2d(&mut f)?;
            let single_w1 = read_2d(&mut f)?;
            let single_w2 = read_2d(&mut f)?;
            let single_b1 = read_1d(&mut f)?;
            let single_b2 = read_1d(&mut f)?;
            let single_ln_gamma = read_1d(&mut f)?;
            let single_ln_beta = read_1d(&mut f)?;
            for _ in 0..l {
                wo.push(single_wo.clone());
                w1.push(single_w1.clone());
                w2.push(single_w2.clone());
                b1.push(single_b1.clone());
                b2.push(single_b2.clone());
                ln_gamma.push(single_ln_gamma.clone());
                ln_beta.push(single_ln_beta.clone());
            }
        }
        let w_output = read_2d(&mut f)?;
        Ok(Transformer {
            embedding_table,
            positional_table,
            wq, wk, wv, wo,
            heads, head_dim, layers,
            w1, w2, b1, b2, ln_gamma, ln_beta,
            w_output,
        })
    }
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
        dlogits: *mut c_void,
        seq_len: i32,
        vocab: i32,
    );

    fn cuda_cross_entropy_loss(
        logits: *const c_void,
        targets: *const f32,
        seq_len: i32,
        vocab: i32,
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

    fn cuda_sgd_momentum(
        g: *const f32,
        w: *mut c_void,
        v: *mut f32,
        lr: f32,
        mu: f32,
        n: i32,
    );
}

pub fn gpu_sgd_update(ws: &GpuWorkspace, lr: f32, layers: usize) {
    let mu = 0.9f32;
    let d = ws.d;
    let h = ws.h;
    let hd = ws.hd;
    let ff = ws.ff;
    let vocab = ws.vocab;
    unsafe {
        cuda_sgd_momentum(ws.g_dw_output.ptr, ws.d_w_output.ptr as *mut c_void, ws.v_dw_output.ptr, lr, mu, (d * vocab) as i32);
        cuda_sgd_momentum(ws.g_d_embedding.ptr, ws.d_embedding.ptr as *mut c_void, ws.v_d_embedding.ptr, lr, mu, (vocab * d) as i32);
        for layer in 0..layers {
            cuda_sgd_momentum(ws.g_wq_all[layer].ptr, ws.d_wq_all[layer].ptr as *mut c_void, ws.v_wq_all[layer].ptr, lr, mu, (h * d * hd) as i32);
            cuda_sgd_momentum(ws.g_wk_all[layer].ptr, ws.d_wk_all[layer].ptr as *mut c_void, ws.v_wk_all[layer].ptr, lr, mu, (h * d * hd) as i32);
            cuda_sgd_momentum(ws.g_wv_all[layer].ptr, ws.d_wv_all[layer].ptr as *mut c_void, ws.v_wv_all[layer].ptr, lr, mu, (h * d * hd) as i32);
            cuda_sgd_momentum(ws.g_dwo[layer].ptr, ws.d_wo[layer].ptr as *mut c_void, ws.v_dwo[layer].ptr, lr, mu, (d * d) as i32);
            cuda_sgd_momentum(ws.g_dw1[layer].ptr, ws.d_w1[layer].ptr as *mut c_void, ws.v_dw1[layer].ptr, lr, mu, (d * ff) as i32);
            cuda_sgd_momentum(ws.g_dw2[layer].ptr, ws.d_w2[layer].ptr as *mut c_void, ws.v_dw2[layer].ptr, lr, mu, (ff * d) as i32);
            cuda_sgd_momentum(ws.g_db1[layer].ptr, ws.d_b1[layer].ptr as *mut c_void, ws.v_db1[layer].ptr, lr, mu, ff as i32);
            cuda_sgd_momentum(ws.g_db2[layer].ptr, ws.d_b2[layer].ptr as *mut c_void, ws.v_db2[layer].ptr, lr, mu, d as i32);
            cuda_sgd_momentum(ws.g_dln_gamma[layer].ptr, ws.d_ln_gamma[layer].ptr as *mut c_void, ws.v_dln_gamma[layer].ptr, lr, mu, d as i32);
            cuda_sgd_momentum(ws.g_dln_beta[layer].ptr, ws.d_ln_beta[layer].ptr as *mut c_void, ws.v_dln_beta[layer].ptr, lr, mu, d as i32);
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
    pub fn to_bytes(&self) -> Result<Vec<u8>, std::io::Error> {
        use std::io::Write;
        let mut buf = Vec::new();
        buf.write_all(b"GORK")?;
        buf.write_all(&3u32.to_le_bytes())?;
        let d = self.wo[0].len() as i64;
        let vocab = self.w_output[0].len() as i64;
        let ctx = self.positional_table.len() as i64;
        let ff = self.w1[0][0].len() as i64;
        buf.write_all(&d.to_le_bytes())?;
        buf.write_all(&vocab.to_le_bytes())?;
        buf.write_all(&ctx.to_le_bytes())?;
        buf.write_all(&self.heads.to_le_bytes())?;
        buf.write_all(&ff.to_le_bytes())?;
        buf.write_all(&self.layers.to_le_bytes())?;
        let write_2d = |buf: &mut Vec<u8>, m: &[Vec<f32>]| {
            let rows = m.len() as u64;
            let cols = m[0].len() as u64;
            buf.write_all(&rows.to_le_bytes())?;
            buf.write_all(&cols.to_le_bytes())?;
            for row in m {
                for &v in row {
                    buf.write_all(&v.to_le_bytes())?;
                }
            }
            Ok::<_, std::io::Error>(())
        };
        let write_1d = |buf: &mut Vec<u8>, v: &[f32]| {
            let len = v.len() as u64;
            buf.write_all(&len.to_le_bytes())?;
            for &x in v {
                buf.write_all(&x.to_le_bytes())?;
            }
            Ok::<_, std::io::Error>(())
        };
        write_2d(&mut buf, &self.embedding_table)?;
        write_2d(&mut buf, &self.positional_table)?;
        for layer_idx in 0..self.layers as usize {
            for head_idx in 0..self.heads as usize {
                write_2d(&mut buf, &self.wq[layer_idx][head_idx])?;
                write_2d(&mut buf, &self.wk[layer_idx][head_idx])?;
                write_2d(&mut buf, &self.wv[layer_idx][head_idx])?;
            }
        }
        for mat in &self.wo {
            write_2d(&mut buf, mat)?;
        }
        for mat in &self.w1 {
            write_2d(&mut buf, mat)?;
        }
        for mat in &self.w2 {
            write_2d(&mut buf, mat)?;
        }
        for v in &self.b1 {
            write_1d(&mut buf, v)?;
        }
        for v in &self.b2 {
            write_1d(&mut buf, v)?;
        }
        for v in &self.ln_gamma {
            write_1d(&mut buf, v)?;
        }
        for v in &self.ln_beta {
            write_1d(&mut buf, v)?;
        }
        write_2d(&mut buf, &self.w_output)?;
        Ok(buf)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, std::io::Error> {
        use std::io::Read;
        let mut cursor = std::io::Cursor::new(bytes);
        let read_u32 = |c: &mut std::io::Cursor<&[u8]>| -> Result<u32, std::io::Error> {
            let mut buf = [0u8; 4];
            c.read_exact(&mut buf)?;
            Ok(u32::from_le_bytes(buf))
        };
        let read_i64 = |c: &mut std::io::Cursor<&[u8]>| -> Result<i64, std::io::Error> {
            let mut buf = [0u8; 8];
            c.read_exact(&mut buf)?;
            Ok(i64::from_le_bytes(buf))
        };
        let read_u64 = |c: &mut std::io::Cursor<&[u8]>| -> Result<u64, std::io::Error> {
            let mut buf = [0u8; 8];
            c.read_exact(&mut buf)?;
            Ok(u64::from_le_bytes(buf))
        };
        let read_2d = |c: &mut std::io::Cursor<&[u8]>| -> Result<Vec<Vec<f32>>, std::io::Error> {
            let rows = read_u64(c)? as usize;
            let cols = read_u64(c)? as usize;
            let mut buf = vec![0u8; rows * cols * 4];
            c.read_exact(&mut buf)?;
            let data: Vec<f32> = buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(data.chunks(cols).map(|c| c.to_vec()).collect())
        };
        let read_1d = |c: &mut std::io::Cursor<&[u8]>| -> Result<Vec<f32>, std::io::Error> {
            let len = read_u64(c)? as usize;
            let mut buf = vec![0u8; len * 4];
            c.read_exact(&mut buf)?;
            Ok(buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != b"GORK" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }
        let version = read_u32(&mut cursor)?;
        let dimensions = read_i64(&mut cursor)?;
        let _vocab = read_i64(&mut cursor)?;
        let _ctx_window = read_i64(&mut cursor)?;
        let heads = read_i64(&mut cursor)?;
        let _ff_multiplier = read_i64(&mut cursor)?;
        let layers = if version >= 2 { read_i64(&mut cursor)? } else { 6 };
        let head_dim = dimensions / heads;
        let embedding_table = read_2d(&mut cursor)?;
        let positional_table = read_2d(&mut cursor)?;
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let h = heads as usize;
        let l = layers as usize;
        if version >= 3 {
            for _ in 0..l {
                let mut layer_wq = Vec::new();
                let mut layer_wk = Vec::new();
                let mut layer_wv = Vec::new();
                for _ in 0..h {
                    layer_wq.push(read_2d(&mut cursor)?);
                    layer_wk.push(read_2d(&mut cursor)?);
                    layer_wv.push(read_2d(&mut cursor)?);
                }
                wq.push(layer_wq);
                wk.push(layer_wk);
                wv.push(layer_wv);
            }
        } else {
            let mut single_wq = Vec::new();
            let mut single_wk = Vec::new();
            let mut single_wv = Vec::new();
            for _ in 0..h {
                single_wq.push(read_2d(&mut cursor)?);
                single_wk.push(read_2d(&mut cursor)?);
                single_wv.push(read_2d(&mut cursor)?);
            }
            for _ in 0..l {
                wq.push(single_wq.clone());
                wk.push(single_wk.clone());
                wv.push(single_wv.clone());
            }
        }
        let mut wo = Vec::new();
        let mut w1 = Vec::new();
        let mut w2 = Vec::new();
        let mut b1 = Vec::new();
        let mut b2 = Vec::new();
        let mut ln_gamma = Vec::new();
        let mut ln_beta = Vec::new();
        if version >= 3 {
            for _ in 0..l {
                wo.push(read_2d(&mut cursor)?);
            }
            for _ in 0..l {
                w1.push(read_2d(&mut cursor)?);
            }
            for _ in 0..l {
                w2.push(read_2d(&mut cursor)?);
            }
            for _ in 0..l {
                b1.push(read_1d(&mut cursor)?);
            }
            for _ in 0..l {
                b2.push(read_1d(&mut cursor)?);
            }
            for _ in 0..l {
                ln_gamma.push(read_1d(&mut cursor)?);
            }
            for _ in 0..l {
                ln_beta.push(read_1d(&mut cursor)?);
            }
        } else {
            let single_wo = read_2d(&mut cursor)?;
            let single_w1 = read_2d(&mut cursor)?;
            let single_w2 = read_2d(&mut cursor)?;
            let single_b1 = read_1d(&mut cursor)?;
            let single_b2 = read_1d(&mut cursor)?;
            let single_ln_gamma = read_1d(&mut cursor)?;
            let single_ln_beta = read_1d(&mut cursor)?;
            for _ in 0..l {
                wo.push(single_wo.clone());
                w1.push(single_w1.clone());
                w2.push(single_w2.clone());
                b1.push(single_b1.clone());
                b2.push(single_b2.clone());
                ln_gamma.push(single_ln_gamma.clone());
                ln_beta.push(single_ln_beta.clone());
            }
        }
        let w_output = read_2d(&mut cursor)?;
        Ok(Transformer {
            embedding_table, positional_table,
            wq, wk, wv, wo,
            heads, head_dim, layers,
            w1, w2, b1, b2, ln_gamma, ln_beta,
            w_output,
        })
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
        gpu_sgd_update(&ws, 0.001, layers as usize);
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
