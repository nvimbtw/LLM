//! Read-only training telemetry sampled off the hot path (once per progress
//! report, ~1% of steps) to feed the training dashboard. Everything here only
//! *reads* the workspace's forward caches, master weights, and gradient
//! accumulators — it never mutates training state, so it can never perturb the
//! numerics of the run it observes.
//!
//! The gradient-norm and update:weight ratios live in `optimizer.rs` (that module
//! owns the per-parameter buffer layout and the Adam constants); this module adds
//! the activation, attention-entropy, and embedding metrics and assembles the
//! whole `Diagnostics` snapshot the training loop emits.

use std::collections::HashMap;
use std::ffi::c_void;

use crate::training::device_buffer::Buffer;
use crate::training::gpu_workspace::Workspace;
use crate::training::optimizer::gpu_update_ratios;

unsafe extern "C" {
    // Σ(x²) over an f16 buffer (block reduction, host-side result).
    fn cuda_sumsq_f16(v: *const c_void, n: i32) -> f32;
    // Per-head mean attention entropy, normalized to [0,1] and summed over query
    // rows into `out_dev` (h floats). See the kernel comment in shaders.cu.
    fn cuda_attention_entropy(p: *const c_void, seq: i32, h: i32, out_dev: *mut f32);
}

// How many of the most-frequent tokens to include in the embedding-anisotropy
// metric. All C(N,2) pairwise cosines are averaged, so this stays small: 32
// tokens = 496 pairs, cheap to sync and reduce on the host once per report.
const ANISOTROPY_TOKENS: usize = 32;

// One progress report's worth of training diagnostics. Every vector is indexed by
// transformer layer (0 = first block); `attn_entropy[layer]` holds one value per
// attention head. Assembled by `compute` and threaded through the progress frame.
pub struct Diagnostics {
    // Gradient L2 norms.
    pub grad_global: f32,
    pub grad_layers: Vec<f32>,
    pub grad_embedding: f32,
    pub grad_unembed: f32,
    // Per-layer AdamW update:weight ratio ‖Δw‖/‖w‖.
    pub update_ratio_layers: Vec<f32>,
    // RMS of the residual-stream activation entering each block.
    pub act_rms_layers: Vec<f32>,
    // Per-head attention entropy (normalized 0..1), grouped by layer.
    pub attn_entropy: Vec<Vec<f32>>,
    // Embedding anisotropy: mean and 5th/95th-percentile of the pairwise cosine
    // similarity among the most-frequent tokens (high mean ⇒ collapsing/isotropic).
    pub emb_cos_mean: f32,
    pub emb_cos_p5: f32,
    pub emb_cos_p95: f32,
}

// The `ANISOTROPY_TOKENS` most-frequent token ids in `corpus`, high→low. Computed
// once per run (the corpus is fixed) and reused for every report so the embedding
// metric tracks the same tokens over training. Ties broken by id for determinism.
pub fn frequent_tokens(corpus: &[u32]) -> Vec<u32> {
    let mut counts: HashMap<u32, u64> = HashMap::new();
    for &t in corpus {
        *counts.entry(t).or_insert(0) += 1;
    }
    let mut ranked: Vec<(u32, u64)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(ANISOTROPY_TOKENS);
    ranked.into_iter().map(|(id, _)| id).collect()
}

// Assemble a full diagnostics snapshot. Reads the caches left by the most recent
// `train_step` (so `seq_len` must be that step's chunk length) plus the gradient
// and moment buffers. `lr`/`step` describe the optimizer step just taken, for the
// update:weight ratio. `freq_tokens` comes from `frequent_tokens`.
//
// Call ordering matters (see the training loop): `grad_*` reads the accumulated
// gradient, so it must run *before* the step zeroes it; the update ratio reads the
// post-step moments. `compute` itself only reads, so the loop passes the grad
// norms in and computes the rest here after the step.
pub fn compute(
    ws: &Workspace,
    grad: crate::training::optimizer::GradNorms,
    layers: usize,
    seq_len: usize,
    lr: f32,
    step: i32,
    freq_tokens: &[u32],
) -> Diagnostics {
    let update_ratio_layers = gpu_update_ratios(ws, layers, lr, step);
    let act_rms_layers = activation_rms(ws, layers, seq_len);
    let attn_entropy = attention_entropy(ws, layers, seq_len);
    let (emb_cos_mean, emb_cos_p5, emb_cos_p95) = embedding_anisotropy(ws, freq_tokens);

    Diagnostics {
        grad_global: grad.global,
        grad_layers: grad.layers,
        grad_embedding: grad.embedding,
        grad_unembed: grad.unembed,
        update_ratio_layers,
        act_rms_layers,
        attn_entropy,
        emb_cos_mean,
        emb_cos_p5,
        emb_cos_p95,
    }
}

// RMS of the residual stream entering each block (the cached layer input), a read
// on activation growth/shrinkage with depth. `n = seq_len * d` because only the
// first `seq_len` rows of the max-sized cache are valid for this step.
fn activation_rms(ws: &Workspace, layers: usize, seq_len: usize) -> Vec<f32> {
    let n = seq_len * ws.d;
    (0..layers)
        .map(|l| {
            let sumsq = unsafe { cuda_sumsq_f16(ws.layer_inputs[l].ptr as *const c_void, n as i32) };
            if n > 0 {
                (sumsq / n as f32).sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

// Per-head attention entropy for every layer, from the cached causal softmax
// (`head_p_all`). The kernel sums the normalized per-row entropies; dividing by
// the query-row count (seq-1, since row 0 is skipped) gives each head's mean.
fn attention_entropy(ws: &Workspace, layers: usize, seq_len: usize) -> Vec<Vec<f32>> {
    let h = ws.h;
    let out = Buffer::alloc(h);
    let rows = seq_len.max(2) - 1;
    (0..layers)
        .map(|l| {
            unsafe {
                cuda_attention_entropy(
                    ws.head_p_all[l].ptr as *const c_void,
                    seq_len as i32,
                    h as i32,
                    out.ptr,
                );
            }
            out.to_vec().iter().map(|&e| e / rows as f32).collect()
        })
        .collect()
}

// Embedding anisotropy: sync the embedding table once, then average the pairwise
// cosine similarity among the chosen frequent tokens. A high mean cosine means the
// embeddings are collapsing toward a common direction (poor separation); a healthy
// space keeps most pairs near-orthogonal. Returns (mean, p5, p95) of the cosines.
fn embedding_anisotropy(ws: &Workspace, freq_tokens: &[u32]) -> (f32, f32, f32) {
    let d = ws.d;
    if freq_tokens.len() < 2 {
        return (0.0, 0.0, 0.0);
    }
    let emb = ws.d_embedding.to_f32_vec();

    // Pre-normalize each chosen token's row so a cosine is a plain dot product.
    let units: Vec<Vec<f32>> = freq_tokens
        .iter()
        .map(|&id| {
            let row = &emb[id as usize * d..id as usize * d + d];
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                row.iter().map(|x| x / norm).collect()
            } else {
                vec![0.0; d]
            }
        })
        .collect();

    let mut cosines: Vec<f32> = Vec::with_capacity(units.len() * (units.len() - 1) / 2);
    for i in 0..units.len() {
        for j in (i + 1)..units.len() {
            let dot: f32 = units[i].iter().zip(&units[j]).map(|(a, b)| a * b).sum();
            cosines.push(dot);
        }
    }
    if cosines.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mean = cosines.iter().sum::<f32>() / cosines.len() as f32;
    cosines.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |q: f32| {
        let idx = ((cosines.len() - 1) as f32 * q).round() as usize;
        cosines[idx]
    };
    (mean, pct(0.05), pct(0.95))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::optimizer::{gpu_adam_update, gpu_grad_norms};
    use crate::training::transformer::Model;

    // Diagnostics must be well-shaped and finite over a real (tiny) step, and
    // every metric must land in its natural range: entropy in [0,1], cosine in
    // [-1,1], norms/RMS/ratios non-negative. This is the guard that a kernel index
    // or a normalizer isn't off — the values feed a live dashboard, so a NaN or an
    // out-of-range number would be visible garbage. (Read-only-ness is structural:
    // `compute` never writes device buffers, so it can't perturb the run.)
    #[test]
    fn test_diagnostics_finite_and_shaped() {
        let (vocab, dim, ctx, heads, ff_mult, layers) = (20, 16, 8, 2, 2, 2);
        let t = Model::init(dim, vocab, ctx, heads, ff_mult, layers);
        let mut ws = Workspace::new(&t, ctx as usize);
        let chunk: Vec<u32> = vec![3, 7, 1, 14, 2, 9, 5, 11];

        // Populate gradient accumulators, forward caches, and (after the step) the
        // Adam moments — the same order the training loop samples in.
        pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));
        let grad = gpu_grad_norms(&ws, layers as usize);
        gpu_adam_update(&ws, 0.01, 1, 1, layers as usize);

        let freq = frequent_tokens(&chunk);
        let diag = compute(&ws, grad, layers as usize, chunk.len(), 0.01, 1, &freq);

        assert_eq!(diag.grad_layers.len(), layers as usize);
        assert_eq!(diag.update_ratio_layers.len(), layers as usize);
        assert_eq!(diag.act_rms_layers.len(), layers as usize);
        assert_eq!(diag.attn_entropy.len(), layers as usize);

        assert!(diag.grad_global.is_finite() && diag.grad_global >= 0.0);
        assert!(diag.grad_embedding.is_finite() && diag.grad_embedding >= 0.0);
        assert!(diag.grad_unembed.is_finite() && diag.grad_unembed >= 0.0);
        for &g in &diag.grad_layers {
            assert!(g.is_finite() && g >= 0.0, "grad norm out of range: {g}");
        }
        for &u in &diag.update_ratio_layers {
            assert!(u.is_finite() && u >= 0.0, "update ratio out of range: {u}");
        }
        for &a in &diag.act_rms_layers {
            assert!(a.is_finite() && a >= 0.0, "activation rms out of range: {a}");
        }
        for row in &diag.attn_entropy {
            assert_eq!(row.len(), heads as usize);
            for &e in row {
                // Normalized entropy: 1 = uniform attention, 0 = a single-key spike.
                // A hair over 1.0 is allowed for f16/rounding slack.
                assert!(e.is_finite() && (0.0..=1.05).contains(&e), "entropy out of range: {e}");
            }
        }
        assert!(diag.emb_cos_mean.is_finite() && (-1.01..=1.01).contains(&diag.emb_cos_mean));
        assert!(diag.emb_cos_p5 <= diag.emb_cos_p95 + 1e-3, "p5 must not exceed p95");
        eprintln!(
            "✓ diagnostics finite & shaped (grad_global={:.4}, emb_cos_mean={:.4})",
            diag.grad_global, diag.emb_cos_mean
        );
    }
}
