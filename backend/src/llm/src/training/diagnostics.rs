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

// How many of the most-frequent tokens to include in the embedding metrics. All
// C(N,2) pairwise cosines are averaged for anisotropy and the same N are projected
// for the PCA scatter, so this stays small: 32 tokens = 496 pairs, cheap to sync
// and reduce on the host once per report.
const ANISOTROPY_TOKENS: usize = 32;

// Activation-distribution histogram: number of bins and the half-range in standard
// deviations. The residual stream is z-scored (its scale is already the activation
// RMS chart's job), so the fixed ±4σ window over these bins gives a scale-free view
// of the distribution *shape* — is it Gaussian, heavy-tailed, saturating — as it
// evolves. Kept in sync with the frontend heatmap's axis labels.
const ACT_HIST_BINS: usize = 24;
const ACT_HIST_RANGE: f32 = 4.0;

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
    // Activation-distribution histogram of the deepest block's residual stream,
    // z-scored: `ACT_HIST_BINS` fractions (summing to ~1) over the ±ACT_HIST_RANGE
    // window. One per report → a distribution-over-time heatmap on the frontend.
    pub act_hist: Vec<f32>,
    // 2-D PCA projection of the most-frequent tokens' current embedding vectors
    // (one [x, y] per token, same order as `frequent_tokens`) — the latest report's
    // is scatter-plotted to show the embedding geometry.
    pub emb_pca: Vec<[f32; 2]>,
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
    let act_hist = activation_histogram(ws, layers, seq_len);

    // Sync the embedding table once, then derive both embedding metrics from it.
    let emb = ws.d_embedding.to_f32_vec();
    let (emb_cos_mean, emb_cos_p5, emb_cos_p95) = embedding_anisotropy(&emb, ws.d, freq_tokens);
    let emb_pca = embedding_pca(&emb, ws.d, freq_tokens);

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
        act_hist,
        emb_pca,
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

// Activation-distribution histogram of the deepest block's residual stream. The
// cached activations are z-scored (mean/std over the live rows) and binned into a
// fixed ±ACT_HIST_RANGE window, so the result describes the distribution *shape*
// over time independent of scale (the RMS chart carries the scale). Returns
// per-bin fractions summing to ~1.
fn activation_histogram(ws: &Workspace, layers: usize, seq_len: usize) -> Vec<f32> {
    let n = seq_len * ws.d;
    if layers == 0 || n == 0 {
        return vec![0.0; ACT_HIST_BINS];
    }
    let full = ws.layer_inputs[layers - 1].to_f32_vec();
    let vals = &full[..n.min(full.len())];
    if vals.is_empty() {
        return vec![0.0; ACT_HIST_BINS];
    }

    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    let std = var.sqrt().max(1e-6);

    let mut hist = vec![0u32; ACT_HIST_BINS];
    for &v in vals {
        let z = (v - mean) / std;
        // Map [-RANGE, RANGE] → [0, BINS); values outside land in the end bins.
        let t = ((z + ACT_HIST_RANGE) / (2.0 * ACT_HIST_RANGE)).clamp(0.0, 0.999_999);
        let bin = (t * ACT_HIST_BINS as f32) as usize;
        hist[bin.min(ACT_HIST_BINS - 1)] += 1;
    }
    let total = vals.len() as f32;
    hist.iter().map(|&c| c as f32 / total).collect()
}

// Embedding anisotropy: average the pairwise cosine similarity among the chosen
// frequent tokens. A high mean cosine means the embeddings are collapsing toward a
// common direction (poor separation); a healthy space keeps most pairs
// near-orthogonal. Returns (mean, p5, p95) of the cosines. `emb` is the already
// host-synced embedding table (`[vocab, d]` row-major).
fn embedding_anisotropy(emb: &[f32], d: usize, freq_tokens: &[u32]) -> (f32, f32, f32) {
    if freq_tokens.len() < 2 {
        return (0.0, 0.0, 0.0);
    }

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

// 2-D PCA of the chosen frequent tokens' embeddings, for the scatter plot. Because
// the token count N is small (≤ ANISOTROPY_TOKENS) and the model dim d can be
// large, this uses the N×N Gram matrix trick: for a centered data matrix X, the
// top eigenvectors u of G = XXᵀ (eigenvalues λ) give the sample coordinates on the
// principal axes directly as uᵢ·√λ — no d×d covariance needed. Returns one [x, y]
// per token in `freq_tokens` order. Sign of an axis is arbitrary (a mirror), which
// doesn't matter for a scatter.
fn embedding_pca(emb: &[f32], d: usize, freq_tokens: &[u32]) -> Vec<[f32; 2]> {
    let n = freq_tokens.len();
    if n < 2 {
        return vec![[0.0, 0.0]; n];
    }

    // Gather the rows and center them (subtract the per-dimension mean).
    let mut rows: Vec<Vec<f32>> = freq_tokens
        .iter()
        .map(|&id| emb[id as usize * d..id as usize * d + d].to_vec())
        .collect();
    let mut mean = vec![0.0f32; d];
    for r in &rows {
        for j in 0..d {
            mean[j] += r[j];
        }
    }
    for m in &mut mean {
        *m /= n as f32;
    }
    for r in &mut rows {
        for j in 0..d {
            r[j] -= mean[j];
        }
    }

    // Gram matrix G = X Xᵀ  (N×N, symmetric).
    let mut g = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for k in i..n {
            let dot: f32 = rows[i].iter().zip(&rows[k]).map(|(a, b)| a * b).sum();
            g[i][k] = dot;
            g[k][i] = dot;
        }
    }

    let (u1, l1) = top_eigenvector(&g, n, None);
    let (u2, l2) = top_eigenvector(&g, n, Some((&u1, l1)));
    let s1 = l1.max(0.0).sqrt();
    let s2 = l2.max(0.0).sqrt();
    (0..n).map(|i| [u1[i] * s1, u2[i] * s2]).collect()
}

// Power iteration for the top eigenvector/eigenvalue of a small symmetric matrix,
// with optional deflation of an already-found (eigenvector, eigenvalue) pair so the
// second call returns the second principal axis. Fixed iteration count — the Gram
// matrix is tiny (N ≤ 32) and this runs once per report, so convergence is cheap.
fn top_eigenvector(g: &[Vec<f32>], n: usize, deflate: Option<(&[f32], f32)>) -> (Vec<f32>, f32) {
    // Asymmetric init so a symmetric matrix doesn't leave the iterate on a saddle.
    let mut v: Vec<f32> = (0..n).map(|i| 1.0 + 0.01 * i as f32).collect();
    normalize(&mut v);

    let apply = |v: &[f32]| -> Vec<f32> {
        let mut w = vec![0.0f32; n];
        for i in 0..n {
            w[i] = (0..n).map(|k| g[i][k] * v[k]).sum();
        }
        if let Some((u, l)) = deflate {
            let uv: f32 = u.iter().zip(v).map(|(a, b)| a * b).sum();
            for i in 0..n {
                w[i] -= l * uv * u[i];
            }
        }
        w
    };

    for _ in 0..100 {
        let mut w = apply(&v);
        let norm = (w.iter().map(|x| x * x).sum::<f32>()).sqrt();
        if norm < 1e-12 {
            break;
        }
        for x in &mut w {
            *x /= norm;
        }
        v = w;
    }

    // Rayleigh quotient λ = vᵀ (G' v) with the same deflation applied.
    let gv = apply(&v);
    let lambda: f32 = v.iter().zip(&gv).map(|(a, b)| a * b).sum();
    (v, lambda)
}

// L2-normalize a vector in place (no-op on a zero vector).
fn normalize(v: &mut [f32]) {
    let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
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

        // Histogram: fixed bin count, non-negative fractions summing to ~1.
        assert_eq!(diag.act_hist.len(), ACT_HIST_BINS);
        let hist_sum: f32 = diag.act_hist.iter().sum();
        assert!((hist_sum - 1.0).abs() < 1e-3, "histogram should sum to 1: {hist_sum}");
        assert!(diag.act_hist.iter().all(|&f| f.is_finite() && f >= 0.0));

        // PCA: one finite 2-D point per frequent token.
        assert_eq!(diag.emb_pca.len(), freq.len());
        for [x, y] in &diag.emb_pca {
            assert!(x.is_finite() && y.is_finite(), "PCA coord not finite");
        }
        eprintln!(
            "✓ diagnostics finite & shaped (grad_global={:.4}, emb_cos_mean={:.4})",
            diag.grad_global, diag.emb_cos_mean
        );
    }
}
