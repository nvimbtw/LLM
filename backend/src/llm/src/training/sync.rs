//! Read trained weights back from the GPU `Workspace` into the CPU `Model` after
//! training. Reads the f32 masters (the source of truth), so the round-trip is
//! lossless. Operates on the `Workspace`, so it lives at `training/` level.

use crate::training::gpu_workspace::Workspace;
use crate::training::transformer::Model;

pub fn sync_weights_from_gpu(ws: &Workspace, t: &mut Model) {
    let d = ws.d;
    let h = ws.h;
    let hd = ws.hd;
    let ff = ws.ff;
    let vocab = ws.vocab;
    let layers = t.layers as usize;

    // Read back the f32 masters (the source of truth), not the f16 views — this
    // makes save lossless and drops the f16 re-quantize round-trip (ISSUE-0052).
    let w = &mut t.w_output;
    let flat = ws.m_w_output.to_vec();
    for i in 0..d {
        for j in 0..vocab {
            w[i][j] = flat[i * vocab + j];
        }
    }
    let emb = &mut t.embedding_table;
    let flat_emb = ws.m_embedding.to_vec();
    for i in 0..vocab {
        for j in 0..d {
            emb[i][j] = flat_emb[i * d + j];
        }
    }
    for layer in 0..layers {
        // Contiguous per-head QKV weights: [h * d * hd]
        let flat_q = ws.m_wq_all[layer].to_vec();
        let flat_k = ws.m_wk_all[layer].to_vec();
        let flat_v = ws.m_wv_all[layer].to_vec();
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
        let flat = ws.m_wo[layer].to_vec();
        for i in 0..d {
            for j in 0..d {
                t.wo[layer][i][j] = flat[i * d + j];
            }
        }
        let flat = ws.m_w1[layer].to_vec();
        for i in 0..d {
            for j in 0..ff {
                t.w1[layer][i][j] = flat[i * ff + j];
            }
        }
        let flat = ws.m_w2[layer].to_vec();
        for i in 0..ff {
            for j in 0..d {
                t.w2[layer][i][j] = flat[i * d + j];
            }
        }
        let flat = ws.m_b1[layer].to_vec();
        for j in 0..ff {
            t.b1[layer][j] = flat[j];
        }
        let flat = ws.m_b2[layer].to_vec();
        for j in 0..d {
            t.b2[layer][j] = flat[j];
        }
        let flat = ws.m_ln_gamma[layer].to_vec();
        for j in 0..d {
            t.ln_gamma[layer][j] = flat[j];
        }
        let flat = ws.m_ln_beta[layer].to_vec();
        for j in 0..d {
            t.ln_beta[layer][j] = flat[j];
        }
    }
}

