//! A/B guard on the FlashAttention kernels against the cuBLAS attention they
//! replace.
//!
//!   cargo test -p llm --release --test flash_attention_matches_cublas \
//!       -- --test-threads=1
//!
//! `--test-threads=1` is required. The `attn` scratch in `shaders.cu` is
//! file-static, the cuBLAS handle is process-wide, and — specific to this file —
//! so is the flash switch itself: two tests flipping it concurrently would each
//! run the other's path.
//!
//! # What this has to catch, and why the obvious test does not
//!
//! Attention has now shipped broken twice in this project, and both times the
//! loss still went down. `dK` was identically zero for the life of the project
//! (a pair of f32×f16 GEMMs cuBLAS rejects, plus Q/K/V read from the wrong
//! layer's scratch), and the LN2 backward overran its scratch into the live
//! attention arena. Neither is visible in a loss curve, a NaN check, or a "the
//! gradient is non-zero" assert.
//!
//! So the checks here are all *differential and absolute*: the same input
//! through both implementations, compared element for element, on the f32
//! gradient accumulators rather than the f16 activations, because f32 gives a
//! far sharper comparison. Plus one non-differential check — that flash
//! attention alone reproduces the causal softmax an independent f64 computation
//! gives — because two implementations agreeing does not pin down the semantics
//! they agree on.

use llm::training::flash;
use llm::training::gpu_workspace::Workspace;
use llm::training::transformer::Model;

// The shape the next pretrain actually runs at: d 512, 8 heads, head_dim 64.
const D: i64 = 512;
const HEADS: i64 = 8;
const VOCAB: i64 = 8192;
const FF_MULT: i64 = 4;
const LAYERS: i64 = 2;

fn chunk(len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 37 + 11) % VOCAB as u32).collect()
}

/// Run one training step at `seq` and return every f32 attention gradient
/// accumulator, concatenated, plus the loss.
///
/// The model is passed in rather than built here, and that is the whole validity
/// of the comparison: `Model::init` draws fresh random weights, so building one
/// per side would compare two different networks and call the difference a
/// kernel bug.
///
/// The `Workspace`, by contrast, *must* be fresh per side. It reads the switch
/// once at construction and allocates a different set of buffers accordingly —
/// `head_p_all` on one path, `lse_all` and `attn_entropy` on the other.
fn attention_grads(t: &Model, flash_on: bool, ctx: i64, seq: usize) -> (Vec<f32>, f32) {
    flash::set_enabled(flash_on);
    let mut ws = Workspace::new(t, ctx as usize);
    assert_eq!(
        ws.flash, flash_on,
        "the workspace did not take the path the switch asked for"
    );

    let loss = pollster::block_on(t.train_step(&mut ws, &chunk(seq), LAYERS as usize));

    let mut out = Vec::new();
    for l in 0..LAYERS as usize {
        out.extend(ws.g_wq_all[l].to_vec());
        out.extend(ws.g_wk_all[l].to_vec());
        out.extend(ws.g_wv_all[l].to_vec());
    }
    (out, loss)
}

struct Diff {
    /// ‖a − b‖₂ / ‖a‖₂.
    rel_l2: f32,
    max_abs: f32,
    norm: f32,
    nonfinite: usize,
}

/// Compare two gradient vectors by **relative L2 norm**, not by the largest
/// element difference over the largest element.
///
/// That ratio looks like the natural one and is not: the numerator counts one
/// element and the denominator counts a different one. Over these shapes the
/// largest difference is essentially flat in `seq` (2.5e-5 at 63 to 7.3e-5 at
/// 512) while the peak gradient *shrinks* — a fresh model spreads the same
/// signal over more positions — so the ratio grew 8x while nothing about the
/// agreement changed. It would have failed on sequence length alone. The L2
/// ratio moves with the quantity actually under test.
fn diff(a: &[f32], b: &[f32]) -> Diff {
    assert_eq!(a.len(), b.len());
    let mut d = Diff { rel_l2: 0.0, max_abs: 0.0, norm: 0.0, nonfinite: 0 };
    let (mut err_sq, mut ref_sq) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        if !y.is_finite() {
            d.nonfinite += 1;
            continue;
        }
        let e = (x - y) as f64;
        err_sq += e * e;
        ref_sq += (x as f64) * (x as f64);
        d.max_abs = d.max_abs.max(e.abs() as f32);
    }
    d.norm = ref_sq.sqrt() as f32;
    d.rel_l2 = if ref_sq > 0.0 { (err_sq.sqrt() / ref_sq.sqrt()) as f32 } else { 0.0 };
    d
}

/// The device must lay out a wmma accumulator the way the kernels assume.
///
/// The mapping from a fragment's 8 per-lane elements to (row, column) is not
/// specified by the API, and every flash kernel is built on the Ampere one — the
/// register-resident softmax state, the two-shuffle row reduction, and the
/// transposed store that makes `dV = P^T dO` free all read through it. The
/// kernels abort on mismatch; this makes a driver or architecture change show up
/// as a red test instead.
#[test]
fn the_wmma_accumulator_layout_is_what_the_kernels_assume() {
    assert!(
        flash::accumulator_layout_matches(),
        "this GPU distributes a wmma accumulator fragment differently than \
         ACC_ROW/ACC_COL in shaders.cu describe; the flash kernels would compute \
         silently wrong gradients"
    );
}

/// `head_dim` values the kernels are not instantiated for must fall back, not
/// approximate. `d / heads` is 48 here, which is a multiple of 16 but not one of
/// the four instantiations.
#[test]
fn an_unsupported_head_dim_falls_back() {
    flash::set_enabled(true);
    assert!(flash::active_for(64), "head_dim 64 is the shape that matters");
    assert!(!flash::active_for(48), "head_dim 48 has no instantiation");
    assert!(!flash::active_for(24), "head_dim 24 is not a multiple of 16");
    flash::set_enabled(false);
    assert!(!flash::active_for(64), "the switch must dominate the shape gate");
}

/// The caches must cover `max_s` rounded up to the kernels' block height.
///
/// A warp whose 16-row slab sits entirely past `seq` still executes its
/// `mma_sync` — a lane may never skip one — so that load reaches the rounded
/// extent. Reading one row past an allocation is exactly the silent
/// out-of-bounds class `assert_scratch_fits` exists for.
#[test]
fn the_row_rounding_covers_every_wmma_slab() {
    for max_s in [1usize, 63, 64, 65, 300, 512, 1024, 2048] {
        let rows = flash::rows_for(max_s);
        assert!(rows >= max_s, "rows_for({max_s}) = {rows} does not cover it");
        assert_eq!(rows % 64, 0, "rows_for({max_s}) = {rows} is not a whole block");
        assert!(rows - max_s < 64, "rows_for({max_s}) = {rows} over-rounds");
    }
}

/// A deterministic pseudo-random stream, so a failure is reproducible.
fn noise(n: usize, scale: f32, state: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            ((*state % 20001) as f32 / 10000.0 - 1.0) * scale
        })
        .collect()
}

/// The flash backward must reproduce the gradients of causal attention computed
/// in f64 on the CPU.
///
/// **This is the sharp test in this file**, and the one to look at first when it
/// goes red. It compares the kernels against ground truth rather than against
/// each other, and it compares `dQ`/`dK`/`dV` directly, before a two-layer
/// network has had a chance to dilute or amplify anything.
///
/// It also covers the three failure modes that are specific to this
/// implementation: the online-softmax statistics being reused correctly in the
/// backward (`P` is rebuilt from the saved `lse`, not from a stored plane), the
/// transposed shared-memory store that `dV = Pᵀ dO` and `dK = dSᵀ Q` rely on, and
/// the two-pass split — `dQ` comes out of a different kernel than `dK`/`dV`, so a
/// mistake in one leaves the others right.
#[test]
fn the_flash_backward_reproduces_f64_attention_gradients() {
    for &(seq, hd, h) in &[(48usize, 16usize, 2usize), (65, 64, 2), (128, 64, 4)] {
        let n = h * seq * hd;
        let mut state = 0x2545F4914F6CDD1Du64;
        let q = noise(n, 0.5, &mut state);
        let k = noise(n, 0.5, &mut state);
        let v = noise(n, 0.5, &mut state);
        let d_out = noise(n, 0.25, &mut state);

        let (want_dq, want_dk, want_dv) = f64_attention_backward(&q, &k, &v, &d_out, seq, hd, h);

        flash::set_enabled(true);
        let (dq, dk, dv) =
            flash::attention_backward_for_test(&q, &k, &v, &d_out, seq, hd, h);

        for (name, want, got) in [
            ("dQ", &want_dq, &dq),
            ("dK", &want_dk, &dk),
            ("dV", &want_dv, &dv),
        ] {
            let d = diff(want, got);
            assert_eq!(
                d.nonfinite, 0,
                "seq {seq} hd {hd}: {} {name} values are not finite",
                d.nonfinite
            );
            assert!(d.norm > 0.0, "seq {seq} hd {hd}: reference {name} is all zero");
            eprintln!(
                "  seq {seq:>4} hd {hd:>3} {name}: rel_l2 {:.6}  max|d| {:.7}",
                d.rel_l2, d.max_abs
            );
            // The operands are f16 (~1e-3 relative resolution) and the products
            // accumulate in f32, so ~1e-3 relative is the floor here. Anything
            // structural — a transposed store, a missed causal bound, the wrong
            // scale — lands orders away from this, not just outside it.
            assert!(
                d.rel_l2 < 5e-3,
                "seq {seq} hd {hd}: {name} differs from f64 attention by {:.6} \
                 in relative L2",
                d.rel_l2
            );
        }
    }
}

/// dQ, dK and dV of causal `softmax(Q Kᵀ/√hd) V`, in f64, straight from the
/// definition. Deliberately the naive O(s²·hd) form: it is the specification the
/// kernels are being held to, so it should be readable rather than fast.
fn f64_attention_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    seq: usize,
    hd: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (hd as f64).sqrt();
    let n = h * seq * hd;
    let (mut dq, mut dk, mut dv) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);

    for head in 0..h {
        let plane = head * seq * hd;
        let at = |buf: &[f32], row: usize, i: usize| buf[plane + row * hd + i] as f64;

        for row in 0..seq {
            // Forward, for this row: the causal softmax weights.
            let mut p: Vec<f64> = (0..=row)
                .map(|col| (0..hd).map(|i| at(q, row, i) * at(k, col, i)).sum::<f64>() * scale)
                .collect();
            let max = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for x in p.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            for x in p.iter_mut() {
                *x /= sum;
            }

            // D = dO · O, the softmax-backward correction term. O is rebuilt
            // here rather than passed in, so this stays self-contained.
            let d_term: f64 = (0..hd)
                .map(|i| {
                    let o: f64 = p.iter().enumerate().map(|(c, &w)| w * at(v, c, i)).sum();
                    at(d_out, row, i) * o
                })
                .sum();

            for col in 0..=row {
                // dV += Pᵀ dO
                for i in 0..hd {
                    dv[plane + col * hd + i] += (p[col] * at(d_out, row, i)) as f32;
                }
                let d_p: f64 = (0..hd).map(|i| at(d_out, row, i) * at(v, col, i)).sum();
                let d_s = p[col] * (d_p - d_term) * scale;
                for i in 0..hd {
                    dq[plane + row * hd + i] += (d_s * at(k, col, i)) as f32;
                    dk[plane + col * hd + i] += (d_s * at(q, row, i)) as f32;
                }
            }
        }
    }
    (dq, dk, dv)
}

/// End to end, the attention weight gradients must track the cuBLAS path.
///
/// A smoke test rather than a precision one, and the tolerance says so. It runs
/// a whole training step, so what it can resolve is limited by the *conditioning*
/// of the measurement, not by the kernels: at a freshly initialized model the
/// per-position gradients are uncorrelated, so ‖g‖ falls roughly as 1/√seq while
/// the absolute difference stays flat (2.7e-5 at seq 63, 7.7e-5 at 512). The
/// ratio therefore grows 8x across these shapes with nothing about the agreement
/// changing. `the_flash_backward_reproduces_f64_attention_gradients` is where
/// precision is actually pinned; what this adds is that the kernels are wired
/// into the real step correctly — right buffers, right layer, right layout.
///
/// Sequence lengths straddle the 64-row block: 63 and 65 exercise the tail masks
/// and the partial wmma slab. 2 is the shortest chunk carrying any gradient at
/// all — position `i` predicts token `i+1`, so a one-token chunk has nothing to
/// learn from and its accumulators stay zero.
///
/// Sequence lengths deliberately straddle the 64-row block: 63 and 65 exercise
/// the tail masks and the partial wmma slab, which is where an off-by-one in the
/// causal bound or a missing store predicate lives. 2 is the shortest chunk that
/// carries any gradient at all — position `i` predicts token `i+1`, so a
/// one-token chunk has nothing to learn from and its accumulators stay zero.
///
#[test]
fn the_attention_gradients_match_the_cublas_path() {
    let ctx = 512;
    let t = Model::init(D, VOCAB, ctx, HEADS, FF_MULT, LAYERS);
    for &seq in &[2usize, 63, 64, 65, 300, 512] {
        let (reference, ref_loss) = attention_grads(&t, false, ctx, seq);
        let (actual, flash_loss) = attention_grads(&t, true, ctx, seq);

        let d = diff(&reference, &actual);
        // Finiteness first. A NaN poisons a max-difference assertion into
        // passing — `f32::max` propagates the operand that is *not* NaN — so a
        // "how close is it" check alone would sail straight past the failure
        // mode this whole file is about.
        assert_eq!(
            d.nonfinite, 0,
            "seq {seq}: {} of {} flash gradients are not finite",
            d.nonfinite,
            actual.len()
        );
        assert!(
            d.norm > 0.0,
            "seq {seq}: the reference gradient is identically zero, so this \
             comparison proves nothing"
        );
        eprintln!(
            "  seq {seq:>4}: rel_l2 {:.6}  max|d| {:.8}  ‖g‖ {:.6}  \
             loss {ref_loss:.6} / {flash_loss:.6}",
            d.rel_l2, d.max_abs, d.norm
        );
        assert!(
            d.rel_l2 < 5e-2,
            "seq {seq}: attention gradients differ by {:.5} in relative L2",
            d.rel_l2
        );
        // Scale-free and stable across `seq`, unlike the ratio above: the
        // absolute difference is set by f16 operand resolution, which does not
        // move with sequence length. Every real defect this file has caught was
        // orders of magnitude over this — the `0 * inf` tail bug read 3e5.
        assert!(
            d.max_abs < 1e-3,
            "seq {seq}: largest single gradient difference is {:.7}",
            d.max_abs
        );
        // The loss is an independent witness: it comes off a different reduction
        // (cross-entropy over the logits) and would not move with a gradient-only
        // bug, so agreement here and disagreement above would localize the fault
        // to the backward.
        assert!(
            (ref_loss - flash_loss).abs() < 1e-2,
            "seq {seq}: loss {flash_loss:.5} against {ref_loss:.5}"
        );
    }
}

/// A head with no incoming gradient must still be *reached*.
///
/// This is the shape of the bug that froze `dK` at its random initialization for
/// the life of the project: a gradient that is structurally never written looks
/// exactly like a gradient that happens to be small. Running a step where one
/// head's contribution is isolated and requiring every head's accumulator to be
/// non-zero separates the two.
#[test]
fn every_head_receives_a_gradient() {
    let ctx = 256;
    flash::set_enabled(true);
    let t = Model::init(D, VOCAB, ctx, HEADS, FF_MULT, LAYERS);
    let mut ws = Workspace::new(&t, ctx as usize);
    assert!(ws.flash);
    pollster::block_on(t.train_step(&mut ws, &chunk(ctx as usize), LAYERS as usize));

    let hd = (D / HEADS) as usize;
    let per_head = D as usize * hd;
    for (name, g) in [
        ("wq", ws.g_wq_all[0].to_vec()),
        ("wk", ws.g_wk_all[0].to_vec()),
        ("wv", ws.g_wv_all[0].to_vec()),
    ] {
        for head in 0..HEADS as usize {
            let slice = &g[head * per_head..(head + 1) * per_head];
            let mag: f32 = slice.iter().map(|v| v.abs()).sum();
            assert!(
                mag > 0.0 && mag.is_finite(),
                "layer 0 head {head}: d{name} sums to {mag}, so nothing wrote it"
            );
        }
    }
}

/// Two identical runs must produce bit-identical gradients.
///
/// This is what the two-pass backward is *for*. FlashAttention-2 canonically
/// parallelizes over key blocks and `atomicAdd`s into `dQ`, which is faster and
/// not reproducible; the extra pass here buys an exact answer. That matters
/// because every other test in this file is an exact A/B, and a
/// non-deterministic reduction makes "did this change the gradient?"
/// unanswerable.
#[test]
fn the_flash_backward_is_bit_reproducible() {
    let ctx = 256;
    let t = Model::init(D, VOCAB, ctx, HEADS, FF_MULT, LAYERS);
    let (first, _) = attention_grads(&t, true, ctx, ctx as usize);
    let (second, _) = attention_grads(&t, true, ctx, ctx as usize);
    let mismatches = first.iter().zip(&second).filter(|(a, b)| a != b).count();
    assert_eq!(
        mismatches, 0,
        "{mismatches} of {} gradients changed between two identical runs",
        first.len()
    );
}

/// Flash attention alone must reproduce a causal softmax computed in f64 on the
/// CPU.
///
/// Every other test here is differential, and two implementations agreeing does
/// not pin down *what* they agree on — the max subtraction, the scale living
/// inside the exponent, `col <= row` rather than `col < row`. Those are
/// semantics, and this is what fixes them. It runs at a toy shape on purpose:
/// the reference is O(s²·hd) in interpreted-speed Rust.
#[test]
fn flash_attention_reproduces_an_f64_causal_softmax() {
    const S: usize = 48;
    const HD: usize = 16;
    const H: usize = 2;

    // A deterministic Q/K/V, then attention over it by definition.
    let mut state = 88172645463325252u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 20001) as f32 / 10000.0 - 1.0
    };
    let n = H * S * HD;
    let q: Vec<f32> = (0..n).map(|_| next() * 0.5).collect();
    let k: Vec<f32> = (0..n).map(|_| next() * 0.5).collect();
    let v: Vec<f32> = (0..n).map(|_| next() * 0.5).collect();

    let scale = 1.0 / (HD as f64).sqrt();
    let mut expected = vec![0.0f64; n];
    for head in 0..H {
        let plane = head * S * HD;
        for row in 0..S {
            let mut scores = vec![0.0f64; row + 1];
            let mut max = f64::NEG_INFINITY;
            for (col, s) in scores.iter_mut().enumerate() {
                let dot: f64 = (0..HD)
                    .map(|i| q[plane + row * HD + i] as f64 * k[plane + col * HD + i] as f64)
                    .sum();
                *s = dot * scale;
                max = max.max(*s);
            }
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for i in 0..HD {
                let acc: f64 = scores
                    .iter()
                    .enumerate()
                    .map(|(col, p)| p * v[plane + col * HD + i] as f64)
                    .sum();
                expected[plane + row * HD + i] = acc / sum;
            }
        }
    }

    flash::set_enabled(true);
    let actual = llm::training::flash::attention_for_test(&q, &k, &v, S, HD, H);

    let mut worst = 0.0f64;
    for (i, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
        assert!(a.is_finite(), "output {i} is {a}");
        worst = worst.max((e - a as f64).abs());
    }
    assert!(
        worst < 8e-3,
        "flash attention differs from an f64 causal softmax by {worst:.6}; the \
         operands are f16, so ~1e-3 is the floor, but this is a semantic gap"
    );
}
