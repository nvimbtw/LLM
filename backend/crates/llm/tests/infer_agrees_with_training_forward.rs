//! The two forward passes in this crate must compute the same function.
//!
//! A model is trained under `transformer::forward` and served under
//! `infer_session::InferSession`. If they disagree, the model serves differently
//! than it trained and nothing about the loss curve or the benchmark score says
//! so — which is exactly the failure mode this repo has shipped twice (the
//! never-trained positional table, the never-trained K projection).
//!
//! The tolerance is a tolerance rather than exact equality because there is one
//! known difference between the two: the training feed-forward
//! (`cuda_feed_forward_forward`) adds the bias, rounds to f16 to save the
//! pre-activation for the backward, then applies GELU to the rounded value, while
//! the inference feed-forward (`cuda_feed_forward`) fuses bias+GELU and keeps the
//! intermediate in f32 — one f16 rounding step fewer, in inference's favour.
//!
//! In practice that difference does not show at these shapes: the measured
//! disagreement is below 1e-5 relative across every logit. The tolerance is sized
//! to catch *wiring* errors instead — a transposed weight, a skipped stage, the
//! wrong layer's parameters, a head-packing mismatch — which are wrong by whole
//! numbers rather than by an ULP. The argmax assertion is the sharper half: noise
//! does not move which token wins unless two logits were already tied.
//!
//! Run with `--test-threads=1`: the attention/feed-forward scratch is file-static
//! and the cuBLAS handle is shared (see `backend/CLAUDE.md`).

use llm::training::gpu_workspace::Workspace;
use llm::training::infer_session::InferSession;
use llm::training::transformer::Model;

// Small enough to be quick, big enough to be a real transformer: more than one
// layer (so a layer-index mixup shows), more than one head (so a head-packing
// mixup shows), and head_dim even (RoPE's adjacent pairing requires it).
const D: i64 = 64;
const VOCAB: i64 = 256;
const CTX: i64 = 32;
const HEADS: i64 = 4;
const FF_MULT: i64 = 4;
const LAYERS: i64 = 3;

struct Comparison {
    max_abs_diff: f32,
    max_rel_diff: f32,
    argmax_mismatches: usize,
    positions: usize,
}

// Runs both forwards over the same tokens with the same weights and compares
// every logit.
fn compare(tokens: &[u32]) -> Comparison {
    let t = Model::init(D, VOCAB, CTX, HEADS, FF_MULT, LAYERS);
    let vocab = VOCAB as usize;
    let s = tokens.len();

    // Training forward. `train_step` runs forward then backward; the backward
    // writes `d_dlogits`, never `d_logits`, so the forward's logits survive it.
    let mut ws = Workspace::new(&t, CTX as usize);
    ws.upload_weights(&t);
    let loss = pollster::block_on(t.train_step(&mut ws, tokens, LAYERS as usize));
    assert!(loss.is_finite(), "training forward produced a non-finite loss");
    let train_logits = ws.d_logits.to_f32_vec_n(s * vocab);
    // Free the workspace before building the session: both size the shared
    // file-static attention scratch, and holding 14 B/param of masters and
    // moments alongside is pointless here.
    drop(ws);

    // Inference forward, same weights.
    let mut session = InferSession::new(&t, CTX as usize);
    let infer_logits = session.logits_all(tokens);

    assert_eq!(
        infer_logits.len(),
        train_logits.len(),
        "the two forwards returned different logit counts"
    );

    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut argmax_mismatches = 0;

    for j in 0..s {
        let a = &train_logits[j * vocab..(j + 1) * vocab];
        let b = &infer_logits[j * vocab..(j + 1) * vocab];

        for (&x, &y) in a.iter().zip(b.iter()) {
            assert!(x.is_finite() && y.is_finite(), "non-finite logit at position {j}");
            let abs = (x - y).abs();
            max_abs_diff = max_abs_diff.max(abs);
            let scale = x.abs().max(y.abs()).max(1e-3);
            max_rel_diff = max_rel_diff.max(abs / scale);
        }

        let pick = |row: &[f32]| {
            let mut best = 0usize;
            for (i, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = i;
                }
            }
            best
        };
        if pick(a) != pick(b) {
            argmax_mismatches += 1;
        }
    }

    Comparison {
        max_abs_diff,
        max_rel_diff,
        argmax_mismatches,
        positions: s,
    }
}

#[test]
fn infer_session_matches_the_training_forward() {
    let tokens: Vec<u32> = (0..24u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    let c = compare(&tokens);

    eprintln!(
        "  {} positions: max abs diff {:.5}, max rel diff {:.5}, argmax mismatches {}",
        c.positions, c.max_abs_diff, c.max_rel_diff, c.argmax_mismatches
    );

    // Rounding noise, not a wiring error. A transposed weight or a skipped
    // sublayer lands orders of magnitude above this.
    assert!(
        c.max_rel_diff < 0.02,
        "the inference and training forwards disagree by {:.4} relative — that is \
         a wiring difference, not f16 rounding",
        c.max_rel_diff
    );
    assert_eq!(
        c.argmax_mismatches, 0,
        "the two forwards pick different top tokens at {} of {} positions",
        c.argmax_mismatches, c.positions
    );
}

// A full-context sequence: the scratch is allocated at max_s and this is the
// length that actually fills it. Guards the boundary rather than only the
// comfortable middle.
#[test]
fn infer_session_matches_at_full_context() {
    let tokens: Vec<u32> = (0..CTX as u32).map(|i| (i * 11 + 5) % VOCAB as u32).collect();
    let c = compare(&tokens);
    eprintln!(
        "  full context ({} positions): max rel diff {:.5}, argmax mismatches {}",
        c.positions, c.max_rel_diff, c.argmax_mismatches
    );
    assert!(c.max_rel_diff < 0.02, "disagreement at full context: {:.4}", c.max_rel_diff);
    assert_eq!(c.argmax_mismatches, 0);
}

// `logits_last` must return the last row `logits_all` returns. They take
// different paths — a 1-row GEMM off an offset pointer versus an s-row GEMM — and
// the offset arithmetic is the kind of thing that is silently wrong.
//
// Not bit-exact, and the reason is worth knowing: cuBLAS chooses its tiling and
// split-k strategy from the GEMM shape, so an m=1 call accumulates the same dot
// product in a different order than an m=s call. The observed disagreement is
// 3.81e-6 on a logit of 0.00607, which is exactly one ULP of the f16 the logits
// are stored in — the smallest difference representable, not an error in the
// offset. A wrong offset reads a different position's row entirely and lands
// nowhere near.
#[test]
fn logits_last_matches_the_last_row_of_logits_all() {
    let t = Model::init(D, VOCAB, CTX, HEADS, FF_MULT, LAYERS);
    let tokens: Vec<u32> = (0..20u32).map(|i| (i * 13 + 1) % VOCAB as u32).collect();
    let vocab = VOCAB as usize;

    // A fresh session for each call. Sharing one would leave the sequence fully
    // cached after `logits_all`, so `logits_last` would take the decode branch and
    // this would become a prefill-versus-decode comparison — a different property,
    // covered by `decode_equals_prefill_at_every_position`. What is under test here
    // is the row offset, so both sides must prefill.
    let all = InferSession::new(&t, CTX as usize).logits_all(&tokens);
    let last = InferSession::new(&t, CTX as usize).logits_last(&tokens);

    assert_eq!(last.len(), vocab);
    let expected = &all[(tokens.len() - 1) * vocab..];

    let mut max_rel = 0.0f32;
    for (&x, &y) in expected.iter().zip(last.iter()) {
        let scale = x.abs().max(y.abs()).max(1e-3);
        max_rel = max_rel.max((x - y).abs() / scale);
    }
    eprintln!("  logits_last vs logits_all last row: max rel diff {max_rel:.7}");
    assert!(
        max_rel < 1e-3,
        "logits_last disagrees with logits_all's last row by {max_rel:.5} relative — \
         that is a row-offset error, not GEMM accumulation order"
    );

    let pick = |row: &[f32]| {
        let mut best = 0usize;
        for (i, &v) in row.iter().enumerate() {
            if v > row[best] {
                best = i;
            }
        }
        best
    };
    assert_eq!(
        pick(expected),
        pick(&last),
        "logits_last and logits_all pick different top tokens"
    );
}

// ── The KV cache's core invariant ───────────────────────────────────────────
//
// A decode step against the cache must produce what a prefill of the same sequence
// produces. This is what catches a wrong absolute position in `rope_decode_kernel`,
// a mask assumption in `decode_softmax_kernel`, a cache pitch mismatch, or an
// off-by-one in which position gets written.
//
// **How this is measured, and why not per-logit relative error.** The two paths
// issue differently-shaped GEMMs over the same numbers — a decode scores one query
// against `s` keys where a prefill scores `s` against `s` — so cuBLAS accumulates
// in a different order and the f32 results differ in their last bits. Through six
// layers that lands at ~0.001 absolute on the logits.
//
// Per-logit *relative* error is the wrong lens on that: most of a vocabulary's
// logits sit near zero, where a 0.001 absolute difference reads as a 30% relative
// one while meaning nothing. What matters is whether the model would behave the
// same, so this measures:
//
//   * **max absolute difference against the logit range** — the noise floor as a
//     fraction of the signal. Measured at 0.04-0.06%, and flat in sequence length,
//     which is the signature of bounded rounding rather than a defect that
//     compounds.
//   * **total variation between the two next-token distributions** — the quantity
//     that decides whether sampling behaves identically. Measured at ~1e-4.
//   * **the top token**, whenever the top-2 gap is wider than the noise floor.
//
// Run with `--test-threads=1`.

struct Divergence {
    max_abs: f32,
    logit_range: f32,
    total_variation: f32,
    argmax_same: bool,
    top2_gap: f32,
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let total: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / total).collect()
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

fn diverge(expected: &[f32], got: &[f32]) -> Divergence {
    assert_eq!(expected.len(), got.len(), "logit counts differ");
    let max_abs = expected
        .iter()
        .zip(got)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let hi = expected.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lo = expected.iter().cloned().fold(f32::INFINITY, f32::min);

    let (pe, pg) = (softmax(expected), softmax(got));
    let total_variation: f32 =
        pe.iter().zip(&pg).map(|(&a, &b)| (a - b).abs()).sum::<f32>() * 0.5;

    let mut sorted = expected.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    Divergence {
        max_abs,
        logit_range: hi - lo,
        total_variation,
        argmax_same: argmax(expected) == argmax(got),
        top2_gap: sorted[0] - sorted[1],
    }
}

// Tolerances sit an order of magnitude above what is measured, so the test fails on
// a defect rather than on a driver or cuBLAS version picking different tiles.
fn assert_agrees(d: &Divergence, what: &str) {
    let fraction = d.max_abs / d.logit_range;
    assert!(
        fraction < 0.005,
        "{what}: logits differ by {:.5} absolute, {:.3}% of a {:.3} range — too much \
         for GEMM accumulation order",
        d.max_abs,
        fraction * 100.0,
        d.logit_range,
    );
    assert!(
        d.total_variation < 0.01,
        "{what}: next-token distributions differ by {:.6} total variation",
        d.total_variation,
    );
    // Only meaningful when the top two are further apart than the noise; a genuine
    // near-tie is allowed to resolve either way rather than making this flaky.
    if d.top2_gap > 4.0 * d.max_abs {
        assert!(
            d.argmax_same,
            "{what}: different top token despite a top-2 gap of {:.4} against a \
             {:.5} noise floor",
            d.top2_gap, d.max_abs,
        );
    }
}

// Walk one session forward a token at a time — every step after the first is a
// decode against the cache — and compare against a cold prefill of the same span.
#[test]
fn decode_equals_prefill_at_every_position() {
    let t = Model::init(D, VOCAB, CTX, HEADS, FF_MULT, LAYERS);
    let tokens: Vec<u32> = (0..20u32).map(|i| (i * 5 + 11) % VOCAB as u32).collect();

    let mut warm = InferSession::new(&t, CTX as usize);
    let mut worst_fraction = 0.0f32;
    let mut worst_tv = 0.0f32;

    for n in 1..=tokens.len() {
        let got = warm.logits_last(&tokens[..n]);
        let mut cold = InferSession::new(&t, CTX as usize);
        let expected = cold.logits_last(&tokens[..n]);

        let d = diverge(&expected, &got);
        assert_agrees(&d, &format!("length {n}"));
        worst_fraction = worst_fraction.max(d.max_abs / d.logit_range);
        worst_tv = worst_tv.max(d.total_variation);
    }
    eprintln!(
        "  worst over every prefix: {:.4}% of logit range, {:.6} total variation",
        worst_fraction * 100.0,
        worst_tv
    );
}

// The same at the shape that actually ships, where six layers give rounding the
// most room to compound. If the error were structural rather than noise, this is
// where it would show.
#[test]
#[ignore = "slower: the real serving shape, run explicitly"]
fn decode_equals_prefill_at_the_serving_shape() {
    let t = Model::init(256, 8192, 512, 4, 4, 6);
    let tokens: Vec<u32> = (0..120u32).map(|i| (i * 7 + 3) % 8192).collect();

    let mut warm = InferSession::new(&t, 512);
    for n in 1..=tokens.len() {
        let got = warm.logits_last(&tokens[..n]);
        if !matches!(n, 2 | 3 | 5 | 8 | 13 | 21 | 34 | 55 | 89 | 120) {
            continue;
        }
        let mut cold = InferSession::new(&t, 512);
        let expected = cold.logits_last(&tokens[..n]);
        let d = diverge(&expected, &got);
        eprintln!(
            "  n={n:4}: {:.4}% of logit range, TV {:.6}, top-2 gap {:.4}, argmax {}",
            d.max_abs / d.logit_range * 100.0,
            d.total_variation,
            d.top2_gap,
            if d.argmax_same { "same" } else { "differs" }
        );
        assert_agrees(&d, &format!("serving shape, length {n}"));
    }
}

// The prefix-reuse path: a sequence the cache already covers in full must still
// return that sequence's logits. This is the case N-candidates-from-one-prompt
// hits, and it takes the "re-decode the final position" branch, which nothing else
// exercises.
#[test]
fn a_fully_cached_sequence_still_returns_its_logits() {
    let t = Model::init(D, VOCAB, CTX, HEADS, FF_MULT, LAYERS);
    let prompt: Vec<u32> = (0..12u32).map(|i| (i * 9 + 4) % VOCAB as u32).collect();

    let mut cold = InferSession::new(&t, CTX as usize);
    let expected = cold.logits_last(&prompt);

    let mut warm = InferSession::new(&t, CTX as usize);
    // Prime it, then generate past the prompt so the cache holds strictly more than
    // the prompt, then ask for the prompt again.
    let _ = warm.logits_last(&prompt);
    let mut longer = prompt.clone();
    for _ in 0..4 {
        let logits = warm.logits_last(&longer);
        longer.push(argmax(&logits) as u32);
    }
    let got = warm.logits_last(&prompt);

    let d = diverge(&expected, &got);
    eprintln!(
        "  fully-cached replay: {:.4}% of logit range, TV {:.6}",
        d.max_abs / d.logit_range * 100.0,
        d.total_variation
    );
    assert_agrees(&d, "replaying a fully cached prompt");
}

// Beyond the context window the sequence is truncated to the last `ctx` tokens,
// and RoPE being relative is what makes that sound. Forwarding a too-long
// sequence must equal forwarding its tail.
//
// A fresh session per side, deliberately. Reusing one would have the second call
// find the tail already cached and take the decode branch, so the comparison would
// be prefill-versus-decode — which is a different property, tested above — and this
// one could no longer assert exact equality. Truncation is about windowing, so both
// sides should prefill.
#[test]
fn sequences_past_the_context_window_are_truncated_to_the_tail() {
    let t = Model::init(D, VOCAB, CTX, HEADS, FF_MULT, LAYERS);
    let ctx = CTX as usize;
    let long: Vec<u32> = (0..ctx as u32 + 9).map(|i| (i * 3 + 2) % VOCAB as u32).collect();
    let tail = &long[long.len() - ctx..];

    let from_long = InferSession::new(&t, ctx).logits_last(&long);
    let from_tail = InferSession::new(&t, ctx).logits_last(tail);

    for (i, (&x, &y)) in from_long.iter().zip(from_tail.iter()).enumerate() {
        assert_eq!(x, y, "truncation changed the logits at vocab index {i}");
    }
}

// A context window that has actually slid: generate past `ctx` on one session and
// check each step still matches a cold forward of the same windowed span.
//
// This is the case the cache cannot reuse. Once the window moves, position `i` holds
// a different token than the cached row written for it, so the prefix comparison
// fails and the sequence is re-prefilled. If that fallback were missing, decode
// would score a new query against history rotated to the wrong positions — and the
// result would still look like plausible text, which is why it needs a test.
#[test]
fn a_sliding_context_window_stays_correct() {
    let t = Model::init(D, VOCAB, CTX, HEADS, FF_MULT, LAYERS);
    let ctx = CTX as usize;
    let mut seq: Vec<u32> = (0..ctx as u32 - 3).map(|i| (i * 3 + 2) % VOCAB as u32).collect();

    let mut session = InferSession::new(&t, ctx);
    // Walk well past the window so it slides repeatedly.
    for step in 0..12 {
        let got = session.logits_last(&seq);
        let expected = InferSession::new(&t, ctx).logits_last(&seq);

        let d = diverge(&expected, &got);
        assert_agrees(&d, &format!("sliding window, step {step} (len {})", seq.len()));
        seq.push(argmax(&got) as u32);
    }
    eprintln!("  slid to length {} past a {ctx}-token window", seq.len());
}

// ── The same agreement, on real trained weights ─────────────────────────────
//
// Everything above runs on `Model::init`, i.e. Glorot noise. That exercises the
// wiring but not the numerics a trained model actually produces: real weights have
// grown apart in scale across layers, the residual stream is larger, and f16
// headroom is genuinely used. If the inference forward were going to disagree with
// the training forward anywhere, it would be here.
//
// This is also the closest available answer to "did the benchmark scores move?".
// The suites score a model through `logits_all`, so if these logits match what the
// training forward computes, the score is the score the model's own training
// objective implies. Point it at a model directory:
//
//   LLM_TEST_MODEL_DIR=data/artifacts/models/<uuid> \
//     cargo test -p llm --release --test infer_agrees_with_training_forward \
//     -- --ignored --nocapture --test-threads=1 on_real_weights
#[test]
#[ignore = "needs a trained model: set LLM_TEST_MODEL_DIR"]
fn the_two_forwards_agree_on_real_weights() {
    let dir = std::env::var("LLM_TEST_MODEL_DIR")
        .expect("set LLM_TEST_MODEL_DIR to a model directory");
    let t = Model::load(&dir).expect("could not load the model");
    // Through `vocab()`, not off `w_output`: from MODEL_VERSION 6 the output
    // projection is tied to the embedding table and `w_output` is empty, so
    // indexing it panicked on every model saved since.
    let vocab = t.vocab();
    let layers = t.layers as usize;
    // Well inside the context window: a Workspace at full ctx on a real model is
    // 14 B/param of masters and moments plus activation scratch, and the property
    // under test does not depend on length.
    let s = 64usize.min(t.ctx_window() - 1);
    let tokens: Vec<u32> = (0..s as u32).map(|i| (i * 37 + 11) % vocab as u32).collect();

    eprintln!(
        "  model: d={} layers={} heads={} vocab={} ctx={}",
        t.wo[0].len(),
        layers,
        t.heads,
        vocab,
        t.ctx_window()
    );

    let mut ws = Workspace::new(&t, t.ctx_window());
    ws.upload_weights(&t);
    let loss = pollster::block_on(t.train_step(&mut ws, &tokens, layers));
    assert!(loss.is_finite(), "training forward gave a non-finite loss");
    let train_logits = ws.d_logits.to_f32_vec_n(s * vocab);
    drop(ws);

    let mut session = InferSession::new(&t, t.ctx_window());
    let infer_logits = session.logits_all(&tokens);

    let mut worst_fraction = 0.0f32;
    let mut worst_tv = 0.0f32;
    let mut mismatches: Vec<(usize, f32, f32)> = Vec::new();

    for j in 0..s {
        let d = diverge(
            &train_logits[j * vocab..(j + 1) * vocab],
            &infer_logits[j * vocab..(j + 1) * vocab],
        );
        worst_fraction = worst_fraction.max(d.max_abs / d.logit_range);
        worst_tv = worst_tv.max(d.total_variation);
        if !d.argmax_same {
            mismatches.push((j, d.top2_gap, d.max_abs));
        }
        // Per position, not on an aggregate: this is the assertion that
        // distinguishes an argmax flip at a near-tie (inherent, and allowed) from
        // one where the top two were clearly separated (a defect).
        assert_agrees(&d, &format!("real weights, position {j}"));
    }

    eprintln!(
        "  worst of {s} positions: {:.4}% of logit range, TV {:.6}",
        worst_fraction * 100.0,
        worst_tv,
    );
    eprintln!("  argmax mismatches: {} of {s}", mismatches.len());
    for (j, gap, noise) in &mismatches {
        eprintln!(
            "    position {j}: top-2 gap {gap:.6} against a {noise:.6} noise floor \
             (gap is {:.2}x the noise)",
            gap / noise
        );
    }
}
