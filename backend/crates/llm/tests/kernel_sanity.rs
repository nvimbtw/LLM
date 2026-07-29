//! Fast correctness gate for the CUDA kernels, at realistic shapes.
//!
//! The other GPU tests either use toy dimensions (d=16, vocab=20 — where a
//! 256-thread block is mostly idle and never exercises a full reduction) or run
//! a whole training job (minutes). This sits in between deliberately: shapes big
//! enough that the block-per-row kernels do real reductions over real rows, but
//! small enough to finish in seconds, so it can gate every kernel change.
//!
//!   cargo test -p llm --release --test kernel_sanity -- --test-threads=1
//!
//! `--test-threads=1` is required, not cosmetic. `Workspace::drop` calls
//! `cublas_shutdown`, which tears down the process-wide cuBLAS handle *and* the
//! shared attention / feed-forward / clip scratch buffers. Two live Workspaces
//! in parallel tests means one dropping pulls the scratch out from under the
//! other, and the failure shows up as a wrong loss rather than a crash.
//!
//! What it actually pins down is the softmax / cross-entropy / layer-norm path,
//! which is where a botched shared-memory reduction hides: a wrong reduction
//! usually still produces finite, plausible-looking numbers, and only an
//! absolute check catches it.

use llm::training::gpu_workspace::Workspace;
use llm::training::optimizer::{gpu_adam_update, gpu_clip_gradients};
use llm::training::transformer::Model;

// d divisible by heads, and d == 256 so layer-norm's row length matches the
// block width — the case the toy tests never cover.
const D: i64 = 256;
const VOCAB: i64 = 8192;
const CTX: i64 = 256;
const HEADS: i64 = 4;
const FF_MULT: i64 = 4;
const LAYERS: i64 = 2;

fn model_and_ws() -> (Model, Workspace) {
    let t = Model::init(D, VOCAB, CTX, HEADS, FF_MULT, LAYERS);
    let ws = Workspace::new(&t, CTX as usize);
    (t, ws)
}

fn chunk() -> Vec<u32> {
    (0..CTX as u32).map(|i| (i * 37 + 11) % VOCAB as u32).collect()
}

/// A freshly initialized model's cross-entropy loss must be ~ln(vocab).
///
/// This is the strongest cheap check there is on the logits → softmax →
/// cross-entropy path. At init the output distribution is near-uniform over the
/// vocabulary, so -log p(target) ≈ -log(1/vocab) = ln(vocab) ≈ 9.01 here. A
/// broken row max, a reduction that sums the wrong elements, or a mis-strided
/// logits read all move this number well outside the tolerance while still
/// producing a finite, innocuous-looking loss.
#[test]
fn initial_loss_matches_uniform_prior() {
    let (t, mut ws) = model_and_ws();
    let loss = pollster::block_on(t.train_step(&mut ws, &chunk(), LAYERS as usize));
    let expected = (VOCAB as f32).ln();

    assert!(loss.is_finite(), "initial loss was not finite: {loss}");
    assert!(
        (loss - expected).abs() < 0.5,
        "initial loss {loss:.4} is not near ln(vocab) = {expected:.4}; \
         the softmax / cross-entropy path is wrong"
    );
}

/// The full forward+backward+Adam loop must actually reduce the loss.
///
/// Covers what the absolute check above cannot: the backward kernels. A broken
/// layer-norm backward, softmax backward or cross-entropy backward leaves the
/// forward perfectly correct and simply fails to learn (or diverges), which is
/// exactly the failure mode that is invisible without training a few steps.
#[test]
fn loss_decreases_over_steps() {
    let (t, mut ws) = model_and_ws();
    let chunk = chunk();

    let first = pollster::block_on(t.train_step(&mut ws, &chunk, LAYERS as usize));
    ws.zero_accumulators();

    let mut last = first;
    for step in 1..=30 {
        last = pollster::block_on(t.train_step(&mut ws, &chunk, LAYERS as usize));
        assert!(last.is_finite(), "loss went non-finite at step {step}: {last}");
        gpu_clip_gradients(&ws, LAYERS as usize, 4.0);
        gpu_adam_update(&ws, 1e-3, step, 1, LAYERS as usize);
        ws.zero_accumulators();
    }

    assert!(
        last < first,
        "loss did not decrease over 30 steps: {first:.4} -> {last:.4}"
    );
}

/// Masked (SFT) and unmasked steps must agree when the mask trains everything.
///
/// An all-ones mask makes `train_step_masked` mathematically identical to
/// `train_step`, so any divergence is a bug in the masked branch of the
/// cross-entropy kernels rather than a difference in the training signal. This
/// catches a mask-indexing regression without needing an SFT dataset on disk.
#[test]
fn all_ones_mask_matches_unmasked_step() {
    let chunk = chunk();

    let (t, mut ws) = model_and_ws();
    let plain = pollster::block_on(t.train_step(&mut ws, &chunk, LAYERS as usize));

    // Same weights (Model::init is reseeded per call, so reuse the same model)
    // against a fresh workspace, with every position trained.
    let mut ws2 = Workspace::new(&t, CTX as usize);
    let mask = vec![1u8; chunk.len()];
    let masked =
        pollster::block_on(t.train_step_masked(&mut ws2, &chunk, &mask, LAYERS as usize));

    assert!(masked.is_finite(), "masked loss not finite: {masked}");
    assert!(
        (plain - masked).abs() < 1e-2,
        "all-ones mask changed the loss: unmasked {plain:.5} vs masked {masked:.5}"
    );
}
