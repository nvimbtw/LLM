//! Regression guard for ISSUE-0093 — backward scratch sized for the context
//! window, not for the model dimension.
//!
//!   cargo test -p llm --test scratch_bounds -- --test-threads=1
//!
//! `--test-threads=1` is required for the same reason as `kernel_sanity`: the
//! GPU scratch and the cuBLAS handle are process-wide.
//!
//! The LN2 backward writes its `dx` as `rows * cols` **halves**, and it used to
//! be handed `b_dd` — a buffer sized `d * d` in **f32** for the unrelated `wo`
//! gradient. That fit only while `s <= 2 * d`. Every run up to and including
//! Shakedown 1 was at `ctx == 2 * d` (512 at d 256), i.e. exactly at the
//! ceiling, which is why it never surfaced; Balanced 1 at ctx 1024 wrote 512 KB
//! into a 256 KB allocation on every layer of every step.
//!
//! Nothing about that is visible from the loss — it stays finite and keeps
//! descending — so these tests check the sizing arithmetic directly rather than
//! trying to observe the corruption.

use llm::training::gpu_workspace::Workspace;
use llm::training::transformer::Model;

// ctx = 4 * d, well past the old ceiling, and small enough to run in a second.
const D: i64 = 32;
const VOCAB: i64 = 64;
const CTX: i64 = 128;
const HEADS: i64 = 2;
const FF_MULT: i64 = 2;
const LAYERS: i64 = 2;

fn workspace() -> (Model, Workspace) {
    let t = Model::init(D, VOCAB, CTX, HEADS, FF_MULT, LAYERS);
    let ws = Workspace::new(&t, CTX as usize);
    (t, ws)
}

/// The scratch the LN2 backward writes into must hold a full `[s, d]` f16 block
/// at the workspace's own `max_s`. This is the invariant that was violated.
#[test]
fn ln2_backward_scratch_holds_a_full_context_window() {
    let (_t, ws) = workspace();
    let needed = ws.max_s * ws.d * size_of::<half::f16>();
    let have = ws.ln2_dgrad.len * size_of::<half::f16>();
    assert!(
        have >= needed,
        "ln2_dgrad holds {have} bytes, kernel writes {needed} at max_s={} d={}",
        ws.max_s,
        ws.d,
    );
}

/// Pin that this shape really does exceed the old ceiling, so the test keeps
/// covering the regression if the constants above are ever edited. If `b_dwo`
/// were still doing double duty here it would be too small — which is the point.
#[test]
fn the_shape_under_test_exceeds_the_old_ceiling() {
    let (_t, ws) = workspace();
    let dwo_bytes = ws.b_dwo.len * size_of::<f32>();
    let ln_dx_bytes = ws.max_s * ws.d * size_of::<half::f16>();
    assert!(
        dwo_bytes < ln_dx_bytes,
        "ctx {} is not > 2 * d ({}), so this no longer covers ISSUE-0093",
        ws.max_s,
        2 * ws.d,
    );
}

/// The shapes of the two real pretrains, as a table, so the boundary is
/// documented in something executable: Shakedown 1 sat exactly on it and
/// Balanced 1 was over it.
#[test]
fn the_real_pretrain_shapes_straddle_the_old_ceiling() {
    let d = 256usize;
    let old_ceiling_bytes = d * d * size_of::<f32>();
    for (name, ctx, over) in [("Shakedown 1", 512usize, false), ("Balanced 1", 1024, true)] {
        let writes = ctx * d * size_of::<half::f16>();
        assert_eq!(
            writes > old_ceiling_bytes,
            over,
            "{name} (ctx {ctx}, d {d}): writes {writes} vs old ceiling {old_ceiling_bytes}",
        );
    }
}

/// End to end at `ctx > 2 * d`, which the old code could not do without writing
/// out of bounds. Weak assertion by design — `Workspace::new`'s own
/// `assert_scratch_fits` is the real gate; this just proves the shape runs.
#[test]
fn a_training_step_runs_with_ctx_beyond_twice_the_model_dimension() {
    let (t, mut ws) = workspace();
    let chunk: Vec<u32> = (0..CTX as u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    let loss = pollster::block_on(t.train_step(&mut ws, &chunk, LAYERS as usize));
    assert!(loss.is_finite(), "loss must stay finite at ctx > 2 * d, got {loss}");
}
