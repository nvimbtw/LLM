//! Throughput probe for the training step.
//!
//! Ignored by default — it wants the GPU to itself and takes tens of seconds.
//!
//!   cargo test -p llm --release --test bench_step -- --ignored --nocapture
//!
//! This deliberately does NOT use the toy dimensions of the correctness tests.
//! What it measures is the *fixed* per-step cost — kernel launches and host
//! synchronization — which only becomes visible once there are enough layers and
//! gradient buffers for it to accumulate. At 2 layers there is nothing to see.
//!
//! Numbers are only comparable within one run of this file: a training job (or
//! anything else) sharing the GPU shifts the absolute values, so A/B a change by
//! running the probe before and after under the same conditions.

use llm::training::gpu_workspace::Workspace;
use llm::training::optimizer::{gpu_adam_update, gpu_clip_gradients};
use llm::training::transformer::Model;
use std::time::Instant;

// Mirrors `train::ACCUM_STEPS` so the optimizer (and the gradient clip with it)
// runs at the same duty cycle as a real run. The clip is a large part of what
// this probe exists to measure, so skipping it would flatter the result.
const ACCUM: usize = 4;

struct Bench {
    ms_per_step: f64,
    tokens_per_sec: f64,
}

#[allow(clippy::too_many_arguments)]
fn bench(
    d: i64,
    vocab: i64,
    ctx: i64,
    heads: i64,
    ff_mult: i64,
    layers: i64,
    steps: usize,
) -> Bench {
    let t = Model::init(d, vocab, ctx, heads, ff_mult, layers);
    let mut ws = Workspace::new(&t, ctx as usize);
    // Arbitrary but deterministic ids. Content does not affect timing — every
    // position does the same work regardless of which token sits there.
    let chunk: Vec<u32> = (0..ctx as u32).map(|i| (i * 7 + 3) % vocab as u32).collect();

    // Warm-up: the first steps pay cuBLAS handle creation and the lazy attention
    // and feed-forward scratch allocations. Timing those would measure setup.
    for _ in 0..5 {
        pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));
    }
    ws.zero_accumulators();

    let start = Instant::now();
    let mut adam_step = 0i32;
    for i in 1..=steps {
        let loss = pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));
        assert!(loss.is_finite(), "non-finite loss at step {i}");
        if i % ACCUM == 0 {
            adam_step += 1;
            gpu_clip_gradients(&ws, layers as usize, 4.0);
            gpu_adam_update(&ws, 1e-4, adam_step, ACCUM, layers as usize);
            ws.zero_accumulators();
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    Bench {
        ms_per_step: elapsed / steps as f64 * 1000.0,
        tokens_per_sec: (steps as f64 * ctx as f64) / elapsed,
    }
}

// Shakedown 1's exact shape, so the probe is directly comparable to the numbers
// a real run reports.
#[test]
#[ignore]
fn bench_shakedown_shape() {
    let steps = 100;
    let b = bench(256, 8192, 512, 4, 4, 6, steps);
    eprintln!(
        "shakedown shape (d256 L6 h4 ff4 ctx512): {:.1} ms/step, {:.0} tok/s",
        b.ms_per_step, b.tokens_per_sec
    );
    // Only prints when LLM_PROFILE_STEP=1; the warm-up steps are included in the
    // stage totals, so divide by the steps they actually cover.
    llm::training::transformer::dump_step_profile(steps + 5);
}

// Which dimension actually costs the time.
//
// Localizes the bottleneck without a profiler: from a baseline, change exactly
// one dimension and see what the step time does. Whatever moves it is where the
// time lives.
//
//   vocab   → the logits GEMM and the cross-entropy kernels
//   layers  → the per-layer transformer stack
//   ctx     → attention (quadratic) plus everything per-token (linear)
//   nothing → fixed per-step cost: launches and host synchronization
#[test]
#[ignore]
fn bench_shape_sensitivity() {
    let base = bench(256, 8192, 512, 4, 4, 6, 40);
    eprintln!(
        "baseline    d256 v8192 ctx512 L6 : {:.1} ms/step",
        base.ms_per_step
    );
    for (label, b) in [
        ("vocab 8192→1024", bench(256, 1024, 512, 4, 4, 6, 40)),
        ("layers   6→2   ", bench(256, 8192, 512, 4, 4, 2, 40)),
        ("ctx    512→256 ", bench(256, 8192, 256, 4, 4, 6, 40)),
        ("d      256→128 ", bench(128, 8192, 512, 4, 4, 6, 40)),
    ] {
        eprintln!(
            "  {label} : {:>7.1} ms/step   ({:+.0}% vs baseline)",
            b.ms_per_step,
            (b.ms_per_step / base.ms_per_step - 1.0) * 100.0
        );
    }
}

// Tokens/second against context length, same model otherwise.
//
// This is the measurement that decides whether a longer context is affordable.
// If the step is dominated by fixed per-step cost, tokens/sec *rises* with ctx
// (the same overhead amortizes over more tokens) and a long context is close to
// free. If the step is dominated by arithmetic, tokens/sec falls as the O(s²)
// attention term takes over. The crossover is the useful context window.
#[test]
#[ignore]
fn bench_ctx_sweep() {
    eprintln!("ctx      ms/step      tok/s");
    for ctx in [256, 512, 1024, 2048] {
        // Fewer steps at long ctx: each one does proportionally more work, and
        // the point is throughput, not a fixed sample count.
        let steps = (32768 / ctx).max(20) as usize;
        let b = bench(256, 8192, ctx, 4, 4, 6, steps);
        eprintln!("{:<5} {:>10.1} {:>10.0}", ctx, b.ms_per_step, b.tokens_per_sec);
    }
}
