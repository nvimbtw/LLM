//! Throughput probe for the inference path.
//!
//! Ignored by default — it wants the GPU to itself.
//!
//!   cargo test -p llm --release --test bench_generate -- --ignored --nocapture --test-threads=1
//!
//! The counterpart to `bench_step.rs`. Two numbers are reported, and they answer
//! different questions:
//!
//!   * **prefill ms** — time to first token: one forward over the whole prompt with
//!     a cold cache. Grows superlinearly with prompt length, because that forward
//!     contains the O(s²) attention.
//!   * **ms/token** — steady-state decode against a warm KV cache. Should be
//!     roughly FLAT in prompt length.
//!
//! Their ratio is also the KV cache's speedup, for free and without keeping the old
//! code around: before the cache, *every* token cost a full prefill of the sequence
//! so far, which is exactly what `prefill_ms` measures.
//!
//! Beware that a steady-state token is ~2 ms at the shapes here, so run-to-run
//! spread is a large fraction of the number. Read a single row as ±40% and compare
//! trends across a column, not individual values.
//!
//! `--test-threads=1` is not optional: the attention/feed-forward scratch is
//! file-static and the cuBLAS handle is shared, so two of these running at once
//! corrupt each other (see `backend/CLAUDE.md`).
//!
//! Numbers are comparable only within one run under the same conditions. A/B a
//! change by running the probe before and after.

use llm::training::infer_session::InferSession;
use llm::training::transformer::Model;
use std::time::Instant;

struct Bench {
    /// Time to first token: one cold prefill of the whole prompt, on a session that
    /// has never seen it. What a user waits for before any text appears.
    prefill_ms: f64,
    /// Steady-state decode: one new token against a warm KV cache. What the tokens
    /// then arrive at.
    ms_per_token: f64,
    tokens_per_sec: f64,
}

// Argmax rather than the real sampler. What this probe measures is the forward
// path; the sampler's own cost (a full-vocab sort per token today) is a separate
// question, and argmax also removes run-to-run RNG variance from the token ids,
// which keeps successive sequence lengths identical between A and B runs.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

// Mirrors `infer::stream_generation` minus the tokenizer: forward the whole
// sequence, unembed the last position, pick a token, append, repeat.
#[allow(clippy::too_many_arguments)]
fn bench(
    d: i64,
    vocab: i64,
    ctx: i64,
    heads: i64,
    ff_mult: i64,
    layers: i64,
    prompt_len: usize,
    new_tokens: usize,
) -> Bench {
    let t = Model::init(d, vocab, ctx, heads, ff_mult, layers);

    // Deterministic prompt. Content does not affect timing — every position does
    // the same work regardless of which token sits there.
    let prompt: Vec<u32> = (0..prompt_len as u32)
        .map(|i| (i * 7 + 3) % vocab as u32)
        .collect();

    // Session construction is excluded from the measurement on purpose: it uploads
    // every weight, and that cost is paid once per loaded model, not per token.
    // It also pre-sizes the shared attention scratch at `ctx`, so unlike the old
    // path there is no per-token reallocation left for a warm-up to hide.
    let mut session = InferSession::new(&t, ctx as usize);

    // Warm cuBLAS before timing anything.
    //
    // cuBLAS selects (and on first use, allocates) a workspace per GEMM shape, and
    // that one-time cost is large — tens of ms on the first call at a new shape,
    // which would swamp both numbers below. It is genuinely not a recurring cost.
    //
    // This runs on a throwaway session so the *measured* session is still cold for
    // its prefill: the prefill is timed on a KV cache that has never been filled,
    // which is what time-to-first-token means.
    {
        let mut warmup = InferSession::new(&t, ctx as usize);
        let mut warm = prompt.clone();
        for _ in 0..3 {
            let logits = warmup.logits_last(&warm);
            warm.push(argmax(&logits));
        }
    }

    // Time to first token: one prefill of the whole prompt, cache cold.
    let prefill_start = Instant::now();
    let logits = session.logits_last(&prompt);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    assert!(logits.iter().all(|v| v.is_finite()), "non-finite logit from the prefill");

    // Steady state: every iteration adds exactly one token, so every one takes the
    // decode path against the warm cache.
    let mut seq = prompt.clone();
    seq.push(argmax(&logits));
    let start = Instant::now();
    for i in 0..new_tokens {
        let logits = session.logits_last(&seq);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite logit at generated token {i}"
        );
        seq.push(argmax(&logits));
    }
    let elapsed = start.elapsed().as_secs_f64();

    Bench {
        prefill_ms,
        ms_per_token: elapsed / new_tokens as f64 * 1000.0,
        tokens_per_sec: new_tokens as f64 / elapsed,
    }
}

// Shakedown 1's shape, matching `bench_step::bench_shakedown_shape`, so
// inference and training numbers are read against the same model.
#[test]
#[ignore]
fn bench_generate_shakedown_shape() {
    for prompt_len in [32usize, 512] {
        let new_tokens = 128;
        let b = bench(256, 8192, 512, 4, 4, 6, prompt_len, new_tokens);
        eprintln!(
            "d256 L6 h4 ff4 ctx512, prompt {:>4}: prefill {:>7.1} ms, \
             {:>6.2} ms/token, {:>6.1} tok/s",
            prompt_len, b.prefill_ms, b.ms_per_token, b.tokens_per_sec
        );
        llm::training::transformer::dump_infer_profile(new_tokens);
    }
}

// Which dimension actually costs the time, same method as
// `bench_step::bench_shape_sensitivity`: from a baseline, change exactly one
// thing and see what moves.
//
// Read this one with care: a decode step at these shapes is launch-bound, not
// arithmetic-bound (~100 kernel launches against ~0.1 ms of actual FLOPs), so most
// rows move much less than their FLOP count would suggest and some move the "wrong"
// way inside the noise. `layers` is the row that reliably matters, because launches
// scale with it.
//
//   layers     → the per-layer stack, and the launch count with it
//   prompt_len → the attention term only, now that the cache carries the rest
//   vocab      → the unembed GEMM
//   d          → the projection GEMMs
#[test]
#[ignore]
fn bench_generate_shape_sensitivity() {
    let base = bench(256, 8192, 512, 4, 4, 6, 128, 96);
    eprintln!(
        "baseline    d256 v8192 ctx512 L6 prompt128 : {:.1} ms/token",
        base.ms_per_token
    );
    for (label, b) in [
        ("prompt 128→384 ", bench(256, 8192, 512, 4, 4, 6, 384, 96)),
        ("prompt 128→32  ", bench(256, 8192, 512, 4, 4, 6, 32, 96)),
        ("layers   6→2   ", bench(256, 8192, 512, 4, 4, 2, 128, 96)),
        ("vocab 8192→1024", bench(256, 1024, 512, 4, 4, 6, 128, 96)),
        ("heads    4→8   ", bench(256, 8192, 512, 8, 4, 6, 128, 96)),
        ("d      256→128 ", bench(128, 8192, 512, 4, 4, 6, 128, 96)),
    ] {
        eprintln!(
            "  {label} : {:>8.1} ms/token   ({:+.0}% vs baseline)",
            b.ms_per_token,
            (b.ms_per_token / base.ms_per_token - 1.0) * 100.0
        );
    }
}

// Does the KV cache actually remove the O(s²) term?
//
// At the shape that ships (d256 L6 ctx512) the answer is invisible, because a
// per-token forward there costs ~70 kernel launches and only ~0.1 ms of arithmetic
// — it is launch-bound, and the prefix recompute the cache removes was never the
// binding constraint. That is a fact about the model's size, not about the cache.
//
// This is the measurement that shows the cache working: at a long context the
// recompute term is unambiguous, and ms/token should stay roughly FLAT as the
// prefix grows. Without a cache it would climb with the prefix, superlinearly once
// the s² attention term takes over.
#[test]
#[ignore]
fn bench_generate_long_context() {
    let ctx = 4096;
    eprintln!("ctx {ctx}, d256 L6 h4 — ms/token against how much prefix is cached");
    eprintln!("prompt   prefill ms   ms/token      tok/s   cache speedup");
    for prompt_len in [128usize, 512, 1024, 2048, 4000] {
        let b = bench(256, 8192, ctx, 4, 4, 6, prompt_len, 64);
        eprintln!(
            "{:<8} {:>11.2} {:>10.2} {:>10.1} {:>13.1}x",
            prompt_len,
            b.prefill_ms,
            b.ms_per_token,
            b.tokens_per_sec,
            b.prefill_ms / b.ms_per_token,
        );
    }
}

// Prefix reuse: N candidates from one prompt, which is what `labeling::autolabel`
// does (`for prompt × for k in 0..candidates`, each from the same `seq.clone()`).
//
// The second and later candidates arrive with the prompt already fully cached, so
// they pay one decode step instead of a fresh prefill.
//
// **What this is worth is entirely a function of the prompt-to-generation ratio,**
// and measuring it is the only way to know. At ctx 512 with 24 tokens per candidate
// it is worth nothing (0.8-1.2x, i.e. inside the noise): five saved prefills at
// ~2.5 ms against 144 decode steps at ~1.8 ms is a 5% effect. At ctx 4096 with a
// 4000-token prompt and 8 tokens per candidate it is worth 2.3x, because there the
// prefill is 35 ms and dominates everything else.
//
// Do not read the ctx-512 rows as "the cache does not work". They say the prefill
// was never the cost in that regime.
#[test]
#[ignore]
fn bench_generate_shared_prefill() {
    // Two regimes, because the answer differs sharply between them:
    //
    //   * short prompt, long generation — the prefill is a rounding error next to
    //     the decode steps, and reuse saves nothing measurable.
    //   * long prompt, short generation — the prefill dominates, and reuse is the
    //     difference between paying it once and paying it per candidate. This is
    //     the shape autolabel is actually in when prompts are conversations.
    for (ctx, prompt_lens, per_candidate) in [
        (512i64, vec![64usize, 256, 480], 24usize),
        (4096, vec![1024, 4000], 8),
    ] {
        let (d, vocab, heads, ff, layers) = (256i64, 8192i64, 4i64, 4i64, 6i64);
        let candidates = 6;
        let t = Model::init(d, vocab, ctx, heads, ff, layers);

        for prompt_len in prompt_lens {
            let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| (i * 7 + 3) % vocab as u32).collect();

            // Warm cuBLAS on a throwaway session.
            {
                let mut w = InferSession::new(&t, ctx as usize);
                let _ = w.logits_last(&prompt);
            }

            // Shared: one session across all candidates.
            let mut shared = InferSession::new(&t, ctx as usize);
            let start = Instant::now();
            for _ in 0..candidates {
                let mut seq = prompt.clone();
                for _ in 0..per_candidate {
                    let logits = shared.logits_last(&seq);
                    seq.push(argmax(&logits));
                }
            }
            let shared_ms = start.elapsed().as_secs_f64() * 1000.0;

            // No reuse: the same session, same amount of work, but each candidate starts
            // from a *different* prompt of the same length, so the prefix comparison
            // fails and every candidate re-prefills.
            //
            // Deliberately not "a fresh session per candidate": that would also re-upload
            // every weight, and this is meant to isolate the prefill, not to measure
            // session construction.
            let mut unshared = InferSession::new(&t, ctx as usize);
            let start = Instant::now();
            for c in 0..candidates {
                let mut seq: Vec<u32> = prompt.clone();
                seq[0] = ((c as u32) * 37 + 1) % vocab as u32;
                for _ in 0..per_candidate {
                    let logits = unshared.logits_last(&seq);
                    seq.push(argmax(&logits));
                }
            }
            let unshared_ms = start.elapsed().as_secs_f64() * 1000.0;

            eprintln!(
                "ctx {:>4} prompt {:>4}, {candidates} x {per_candidate} tokens: \
                 shared prefill {:>7.1} ms, re-prefilled {:>7.1} ms  ({:.2}x)",
                ctx,
                prompt_len,
                shared_ms,
                unshared_ms,
                unshared_ms / shared_ms,
            );
        }
    }
}

// How per-token cost scales with how much context is already there.
//
// With no KV cache this should climb roughly linearly with prompt length (each
// token re-forwards the whole prefix). Once a cache exists it should be nearly
// flat, with only the attention term growing. This is the measurement that says
// whether the cache landed.
#[test]
#[ignore]
fn bench_generate_prompt_sweep() {
    eprintln!("prompt   prefill ms   ms/token      tok/s");
    for prompt_len in [16usize, 64, 128, 256, 384] {
        let b = bench(256, 8192, 512, 4, 4, 6, prompt_len, 96);
        eprintln!(
            "{:<8} {:>11.2} {:>10.2} {:>10.1}",
            prompt_len, b.prefill_ms, b.ms_per_token, b.tokens_per_sec
        );
    }
}
