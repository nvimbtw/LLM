//! Where the *host* time goes in the generation loop.
//!
//!   cargo test -p llm --release --test bench_host_loop -- --ignored --nocapture
//!
//! Needs no GPU. Sizes the per-token host work against the ~1.9 ms a decode step
//! costs on the device, which is the only thing that says whether any of it matters.

use rand::SeedableRng;
use std::time::Instant;

fn build_merges(vocab: u32) -> Vec<((u32, u32), u32)> {
    // Shaped like a real merge table: each new token merges two existing ones.
    (256..vocab)
        .map(|t| {
            let a = (t - 256) % 256;
            let b = (t - 255) % 256;
            ((a, b), t)
        })
        .collect()
}

// The serving sampler as it stands today, copied so the probe measures the shipped
// code rather than whatever it is replaced with.
fn sample_token_full_sort(logits: &[f32], temperature: f32, top_k: u32) -> u32 {
    let mut candidates: Vec<(u32, f32)> =
        logits.iter().enumerate().map(|(i, &p)| (i as u32, p)).collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(top_k as usize);
    let total: f32 = candidates.iter().map(|(_, p)| (p / temperature).exp()).sum();
    let mut r = 0.5f32 * total;
    let mut next = candidates[0].0;
    for &(tok, prob) in &candidates {
        r -= (prob / temperature).exp();
        if r <= 0.0 {
            next = tok;
            break;
        }
    }
    next
}

#[test]
#[ignore]
fn bench_host_generation_loop() {
    let vocab = 8192u32;
    let merges = build_merges(vocab);
    let logits: Vec<f32> = (0..vocab).map(|i| ((i * 37 % 1000) as f32) / 500.0 - 1.0).collect();
    let reps = 200;

    let start = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(sample_token_full_sort(&logits, 0.8, 40));
    }
    let sampler_us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;

    // A reply of a few hundred tokens, decoded the way the loop does it: the whole
    // output, every token.
    for out_len in [64usize, 256, 1024] {
        let ids: Vec<u32> = (0..out_len as u32).map(|i| 256 + (i * 7) % (vocab - 256)).collect();
        let start = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(tokenize::bpe::decode(&ids, &merges));
        }
        let decode_us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
        eprintln!(
            "output {out_len:>5} tokens: bpe::decode {decode_us:>9.1} us per call \
             ({:.1}% of a 1.9 ms token)",
            decode_us / 1900.0 * 100.0
        );
    }
    eprintln!(
        "\nsampler (full vocab sort, top_k 40): {sampler_us:.1} us per call \
         ({:.1}% of a 1.9 ms token)",
        sampler_us / 1900.0 * 100.0
    );

    // How much of decode is rebuilding the merge map rather than expanding tokens?
    let ids: Vec<u32> = vec![256 + 5];
    let start = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(tokenize::bpe::decode(&ids, &merges));
    }
    let one_token_us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
    eprintln!(
        "bpe::decode of a SINGLE token: {one_token_us:.1} us — this is almost all \
         table rebuild, paid on every call ({:.1}% of a token)",
        one_token_us / 1900.0 * 100.0
    );

    // ── and what the shipped versions cost now ──
    eprintln!();
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    // top_p disabled, so this measures the same work as the full-sort baseline it
    // is being compared against rather than a sort plus a nucleus scan.
    let params = llm::sample::SamplingParams { temperature: 0.8, top_k: 40, top_p: 1.0 };
    let start = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(llm::sample::sample(&logits, params, &mut rng));
    }
    let new_sampler_us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
    eprintln!(
        "sampler, partition instead of full sort: {new_sampler_us:.1} us  ({:.1}x faster, \
         {:.1}% of a token)",
        sampler_us / new_sampler_us,
        new_sampler_us / 1900.0 * 100.0
    );

    let decoder = tokenize::bpe::Decoder::new(&merges);
    for out_len in [64usize, 256, 1024] {
        let ids: Vec<u32> = (0..out_len as u32).map(|i| 256 + (i * 7) % (vocab - 256)).collect();
        let start = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(decoder.decode(&ids));
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
        eprintln!(
            "Decoder::decode, table built once, {out_len:>5} tokens: {us:>8.1} us \
             ({:.1}% of a token)",
            us / 1900.0 * 100.0
        );
    }
}
