//! Does attention actually learn? A copy task is the discriminator.
//!
//! The sequence is a random block repeated twice. Position i in the second copy
//! is perfectly predictable — but only by attending to the matching position in
//! the first copy. No amount of embedding, feed-forward or output-projection
//! capacity can do it: the token identity carries no information about what
//! follows, so a model whose attention is frozen at random init is stuck near
//! the unigram entropy, while one with working attention drops far below it.
//!
//! This is the test that fails if the attention backward is wrong. It stayed
//! green on the ordinary "loss goes down" gate for the life of the project,
//! because loss goes down on prose whether or not K is ever trained.

use llm::training::gpu_workspace::Workspace;
use llm::training::optimizer::{gpu_adam_update, gpu_clip_gradients};
use llm::training::transformer::Model;

const D: i64 = 128;
const VOCAB: i64 = 64;
const CTX: i64 = 64;
const HEADS: i64 = 4;
const FF_MULT: i64 = 4;
const LAYERS: i64 = 2;
const STEPS: usize = 4000;

// [block, block] — the second half is a copy of the first.
fn copy_chunk(seed: u64) -> Vec<u32> {
    let half = CTX as usize / 2;
    let mut state = seed | 1;
    let block: Vec<u32> = (0..half)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % VOCAB as u64) as u32
        })
        .collect();
    block.iter().chain(block.iter()).copied().collect()
}

#[test]
#[ignore = "GPU, ~30s — run with --ignored --test-threads=1"]
fn attention_learns_to_copy() {
    let t = Model::init(D, VOCAB, CTX, HEADS, FF_MULT, LAYERS);
    let mut ws = Workspace::new(&t, CTX as usize);
    ws.upload_weights(&t);

    let mut first = 0.0f32;
    let mut last = 0.0f32;
    let mut tail: Vec<f32> = Vec::new();
    for step in 0..STEPS {
        // A FRESH block every step. This is what makes the task attention-only:
        // with one fixed sequence the positional table alone memorizes it
        // (position i -> token), and a model with dead attention scores 0.0 —
        // which is exactly how an earlier version of this test passed against
        // the broken kernel. Random blocks cannot be memorized, so the only way
        // to predict the second half is to attend to the first.
        let chunk = copy_chunk(0xC0FFEE ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let loss = pollster::block_on(t.train_step(&mut ws, &chunk, LAYERS as usize));
        gpu_clip_gradients(&ws, LAYERS as usize, 1.0);
        gpu_adam_update(&ws, 1e-3, step as i32 + 1, 1, LAYERS as usize);
        ws.zero_accumulators();
        if step == 0 {
            first = loss;
        }
        last = loss;
        if step >= STEPS - 200 {
            tail.push(loss);
        }
        if step % 500 == 0 {
            println!("step {step:>4}  loss {loss:.4}");
        }
    }
    println!("first {first:.4} -> last {last:.4} (ln(vocab) = {:.4})", (VOCAB as f32).ln());

    // The first half is fresh noise (~ln(vocab) per token, irreducible); the
    // second half is fully determined by it. So the floor for the sequence
    // average is about ln(vocab)/2 = 2.77, and a model that cannot attend is
    // stuck near ln(vocab) = 5.55. Anything under 4.0 means real copying.
    let avg_tail: f32 = tail.iter().sum::<f32>() / tail.len() as f32;
    println!("mean loss over the last 200 steps: {avg_tail:.4}");
    assert!(
        avg_tail < 4.0,
        "mean tail loss {avg_tail:.4} (last {last:.4}, started {first:.4}) \
         — attention is not learning to copy",
    );
}
