// Deterministic sampling for the benchmark's generation stage.
//
// `infer` has its own sampler, but it draws from the thread RNG and works through
// the global model cache — both fine for serving a chat, both wrong for a
// benchmark, where the same model and the same prompts must produce the same
// numbers on every run or the scores can't be compared across models. So this is a
// seeded sampler over a directly-supplied `Model`.
//
// Pure greedy decoding would be the simplest way to be deterministic, but it makes
// small models loop almost immediately, which would peg the repetition metric at
// its floor for every model and destroy the metric's ability to separate them.
// Seeded temperature sampling is deterministic *and* representative of what the
// model actually produces in use.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::training::transformer::Model;

// Decoding settings for the generation stage. Defaults mirror a typical chat
// request so the surface metrics describe text the model would really emit.
#[derive(Clone, Copy)]
pub struct SamplingConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        SamplingConfig {
            max_tokens: 96,
            temperature: 0.8,
            top_k: 40,
            seed: 42,
        }
    }
}

// Continue `prompt` and return only the newly generated text. The prompt's own
// tokens are excluded from the returned string so the surface metrics score the
// model's output rather than the fixed prompt set, which would otherwise dominate
// the word-rate on short generations.
pub async fn continue_prompt(
    model: &Model,
    merges: &[((u32, u32), u32)],
    prompt: &str,
    config: SamplingConfig,
) -> String {
    let seq = tokenize::bpe::encode(prompt, merges);
    continue_tokens(model, merges, seq, config).await.text
}

/// What one generation produced.
pub struct Continuation {
    /// The newly generated text, terminal marker included (the caller decides
    /// whether that belongs in what it stores).
    pub text: String,
    /// The same tokens before decoding, prompt excluded.
    ///
    /// The markup benchmark works on these rather than on `text`: every marker is a
    /// single reserved id (`pretokenize::SPECIAL_BLOCK`), so checking that a reply's
    /// markup nests correctly is a walk over ids with no scanning, no ambiguity about
    /// prose that merely looks like `<|par|>`, and no dependence on the decoder
    /// round-tripping the private-use characters.
    pub ids: Vec<u32>,
    /// Whether generation stopped because the model emitted an end-of-turn marker,
    /// as opposed to running out of `max_tokens`.
    ///
    /// This is signal in its own right, not bookkeeping: a model that never closes
    /// its turn is producing a runaway reply, and that is exactly the failure the
    /// auto-labeller wants to score down. It is not recoverable from the text alone
    /// once the marker is stripped.
    pub stopped_on_marker: bool,
}

/// `continue_tokens` for a caller already on its own thread.
///
/// The async here is an implementation detail of the GPU path, not something the
/// backend should have to know about — same reason `kto::reference_logprob` is
/// blocking. Callers that drive this are worker threads, never the actix runtime.
pub fn continue_tokens_blocking(
    model: &Model,
    merges: &[((u32, u32), u32)],
    seq: Vec<u32>,
    config: SamplingConfig,
) -> Continuation {
    pollster::block_on(continue_tokens(model, merges, seq, config))
}

/// `continue_prompt` over already-tokenized input.
///
/// Split out because the chat path needs it: a chat prompt is assembled by
/// `chat_template::build_prompt`, which returns ids rather than a string, and
/// re-encoding its rendered text would shift the token boundaries at the marker
/// seams (see `build_training_example`'s note on why markers are encoded apart).
pub async fn continue_tokens(
    model: &Model,
    merges: &[((u32, u32), u32)],
    mut seq: Vec<u32>,
    config: SamplingConfig,
) -> Continuation {
    let ctx = model.ctx_window();
    let layers = model.layers as usize;

    // Seeded per prompt-length so each prompt gets its own stream while the run as
    // a whole stays reproducible.
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut generated: Vec<u32> = Vec::new();
    let mut stopped_on_marker = false;

    for _ in 0..config.max_tokens {
        let hidden = model.forward_seq(&seq, layers, ctx).await;
        if hidden.is_empty() {
            break;
        }
        // Only the final position predicts the next token; unembedding just that
        // row keeps the per-step cost at one vocab-sized matmul.
        let last = vec![hidden[hidden.len() - 1].clone()];
        let logits = model.unembed(&last).await;

        let next = sample(&logits[0], config.temperature, config.top_k, &mut rng);
        generated.push(next);
        seq.push(next);

        // An end-of-text or end-of-turn marker means the model considers the
        // passage finished; continuing past it would score scaffolding as prose.
        if is_terminal(next) {
            stopped_on_marker = true;
            break;
        }
    }

    Continuation {
        text: tokenize::bpe::decode(&generated, merges),
        ids: generated,
        stopped_on_marker,
    }
}

// The reserved ids that end a passage. Compared as ids rather than by decoding,
// so a partially-formed marker in the decoded text can't end generation early.
fn is_terminal(token: u32) -> bool {
    let terminal = ["<|endoftext|>", "<|/assistant|>"];
    terminal.iter().any(|marker| {
        tokenize::pretokenize::SPECIAL_TOKENS
            .iter()
            .find(|(name, _)| name == marker)
            .and_then(|&(_, ch)| tokenize::pretokenize::special_id(ch))
            == Some(token)
    })
}

// Top-k temperature sampling over the logit row, drawing from a supplied RNG.
// Softmax is taken over the truncated candidate set after the usual
// max-subtraction, so the distribution is normalized and can't overflow.
fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut StdRng) -> u32 {
    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l))
        .collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(top_k.max(1).min(candidates.len()));

    // Temperature 0 (or below) is argmax — the sorted head. Guarding here also
    // keeps the division below from producing inf.
    if temperature <= 0.0 {
        return candidates[0].0;
    }

    let max = candidates[0].1;
    let weights: Vec<f32> = candidates
        .iter()
        .map(|&(_, l)| ((l - max) / temperature).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return candidates[0].0;
    }

    let mut r = rng.gen::<f32>() * total;
    for (i, &(token, _)) in candidates.iter().enumerate() {
        r -= weights[i];
        if r <= 0.0 {
            return token;
        }
    }
    // Floating-point accumulation can leave `r` marginally positive after the
    // whole loop; the last candidate is the correct answer in that case.
    candidates[candidates.len() - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_temperature_is_argmax() {
        let mut rng = StdRng::seed_from_u64(1);
        let logits = [0.1, 0.9, 0.4];
        for _ in 0..10 {
            assert_eq!(sample(&logits, 0.0, 3, &mut rng), 1);
        }
    }

    #[test]
    fn top_k_restricts_the_candidate_set() {
        let mut rng = StdRng::seed_from_u64(7);
        let logits = [5.0, 4.0, -1.0, -2.0, -3.0];
        for _ in 0..200 {
            assert!(sample(&logits, 1.0, 2, &mut rng) < 2);
        }
    }

    #[test]
    fn the_same_seed_produces_the_same_draws() {
        let logits = [1.0, 0.9, 0.8, 0.7];
        let draw = |seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..20).map(|_| sample(&logits, 1.0, 4, &mut rng)).collect::<Vec<_>>()
        };
        assert_eq!(draw(99), draw(99));
    }

    #[test]
    fn huge_logits_do_not_overflow_into_a_degenerate_draw() {
        let mut rng = StdRng::seed_from_u64(3);
        let logits = [400.0, 399.0, 1.0];
        let token = sample(&logits, 1.0, 3, &mut rng);
        assert!(token < 3);
    }

    #[test]
    fn end_of_text_is_recognized_as_terminal() {
        let id = tokenize::pretokenize::special_id('\u{E000}').expect("endoftext id");
        assert!(is_terminal(id));
        assert!(!is_terminal(42));
    }
}
