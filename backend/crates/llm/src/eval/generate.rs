// Deterministic sampling for the benchmark's generation stage.
//
// `infer` has its own sampler, but it draws from the thread RNG and works through
// the global model cache — both fine for serving a chat, both wrong for a
// benchmark, where the same model and the same prompts must produce the same
// numbers on every run or the scores can't be compared across models. So this is a
// seeded sampler over a directly-supplied `InferSession`.
//
// Pure greedy decoding would be the simplest way to be deterministic, but it makes
// small models loop almost immediately, which would peg the repetition metric at
// its floor for every model and destroy the metric's ability to separate them.
// Seeded temperature sampling is deterministic *and* representative of what the
// model actually produces in use.

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::training::infer_session::InferSession;

// Decoding settings for the generation stage. Defaults mirror a typical chat
// request so the surface metrics describe text the model would really emit.
#[derive(Clone, Copy)]
pub struct SamplingConfig {
    pub max_tokens: usize,
    /// What to do with each logit row. Held rather than restated so the benchmark
    /// and the chat path cannot drift on what "top-k 40" means.
    pub params: crate::sample::SamplingParams,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        SamplingConfig {
            max_tokens: 96,
            // Deliberately *not* `SamplingParams::default()`, which tracks the chat
            // defaults and is expected to move with them. A benchmark score is only
            // comparable against its own history, so these are pinned here and
            // changing one is a `SCORE_VERSION` bump. `top_p` is 1.0 — disabled —
            // because every score already on record was measured without a nucleus.
            params: crate::sample::SamplingParams {
                temperature: 0.8,
                top_k: 40,
                top_p: 1.0,
            },
            seed: 42,
        }
    }
}

// Continue `prompt` and return only the newly generated text. The prompt's own
// tokens are excluded from the returned string so the surface metrics score the
// model's output rather than the fixed prompt set, which would otherwise dominate
// the word-rate on short generations.
pub fn continue_prompt(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    prompt: &str,
    config: SamplingConfig,
) -> String {
    let seq = tokenize::bpe::encode(prompt, merges);
    continue_tokens(session, merges, seq, config).text
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

/// `continue_prompt` over already-tokenized input.
///
/// Split out because the chat path needs it: a chat prompt is assembled by
/// `chat_template::build_prompt`, which returns ids rather than a string, and
/// re-encoding its rendered text would shift the token boundaries at the marker
/// seams (see `build_training_example`'s note on why markers are encoded apart).
///
/// Blocking, and callers must already be on their own OS thread: this drives CUDA
/// synchronously and must never run on the actix runtime. It used to be `async`
/// with a `continue_tokens_blocking` wrapper, but the async awaited nothing — the
/// GPU path was only ever synchronous — so the wrapper described a distinction
/// that did not exist.
pub fn continue_tokens(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    mut seq: Vec<u32>,
    config: SamplingConfig,
) -> Continuation {
    // Seeded per prompt-length so each prompt gets its own stream while the run as
    // a whole stays reproducible.
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut generated: Vec<u32> = Vec::new();
    let mut stopped_on_marker = false;

    for _ in 0..config.max_tokens {
        // Only the final position predicts the next token, and `logits_last` keeps
        // the per-step cost at one vocab-sized matmul over one row. The session
        // truncates to its context window internally.
        let logits = session.logits_last(&seq);
        if logits.is_empty() {
            break;
        }

        let next = crate::sample::sample(&logits, config.params, &mut rng);
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

#[cfg(test)]
mod tests {
    use super::*;

    // The sampler itself is tested in `crate::sample`. What matters *here* is the
    // property this module exists for: seeding it makes a benchmark reproducible, so
    // the same model and prompts produce the same score on every run.
    #[test]
    fn the_same_seed_produces_the_same_draws() {
        let logits = [1.0, 0.9, 0.8, 0.7];
        let params =
            crate::sample::SamplingParams { temperature: 1.0, top_k: 4, top_p: 1.0 };
        let draw = |seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..20)
                .map(|_| crate::sample::sample(&logits, params, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(99), draw(99));
        assert_ne!(draw(99), draw(100));
    }

    #[test]
    fn end_of_text_is_recognized_as_terminal() {
        let id = tokenize::pretokenize::special_id('\u{E000}').expect("endoftext id");
        assert!(is_terminal(id));
        assert!(!is_terminal(42));
    }
}
