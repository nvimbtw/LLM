// The scoring primitive every benchmark is built on.
//
// Generation answers "what would the model say"; scoring answers "how likely did
// the model think the text it was *shown* was". That second question is what a
// benchmark needs: held-out perplexity, minimal-pair grammar judgements, and the
// n-gram comparison are all just sums of `log p(next token | prefix)` over text
// the model didn't write. So there is exactly one forward path here, and the rest
// of the module is arithmetic on its output.
//
// No new CUDA kernels: `forward_seq` + `unembed` already produce a hidden state
// and a logit row for *every* position in one pass, which is precisely a
// teacher-forced score. The only work left is a log-softmax on the host.

use crate::training::transformer::Model;

// Log-probabilities the model assigned to the tokens it was actually shown.
// `logprobs[i]` is `log p(tokens[i + 1] | tokens[..=i])`, so the vector is one
// shorter than the input — the first token is context, never a prediction.
pub struct TokenScores {
    pub logprobs: Vec<f32>,
}

impl TokenScores {
    // Natural-log total. Cross-entropy sums, so this is the quantity that
    // composes: concatenating two scored spans adds their totals.
    pub fn total(&self) -> f64 {
        self.logprobs.iter().map(|&l| l as f64).sum()
    }

    pub fn count(&self) -> usize {
        self.logprobs.len()
    }
}

// Score a whole token sequence under the model, sliding a window when the text is
// longer than the context.
//
// The window matters for honesty. A model with a 64-token context cannot condition
// on token 500, so scoring a long document in one shot is impossible — but scoring
// it in *disjoint* chunks is also wrong in the other direction: every chunk's first
// tokens would be predicted from nothing, inflating perplexity with an artifact of
// where we happened to cut. So windows overlap by half the context and each window
// only contributes the positions the previous one couldn't score with decent
// context. Every token after the first is scored exactly once, with at least
// `ctx/2` tokens of history (except in the very first window, where less history is
// all that exists).
pub async fn token_logprobs(model: &Model, tokens: &[u32], layers: usize) -> TokenScores {
    let ctx = model.positional_table.len();
    if tokens.len() < 2 || ctx < 2 {
        return TokenScores { logprobs: Vec::new() };
    }

    // Every position but the last predicts its successor.
    let mut logprobs = vec![f32::NAN; tokens.len() - 1];

    for window in window_plan(tokens.len(), ctx) {
        let slice = &tokens[window.start..window.end];
        let hidden = model.forward_seq(slice, layers, ctx).await;
        let logits = model.unembed(&hidden).await;

        for j in window.first_scored..slice.len() - 1 {
            let target = slice[j + 1] as usize;
            logprobs[window.start + j] = log_softmax_at(&logits[j], target);
        }
    }

    // `window_plan` covers every position exactly once, so nothing should be left
    // unscored. Kept as a filter rather than an unwrap because a stray NaN would
    // otherwise poison the whole corpus total, turning one bad position into a
    // missing metric; the assertion catches it in development instead.
    debug_assert!(
        logprobs.iter().all(|l| l.is_finite()),
        "window plan left {} of {} positions unscored",
        logprobs.iter().filter(|l| !l.is_finite()).count(),
        logprobs.len(),
    );
    logprobs.retain(|l| l.is_finite());

    TokenScores { logprobs }
}

// One forward pass: the token span to run, and the first position within it whose
// prediction we keep.
#[derive(Debug, PartialEq)]
struct Window {
    start: usize,
    end: usize,
    first_scored: usize,
}

// Plans the sliding windows so that every position from 0 to len-2 is scored
// exactly once, each with as much left context as the window allows.
//
// The index math is fiddly enough to be worth isolating: position `j` of a window
// predicts token `j + 1`, so a window ending at `end` can score up to global index
// `end - 2`. The next window must therefore begin scoring at `end - 1` — not at
// `end`, which was the original off-by-one and silently left one position per
// window unscored.
fn window_plan(len: usize, ctx: usize) -> Vec<Window> {
    let stride = (ctx / 2).max(1);
    let mut windows = Vec::new();
    let mut start = 0usize;

    loop {
        let end = (start + ctx).min(len);
        if end - start < 2 {
            break;
        }
        windows.push(Window {
            start,
            end,
            // The first window has no predecessor and so contributes everything.
            // Later windows pick up exactly where the previous one had to stop:
            // global index `start_prev + ctx - 1`, which is this window's local
            // `ctx - stride - 1`. Those positions still carry `ctx - stride`
            // tokens of history, which is why the overlap exists at all.
            first_scored: if start == 0 { 0 } else { ctx - stride - 1 },
        });
        if end == len {
            break;
        }
        start += stride;
    }

    windows
}

// Total log-probability of a full sequence — the quantity minimal-pair judgements
// compare. Conditioning on the first token (rather than scoring it from an empty
// prefix) is fine because both sentences in a pair start the same way in practice,
// and a constant offset cancels in a comparison.
pub async fn sequence_logprob(model: &Model, tokens: &[u32], layers: usize) -> f64 {
    token_logprobs(model, tokens, layers).await.total()
}

// log softmax evaluated at one index, without materializing the whole normalized
// row. Subtracting the max before exponentiating is the usual guard: raw logits
// off an f16 matmul can be large enough that `exp` overflows to inf, which would
// turn the whole score into NaN.
fn log_softmax_at(logits: &[f32], index: usize) -> f32 {
    if index >= logits.len() {
        // The tokenizer and the model's output width disagree — a corrupt or
        // mismatched merges.bin next to the weights. Score it as maximally
        // surprising rather than panicking mid-benchmark.
        return f32::MIN;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f32::MIN;
    }
    let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    logits[index] - max - sum_exp.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_matches_hand_computation() {
        // Two equal logits ⇒ p = 0.5 ⇒ log p = -ln 2.
        let lp = log_softmax_at(&[1.0, 1.0], 0);
        assert!((lp - -std::f32::consts::LN_2).abs() < 1e-5, "got {lp}");
    }

    #[test]
    fn log_softmax_is_a_normalized_distribution() {
        let logits = [0.3, -1.2, 2.5, 0.0];
        let total: f32 = (0..logits.len())
            .map(|i| log_softmax_at(&logits, i).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-5, "probabilities summed to {total}");
    }

    #[test]
    fn log_softmax_survives_logits_that_would_overflow_exp() {
        // exp(200) is inf in f32; the max-subtraction is what keeps this finite.
        let lp = log_softmax_at(&[200.0, 200.0], 1);
        assert!(lp.is_finite(), "overflowed to {lp}");
        assert!((lp - -std::f32::consts::LN_2).abs() < 1e-4);
    }

    #[test]
    fn log_softmax_out_of_range_index_does_not_panic() {
        assert_eq!(log_softmax_at(&[1.0, 2.0], 9), f32::MIN);
    }

    // ── window_plan ──
    //
    // These are the tests that matter most in this file: the sliding window is the
    // one piece of index arithmetic where a silent off-by-one produces a plausible
    // but wrong perplexity rather than a crash.

    // Which global positions a plan actually fills.
    fn covered(len: usize, ctx: usize) -> Vec<usize> {
        let mut hits = Vec::new();
        for window in window_plan(len, ctx) {
            for j in window.first_scored..(window.end - window.start) - 1 {
                hits.push(window.start + j);
            }
        }
        hits
    }

    #[test]
    fn every_position_is_scored_exactly_once_across_many_shapes() {
        // The original bug only appeared once a sequence was long enough to need a
        // third window, which is why it survived until a real corpus hit it.
        for ctx in [2usize, 3, 4, 8, 16, 64] {
            for len in 2..=(ctx * 4 + 5) {
                let hits = covered(len, ctx);
                let mut sorted = hits.clone();
                sorted.sort_unstable();
                sorted.dedup();

                assert_eq!(
                    sorted,
                    (0..len - 1).collect::<Vec<_>>(),
                    "ctx={ctx} len={len}: coverage has a gap",
                );
                assert_eq!(
                    hits.len(),
                    sorted.len(),
                    "ctx={ctx} len={len}: a position was scored twice",
                );
            }
        }
    }

    #[test]
    fn a_sequence_shorter_than_the_context_is_one_window() {
        assert_eq!(
            window_plan(10, 64),
            vec![Window { start: 0, end: 10, first_scored: 0 }],
        );
    }

    #[test]
    fn a_long_sequence_slides_by_half_the_context() {
        let plan = window_plan(200, 64);
        assert_eq!(plan[0], Window { start: 0, end: 64, first_scored: 0 });
        // Stride 32, and scoring resumes at local 31 — the position predicting the
        // token right after the previous window's last.
        assert_eq!(plan[1], Window { start: 32, end: 96, first_scored: 31 });
        assert_eq!(plan[2], Window { start: 64, end: 128, first_scored: 31 });
    }

    #[test]
    fn later_windows_keep_at_least_half_a_context_of_history() {
        // The point of overlapping at all: no position is predicted from a nearly
        // empty prefix just because a window boundary landed there.
        for window in window_plan(500, 64).into_iter().skip(1) {
            assert!(window.first_scored >= 64 / 2 - 1, "{window:?}");
        }
    }

    #[test]
    fn a_sequence_too_short_to_predict_yields_no_windows() {
        assert!(window_plan(1, 64).is_empty());
        assert!(window_plan(0, 64).is_empty());
    }

    #[test]
    fn totals_sum_the_logprobs() {
        let s = TokenScores { logprobs: vec![-1.0, -2.0, -0.5] };
        assert!((s.total() - -3.5).abs() < 1e-9);
        assert_eq!(s.count(), 3);
    }
}
