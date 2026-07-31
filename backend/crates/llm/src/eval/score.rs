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

use crate::training::infer_session::InferSession;

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

    // True when every position produced a finite score, so `logprobs[i]` still lines
    // up with a mask built over the token sequence.
    //
    // `drop_unscored` removes a dropped position rather than holding its place, which
    // shifts every index after it — a caller that masks by index must refuse the
    // example instead of scoring the wrong tokens. `chat` and `preference` both do,
    // and both used to hand-write this comparison; it is the most consequential
    // invariant in the file and deserves one definition.
    pub fn aligned_with(&self, tokens_len: usize) -> bool {
        self.logprobs.len() + 1 == tokens_len
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
pub fn token_logprobs(session: &mut InferSession, tokens: &[u32]) -> TokenScores {
    if tokens.len() < 2 || session.ctx_window() < 2 {
        return TokenScores { logprobs: Vec::new() };
    }
    // Every position but the last predicts its successor.
    let mut logprobs = vec![f32::NAN; tokens.len() - 1];

    for_each_position(session, tokens, |pos, logits| {
        logprobs[pos] = log_softmax_at(logits, tokens[pos + 1] as usize);
    });

    drop_unscored(&mut logprobs);
    TokenScores { logprobs }
}

// What the model predicted at one position, beyond how it scored the token that
// actually came next.
//
// The markup benchmark needs this and the language one does not: asking "would the
// model have put `<|/par|>` here" is a question about the *argmax*, not about the
// reference token's probability. A model can assign a marker a respectable
// log-probability while confidently predicting something else, and only the first
// of those is a model that would emit well-formed markup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    // Indexes `tokens`; the target is `tokens[position + 1]`, the convention
    // `TokenScores` uses. Carried explicitly rather than implied by the vector index
    // because this vector is sparse — only selected positions appear.
    pub position: u32,
    // The same number `token_logprobs` produces here, off the same row, so the two
    // can never disagree.
    pub logprob: f32,
    // Highest-scoring id over the whole vocabulary. Ties resolve to the lowest id, so
    // the value is deterministic across runs.
    pub argmax: u32,
    // 1 = the target was the model's top choice. A tie counts *against* the model,
    // matching the strict `>` in `minimal_pairs` and `preference::paired_accuracy`: a
    // flat, opinionless model must not read as rank 1 everywhere.
    pub rank: u32,
    // Highest-scoring id inside the caller's `restrict` range. `None` when that range
    // is empty or starts past the model's output width — a corrupt `merges.bin` next
    // to the weights, the same condition `log_softmax_at` guards.
    pub restricted_argmax: Option<u32>,
}

// Dense scores plus sparse per-position detail, from one pass over the model.
pub struct TokenPredictions {
    // Exactly what `token_logprobs` would have returned for the same input, dropped
    // positions included, so `aligned_with` applies unchanged.
    pub scores: TokenScores,
    // Ascending by `position`. A selected position whose row did not score is absent.
    pub predictions: Vec<Prediction>,
}

// `token_logprobs`, additionally reporting the argmax and rank at selected positions.
//
// Rides the same forward passes — the extra cost is a few passes over each selected
// logit row on the host, invisible next to the matmuls that produced it.
//
// `restrict` is a parameter rather than a constant because this file is the scoring
// primitive every benchmark is built on and knows nothing about markers; `markup`
// passes the protected-marker block.
pub fn token_predictions(
    session: &mut InferSession,
    tokens: &[u32],
    // Parallel to `tokens`. Position `i` gets a `Prediction` iff `select[i] == 1`.
    // Shorter than `tokens` is legal and means "not selected" for the tail.
    select: &[u8],
    restrict: std::ops::Range<u32>,
) -> TokenPredictions {
    if tokens.len() < 2 || session.ctx_window() < 2 {
        return TokenPredictions {
            scores: TokenScores { logprobs: Vec::new() },
            predictions: Vec::new(),
        };
    }
    let mut logprobs = vec![f32::NAN; tokens.len() - 1];
    let mut predictions = Vec::new();

    for_each_position(session, tokens, |pos, logits| {
        let target = tokens[pos + 1] as usize;
        let logprob = log_softmax_at(logits, target);
        logprobs[pos] = logprob;
        if select.get(pos) != Some(&1) {
            return;
        }
        let (argmax, rank, restricted_argmax) = examine(logits, target, &restrict);
        predictions.push(Prediction {
            position: pos as u32,
            logprob,
            argmax,
            rank,
            restricted_argmax,
        });
    });

    // The two must agree about what counted, or a caller could read a prediction for
    // a position whose score was dropped.
    predictions.retain(|p| p.logprob.is_finite());
    drop_unscored(&mut logprobs);

    TokenPredictions { scores: TokenScores { logprobs }, predictions }
}

// `window_plan` covers every position exactly once, so nothing should be left
// unscored. Kept as a filter rather than an unwrap because a stray NaN would
// otherwise poison the whole corpus total, turning one bad position into a missing
// metric; the assertion catches it in development instead.
fn drop_unscored(values: &mut Vec<f32>) {
    debug_assert!(
        values.iter().all(|v| v.is_finite()),
        "window plan left {} of {} positions unscored",
        values.iter().filter(|v| !v.is_finite()).count(),
        values.len(),
    );
    values.retain(|v| v.is_finite());
}

// Argmax, the target's rank, and the argmax restricted to `restrict`, from one logit
// row. Split out from the async path so a hand-built row can exercise it without a GPU.
//
// Rank counts every id scoring at least as high as the target, so a tie costs the
// model a rank — see `Prediction::rank`.
fn examine(logits: &[f32], target: usize, restrict: &std::ops::Range<u32>) -> (u32, u32, Option<u32>) {
    let mut argmax = 0u32;
    let mut best = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > best {
            best = l;
            argmax = i as u32;
        }
    }

    // A target outside the row is already scored as maximally surprising by
    // `log_softmax_at`; rank it last rather than indexing out of bounds.
    let rank = match logits.get(target) {
        Some(&at) => 1 + logits
            .iter()
            .enumerate()
            .filter(|&(i, &l)| i != target && l >= at)
            .count() as u32,
        None => logits.len() as u32 + 1,
    };

    let start = (restrict.start as usize).min(logits.len());
    let end = (restrict.end as usize).min(logits.len());
    let mut restricted_argmax = None;
    let mut restricted_best = f32::NEG_INFINITY;
    for (offset, &l) in logits[start..end].iter().enumerate() {
        if l > restricted_best {
            restricted_best = l;
            restricted_argmax = Some((start + offset) as u32);
        }
    }

    (argmax, rank, restricted_argmax)
}

// The only place in the module that runs the model.
//
// Walks `window_plan` and hands `visit` the full logit row for each position it is
// responsible for, exactly once, in strictly ascending order of position. Both public
// entry points are built on this so there is a single forward path and a single
// interpretation of the window arithmetic — the part of this file where a silent
// off-by-one produces a plausible wrong number rather than a crash.
//
// The row is borrowed out of the current window's logit buffer and is invalid once
// the callback returns: a visitor that wants to keep something must reduce it.
//
// Ascending order is a guarantee, not an accident — `window_plan`'s scored ranges are
// disjoint and emitted in increasing `start` — and `token_predictions` returns a
// sorted sparse vector on the strength of it.
fn for_each_position(
    session: &mut InferSession,
    tokens: &[u32],
    mut visit: impl FnMut(usize, &[f32]),
) {
    let ctx = session.ctx_window();
    let vocab = session.vocab();
    if tokens.len() < 2 || ctx < 2 {
        return;
    }
    for window in window_plan(tokens.len(), ctx) {
        let slice = &tokens[window.start..window.end];
        // Flat `[s * vocab]`, so a row is a subslice rather than its own `Vec`.
        let logits = session.logits_all(slice);

        for j in window.first_scored..slice.len() - 1 {
            visit(window.start + j, &logits[j * vocab..(j + 1) * vocab]);
        }
    }
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
pub fn sequence_logprob(session: &mut InferSession, tokens: &[u32]) -> f64 {
    token_logprobs(session, tokens).total()
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

    // Positions must arrive in order, not merely be covered: `token_predictions`
    // promises a sorted sparse vector and builds it by pushing as it goes.
    #[test]
    fn positions_are_visited_in_ascending_order() {
        for ctx in [2usize, 3, 4, 8, 16, 64] {
            for len in 2..=(ctx * 4 + 5) {
                let hits = covered(len, ctx);
                assert!(
                    hits.windows(2).all(|w| w[0] < w[1]),
                    "ctx={ctx} len={len}: positions are not ascending",
                );
            }
        }
    }

    // ── the alignment guard ──

    #[test]
    fn aligned_with_is_false_after_a_drop() {
        // 3 tokens ⇒ 2 scoreable positions.
        assert!(TokenScores { logprobs: vec![-1.0, -2.0] }.aligned_with(3));
        assert!(!TokenScores { logprobs: vec![-1.0] }.aligned_with(3));
    }

    // Pins the exact expression `chat` and `preference` used to hand-write.
    #[test]
    fn aligned_with_matches_the_hand_written_guard() {
        for len in 1..8usize {
            for scored in 0..8usize {
                let s = TokenScores { logprobs: vec![-1.0; scored] };
                assert_eq!(s.aligned_with(len), s.logprobs.len() == len.saturating_sub(1));
            }
        }
    }

    #[test]
    fn drop_unscored_leaves_a_fully_scored_vector_alone() {
        let mut values = vec![-1.0, -2.0, -0.5];
        drop_unscored(&mut values);
        assert_eq!(values, vec![-1.0, -2.0, -0.5]);
    }

    // An unscored position is a bug, so it trips the assertion in development…
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "window plan left 1 of 3 positions unscored")]
    fn drop_unscored_is_loud_about_a_gap_in_development() {
        drop_unscored(&mut vec![-1.0, f32::NAN, -2.0]);
    }

    // …and in release it drops the position rather than letting one NaN poison a
    // whole corpus total. Both halves of that bargain are deliberate.
    #[cfg(not(debug_assertions))]
    #[test]
    fn drop_unscored_removes_only_the_non_finite() {
        let mut values = vec![-1.0, f32::NAN, -2.0];
        drop_unscored(&mut values);
        assert_eq!(values, vec![-1.0, -2.0]);
    }

    // ── examine ──
    //
    // The row reducer behind `token_predictions`, exercised on hand-built rows so the
    // whole markup benchmark's arithmetic is testable without a GPU.

    fn full(len: usize) -> std::ops::Range<u32> {
        0..len as u32
    }

    #[test]
    fn examine_finds_the_argmax_and_ranks_it_first() {
        let (argmax, rank, _) = examine(&[0.1, 2.0, 0.4], 1, &full(3));
        assert_eq!(argmax, 1);
        assert_eq!(rank, 1);
    }

    #[test]
    fn examine_breaks_argmax_ties_to_the_lowest_id() {
        // Deterministic across runs is the point; either id is "correct" otherwise.
        let (argmax, _, _) = examine(&[1.0, 5.0, 5.0], 0, &full(3));
        assert_eq!(argmax, 1);
    }

    #[test]
    fn examine_counts_a_tie_against_the_model() {
        // The target ties the top logit. A model with no opinion must not read as
        // rank 1 — the same rule minimal_pairs applies with its strict `>`.
        let (_, rank, _) = examine(&[5.0, 5.0, 0.0], 1, &full(3));
        assert_eq!(rank, 2);
    }

    #[test]
    fn examine_rank_over_a_uniform_row_is_the_vocab_size() {
        // The degenerate case the tie rule exists for: an untrained model ranks
        // everything last, rather than everything first.
        let (_, rank, _) = examine(&[0.0; 16], 7, &full(16));
        assert_eq!(rank, 16);
    }

    #[test]
    fn examine_rank_grows_as_the_target_gets_less_likely() {
        let logits = [5.0, 4.0, 3.0, 2.0, 1.0];
        let rank = |t| examine(&logits, t, &full(5)).1;
        assert_eq!(rank(0), 1);
        assert_eq!(rank(2), 3);
        assert_eq!(rank(4), 5);
    }

    #[test]
    fn examine_restricted_argmax_ignores_a_larger_logit_outside_the_range() {
        // Id 0 wins outright, but the restricted answer must come from [2, 4).
        let (argmax, _, restricted) = examine(&[9.0, 8.0, 1.0, 3.0], 0, &(2..4));
        assert_eq!(argmax, 0);
        assert_eq!(restricted, Some(3));
    }

    #[test]
    fn examine_restricted_argmax_is_none_for_an_empty_or_out_of_range_restrict() {
        assert_eq!(examine(&[1.0, 2.0], 0, &(1..1)).2, None);
        // A vocabulary that does not reach the block at all — a mismatched tokenizer.
        assert_eq!(examine(&[1.0, 2.0], 0, &(256..320)).2, None);
    }

    #[test]
    fn examine_restricted_argmax_is_clipped_to_the_row() {
        // Range overhangs the row; the in-range part still answers.
        assert_eq!(examine(&[1.0, 5.0, 2.0], 0, &(1..99)).2, Some(1));
    }

    #[test]
    fn examine_out_of_range_target_does_not_panic() {
        // Mirrors the `log_softmax_at` guard: a mismatched merges.bin must not abort
        // a benchmark mid-run.
        let (_, rank, _) = examine(&[1.0, 2.0], 9, &full(2));
        assert_eq!(rank, 3);
    }
}
