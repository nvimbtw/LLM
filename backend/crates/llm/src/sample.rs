//! Temperature / top-k / top-p sampling over a logit row.
//!
//! One definition, two callers with different randomness needs: `infer` serves chat
//! from the thread RNG, and `eval::generate` uses a seeded `StdRng` because a
//! benchmark has to produce the same numbers on every run or its scores cannot be
//! compared across models. That difference is the RNG, not the arithmetic, so the
//! RNG is a parameter and the arithmetic lives here.
//!
//! There used to be two copies, and they had drifted into disagreeing about
//! correctness rather than about determinism. The serving one
//! (`infer::sample_token`) exponentiated raw logits with **no max subtraction**, so
//! a confident model overflowed `exp` to `inf` and the whole distribution became
//! `NaN`; and `temperature = 0.0` divided by zero, returning the argmax only by
//! accident (every `r -= inf` gives `NaN`, `NaN <= 0.0` is false, so the loop fell
//! through to its initial value, which happened to be the top candidate). Its unit
//! test passed on that accident.

use rand::Rng;

/// The chat defaults. Serving is the only caller that has a *preference* rather
/// than a fixed protocol — the benchmark suites pin their own values so a score
/// stays comparable — so the numbers live here and the request layer falls back
/// to them when a client sends nothing.
pub const DEFAULT_TEMPERATURE: f32 = 0.8;
pub const DEFAULT_TOP_K: u32 = 40;
/// Below 1.0 because the models this serves are small enough that the tail of a
/// flat distribution is mostly noise; see the range checks in `SamplingParams`.
pub const DEFAULT_TOP_P: f32 = 0.9;

/// How to turn a logit row into a token.
///
/// The three filters compose in the order the fields are declared: `temperature`
/// reshapes the distribution, `top_k` caps how many candidates survive, and
/// `top_p` keeps the shortest prefix of those whose probabilities already sum to
/// `top_p`. Each is disabled by its own neutral value — temperature `0.0`
/// collapses to argmax, a `top_k` past the vocabulary keeps everything, and a
/// `top_p` of `1.0` cuts nothing — so a caller wanting one filter need not know
/// about the other two.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            top_k: DEFAULT_TOP_K,
            top_p: DEFAULT_TOP_P,
        }
    }
}

/// Pick a token from `logits`.
///
/// Every field of `params` is self-clamping, so no combination of values can ask
/// for an empty candidate set or divide by zero.
pub fn sample<R: Rng>(logits: &[f32], params: SamplingParams, rng: &mut R) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    // Non-finite as well as non-positive: dividing by NaN or infinity would make
    // every weight NaN or a uniform 1.0, neither of which is a distribution. Both
    // reach here only through a bug or a hand-written request, and argmax is the
    // one answer that is always defensible.
    if !params.temperature.is_finite() || params.temperature <= 0.0 {
        return argmax(logits);
    }

    let mut candidates = top_k_candidates(logits, params.top_k);

    // Softmax over the truncated set, max-subtracted so `exp` cannot overflow.
    let max = candidates[0].1;
    let mut weights: Vec<f32> = candidates
        .iter()
        .map(|&(_, l)| ((l - max) / params.temperature).exp())
        .collect();

    let keep = nucleus_len(&weights, params.top_p);
    candidates.truncate(keep);
    weights.truncate(keep);

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
    // Floating-point accumulation can leave `r` marginally positive after the whole
    // loop; the last candidate is the correct answer in that case.
    candidates[candidates.len() - 1].0
}

/// The `k` highest-scoring ids, ordered descending. Never empty for a non-empty
/// row: `k` is clamped to at least 1 and at most the vocabulary.
fn top_k_candidates(logits: &[f32], top_k: u32) -> Vec<(u32, f32)> {
    let k = (top_k as usize).clamp(1, logits.len());
    let mut candidates: Vec<(u32, f32)> =
        logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();

    // Partition so the k best are in front, then order just those.
    //
    // Sorting the whole vocabulary and truncating gave the same answer but cost
    // O(V log V) per token — 106 us at vocab 8,192, about 5% of a generated token,
    // to rank 8,152 candidates that get thrown away. This is O(V + k log k).
    //
    // The result is bit-identical to the old full sort because `by_score` is a
    // *total* order: ties break on the ascending token id, so which of two equal
    // logits lands in the top k, and in what order, does not depend on the
    // partitioning being stable. Without that tie-break, `select_nth_unstable`
    // could pick either of a tied pair and a seeded benchmark would stop
    // reproducing its own past runs.
    if k < candidates.len() {
        candidates.select_nth_unstable_by(k - 1, by_score);
        candidates.truncate(k);
    }
    candidates.sort_by(by_score);
    candidates
}

/// How many of `weights` the nucleus keeps: the shortest descending prefix whose
/// probabilities reach `top_p`.
///
/// `weights` is unnormalized (`exp` of the shifted logits), so the target is a
/// fraction of its sum rather than of 1.0 — normalizing first would cost a second
/// pass to buy nothing. A `top_p` outside `(0, 1)`, or a degenerate sum, keeps
/// everything and leaves the decision to top-k alone; the result is never 0 for a
/// non-empty input, so the caller cannot be handed an empty set.
fn nucleus_len(weights: &[f32], top_p: f32) -> usize {
    if !(top_p > 0.0 && top_p < 1.0) {
        return weights.len();
    }
    let total: f32 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return weights.len();
    }
    let target = total * top_p;
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if cumulative >= target {
            return i + 1;
        }
    }
    weights.len()
}

// Descending by logit, ascending by token id on a tie. Total, so ordering never
// depends on sort stability.
fn by_score(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.0.cmp(&b.0))
}

/// Highest-scoring id, ties resolving to the lowest id so the result is
/// deterministic across runs.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(7)
    }

    /// Temperature and top-k with the nucleus disabled, which is what every test
    /// predating top-p is asserting about.
    fn params(temperature: f32, top_k: u32) -> SamplingParams {
        SamplingParams { temperature, top_k, top_p: 1.0 }
    }

    #[test]
    fn zero_temperature_is_argmax() {
        let logits = [0.1, 0.9, 0.4];
        for _ in 0..10 {
            assert_eq!(sample(&logits, params(0.0, 3), &mut rng()), 1);
        }
    }

    #[test]
    fn negative_temperature_is_also_argmax() {
        // Guards the `<= 0.0` rather than `== 0.0`: a negative temperature would
        // otherwise invert the distribution and sample the *least* likely token.
        let logits = [0.1, 0.9, 0.4];
        assert_eq!(sample(&logits, params(-1.0, 3), &mut rng()), 1);
    }

    // Every comparison against NaN is false, so a plain `t <= 0.0` would let it
    // through and turn the whole distribution into NaN weights. Infinity is the
    // same guard from the other end: it would flatten every weight to 1.0.
    #[test]
    fn a_non_finite_temperature_is_argmax() {
        let logits = [0.1, 0.9, 0.4];
        for t in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(sample(&logits, params(t, 3), &mut rng()), 1, "temperature {t}");
        }
    }

    #[test]
    fn top_k_restricts_the_candidate_set() {
        let logits = [10.0, 9.0, 1.0, 0.5, 0.3];
        for _ in 0..200 {
            let t = sample(&logits, params(1.0, 2), &mut rng());
            assert!(t < 2, "top_k=2 returned {t}");
        }
    }

    #[test]
    fn top_k_larger_than_the_vocabulary_is_clamped() {
        let logits = [0.1, 0.2, 0.3];
        for _ in 0..20 {
            assert!(sample(&logits, params(1.0, 100), &mut rng()) < 3);
        }
    }

    #[test]
    fn top_k_zero_still_returns_a_token() {
        let logits = [0.1, 0.9, 0.4];
        assert_eq!(sample(&logits, params(1.0, 0), &mut rng()), 1);
    }

    #[test]
    fn an_empty_row_does_not_panic() {
        assert_eq!(sample(&[], params(1.0, 4), &mut rng()), 0);
    }

    // The nucleus is the point of the whole exercise: one token holding ~95% of
    // the mass should be the only survivor at top_p 0.9, however many candidates
    // top-k let through.
    #[test]
    fn top_p_keeps_only_the_nucleus() {
        let logits = [10.0, 3.0, 2.9, 2.8, 2.7];
        let p = SamplingParams { temperature: 1.0, top_k: 5, top_p: 0.9 };
        for _ in 0..200 {
            assert_eq!(sample(&logits, p, &mut rng()), 0);
        }
    }

    // top_p 1.0 must cut nothing at all, since it is the "disabled" value every
    // caller that does not care about the nucleus relies on.
    #[test]
    fn top_p_of_one_is_disabled() {
        let weights = [4.0, 3.0, 2.0, 1.0];
        assert_eq!(nucleus_len(&weights, 1.0), weights.len());
    }

    // A flat distribution is where top-p earns its keep over top-k: no token
    // dominates, so the nucleus is a real prefix rather than everything or one.
    #[test]
    fn top_p_takes_a_prefix_of_a_flat_distribution() {
        let weights = [1.0; 10];
        assert_eq!(nucleus_len(&weights, 0.5), 5);
        assert_eq!(nucleus_len(&weights, 0.25), 3);
    }

    // Out-of-range values are reachable from a hand-written request, and the
    // route clamps them — but the sampler must not depend on that having happened.
    #[test]
    fn an_out_of_range_top_p_keeps_everything() {
        let weights = [4.0, 3.0, 2.0, 1.0];
        for p in [0.0, -1.0, 2.0, f32::NAN] {
            assert_eq!(nucleus_len(&weights, p), weights.len(), "top_p {p} truncated");
        }
    }

    // Never an empty candidate set: a vanishingly small top_p still has to leave
    // the sampler something to return.
    #[test]
    fn a_tiny_top_p_still_keeps_one_token() {
        let logits = [5.0, 4.0, 3.0];
        let p = SamplingParams { temperature: 1.0, top_k: 3, top_p: f32::MIN_POSITIVE };
        assert_eq!(sample(&logits, p, &mut rng()), 0);
    }

    // The bug the old serving sampler had: no max subtraction, so `exp` of a large
    // logit overflowed to inf and the distribution became NaN. Large logits are not
    // hypothetical off an f16 matmul.
    #[test]
    fn large_logits_do_not_overflow() {
        let logits = [200.0, 199.0, 100.0, 1.0];
        for _ in 0..50 {
            let t = sample(&logits, params(0.8, 4), &mut rng());
            assert!(t < 4, "large logits produced {t}");
        }
    }

    #[test]
    fn negative_logits_sample_normally() {
        let logits = [-1.0, -2.0, -3.0, -4.0];
        for _ in 0..20 {
            assert!(sample(&logits, params(1.0, 4), &mut rng()) < 4);
        }
    }

    #[test]
    fn a_single_candidate_is_returned() {
        assert_eq!(sample(&[42.0], params(1.0, 1), &mut rng()), 0);
    }

    // Selection must agree with a full sort exactly, including on ties — this is
    // what lets a seeded benchmark keep reproducing its earlier runs.
    #[test]
    fn top_k_selection_matches_a_full_sort_including_ties() {
        let logits: Vec<f32> = (0..500)
            .map(|i| ((i * 37 % 11) as f32) / 4.0) // deliberately many exact ties
            .collect();
        let k = 40usize;

        let mut sorted: Vec<(u32, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
        sorted.sort_by(by_score);
        sorted.truncate(k);

        let mut picked: Vec<(u32, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
        picked.select_nth_unstable_by(k - 1, by_score);
        picked.truncate(k);
        picked.sort_by(by_score);

        assert_eq!(
            sorted, picked,
            "partition-then-sort disagreed with a full sort on a row full of ties"
        );
    }

    #[test]
    fn temperature_shapes_the_distribution() {
        // Low temperature should concentrate on the top token; high should spread.
        let logits = [3.0, 2.0, 1.0, 0.0];
        let mut r = rng();
        let cold = (0..400).filter(|_| sample(&logits, params(0.1, 4), &mut r) == 0).count();
        let hot = (0..400).filter(|_| sample(&logits, params(5.0, 4), &mut r) == 0).count();
        assert!(
            cold > hot,
            "temperature 0.1 picked the top token {cold}/400, temperature 5.0 {hot}/400"
        );
    }
}
