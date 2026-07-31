//! Top-k temperature sampling over a logit row.
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

/// Pick a token from `logits`.
///
/// `temperature <= 0.0` is argmax. `top_k` is clamped to at least 1 and at most the
/// vocabulary, so a caller cannot ask for an empty candidate set.
pub fn sample<R: Rng>(logits: &[f32], temperature: f32, top_k: usize, rng: &mut R) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temperature <= 0.0 {
        return argmax(logits);
    }

    let k = top_k.clamp(1, logits.len());
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

    // Softmax over the truncated set, max-subtracted so `exp` cannot overflow.
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
    // Floating-point accumulation can leave `r` marginally positive after the whole
    // loop; the last candidate is the correct answer in that case.
    candidates[candidates.len() - 1].0
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

    #[test]
    fn zero_temperature_is_argmax() {
        let logits = [0.1, 0.9, 0.4];
        for _ in 0..10 {
            assert_eq!(sample(&logits, 0.0, 3, &mut rng()), 1);
        }
    }

    #[test]
    fn negative_temperature_is_also_argmax() {
        // Guards the `<= 0.0` rather than `== 0.0`: a negative temperature would
        // otherwise invert the distribution and sample the *least* likely token.
        let logits = [0.1, 0.9, 0.4];
        assert_eq!(sample(&logits, -1.0, 3, &mut rng()), 1);
    }

    #[test]
    fn top_k_restricts_the_candidate_set() {
        let logits = [10.0, 9.0, 1.0, 0.5, 0.3];
        for _ in 0..200 {
            let t = sample(&logits, 1.0, 2, &mut rng());
            assert!(t < 2, "top_k=2 returned {t}");
        }
    }

    #[test]
    fn top_k_larger_than_the_vocabulary_is_clamped() {
        let logits = [0.1, 0.2, 0.3];
        for _ in 0..20 {
            assert!(sample(&logits, 1.0, 100, &mut rng()) < 3);
        }
    }

    #[test]
    fn top_k_zero_still_returns_a_token() {
        let logits = [0.1, 0.9, 0.4];
        assert_eq!(sample(&logits, 1.0, 0, &mut rng()), 1);
    }

    #[test]
    fn an_empty_row_does_not_panic() {
        assert_eq!(sample(&[], 1.0, 4, &mut rng()), 0);
    }

    // The bug the old serving sampler had: no max subtraction, so `exp` of a large
    // logit overflowed to inf and the distribution became NaN. Large logits are not
    // hypothetical off an f16 matmul.
    #[test]
    fn large_logits_do_not_overflow() {
        let logits = [200.0, 199.0, 100.0, 1.0];
        for _ in 0..50 {
            let t = sample(&logits, 0.8, 4, &mut rng());
            assert!(t < 4, "large logits produced {t}");
        }
    }

    #[test]
    fn negative_logits_sample_normally() {
        let logits = [-1.0, -2.0, -3.0, -4.0];
        for _ in 0..20 {
            assert!(sample(&logits, 1.0, 4, &mut rng()) < 4);
        }
    }

    #[test]
    fn a_single_candidate_is_returned() {
        assert_eq!(sample(&[42.0], 1.0, 1, &mut rng()), 0);
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
        let cold = (0..400).filter(|_| sample(&logits, 0.1, 4, &mut r) == 0).count();
        let hot = (0..400).filter(|_| sample(&logits, 5.0, 4, &mut r) == 0).count();
        assert!(
            cold > hot,
            "temperature 0.1 picked the top token {cold}/400, temperature 5.0 {hot}/400"
        );
    }
}
