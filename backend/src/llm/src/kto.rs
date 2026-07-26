//! KTO (Kahneman-Tversky Optimization) — the loss preference training uses here.
//!
//! **Why KTO and not DPO.** DPO consumes triples `(prompt, chosen, rejected)`: two
//! responses to the *same* prompt, ranked against each other. A thumbs-up/down
//! button does not produce that. It produces a binary verdict on one reply, with
//! no counterpart — different prompts, no pairing. KTO is the member of this family
//! built for exactly that shape: unpaired examples, each simply desirable or
//! undesirable. Collecting DPO data from this UI would mean adding a
//! regenerate-and-compare flow first.
//!
//! **The whole trick.** KTO's loss is a function of the sequence log-probability
//! `log π(y|x)` only. So its derivative with respect to the logits is the ordinary
//! cross-entropy gradient times a single scalar — positive for a desirable reply
//! (make it likelier), negative for an undesirable one (make it less likely). That
//! scalar depends on the forward pass, which is already computed before the
//! backward runs, so preference training costs one forward and one backward per
//! example, the same as SFT. No new kernel, no second model resident on the GPU.
//!
//! **The reference model.** The KL anchor to a frozen `π_ref` is what stops this
//! wrecking the model — the failure mode plain SFT has, since SFT has no anchor at
//! all. `π_ref` is the base model the run starts from, and its log-probabilities
//! are precomputed once when the dataset is built (`ref_logprob` per example), so
//! nothing has to hold two sets of weights at once.

use serde::{Deserialize, Serialize};

/// One rated reply in a built KTO dataset.
///
/// `offset`/`len` index the flat token stream in `kto_corpus.bin`, and the same
/// span of `kto_mask.bin` marks which of those tokens are the reply (the SFT
/// masking convention, unchanged). Every example is a single span no longer than
/// the context window, so it trains as exactly one chunk — which is what makes
/// `ref_logprob` comparable, since it was computed over that same single window.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct KtoExample {
    pub offset: usize,
    pub len: usize,
    /// Thumbs up.
    pub desirable: bool,
    /// `log π_ref(reply | prompt)`, summed over the reply's tokens, under the base
    /// model this run fine-tunes from. Precomputed at dataset-build time so the
    /// training loop never has to hold a second set of weights on the GPU.
    pub ref_logprob: f64,
}

/// The sidecar written next to `kto_corpus.bin` / `kto_mask.bin`.
#[derive(Serialize, Deserialize)]
pub struct KtoIndex {
    pub examples: Vec<KtoExample>,
}

impl KtoIndex {
    pub fn desirable(&self) -> usize {
        self.examples.iter().filter(|e| e.desirable).count()
    }

    pub fn undesirable(&self) -> usize {
        self.examples.len() - self.desirable()
    }
}

/// `log π_ref(reply | prompt)` for one rendered example: the sum of
/// `log p(token)` over exactly the masked (reply) positions.
///
/// Computed once per example when the dataset is built, and the *only* place that
/// sum is defined — `train_step_kto` derives the same quantity from the masked
/// cross-entropy the forward already produces, so the two must agree on which
/// positions count. They do because both use the same mask under the same
/// convention: `mask[i] == 1` iff the token at `i + 1` is a reply token, and
/// `logprobs[i]` is `log p(tokens[i+1] | tokens[..=i])`.
///
/// `ids` must fit the model's context window, so this is a single forward pass with
/// no sliding — the same view of the example that training will have. `None` when
/// the example has no scoreable reply position.
///
/// Blocking: the GPU path underneath is async only as an implementation detail, and
/// callers here are already on a dedicated OS thread.
pub fn reference_logprob(
    model: &crate::training::transformer::Model,
    ids: &[u32],
    mask: &[u8],
    layers: usize,
) -> Option<f64> {
    if ids.len() < 2 {
        return None;
    }
    let scores = pollster::block_on(crate::eval::score::token_logprobs(model, ids, layers));
    // `token_logprobs` drops non-finite positions, which would shift every index
    // after the drop and silently misalign the mask. Refuse rather than sum the
    // wrong tokens.
    if scores.logprobs.len() != ids.len() - 1 {
        return None;
    }

    let mut total = 0.0f64;
    let mut counted = 0usize;
    for (i, &logprob) in scores.logprobs.iter().enumerate() {
        if mask[i] == 1 {
            total += logprob as f64;
            counted += 1;
        }
    }
    if counted == 0 {
        return None;
    }
    Some(total)
}

// Everything below works in f64. The quantities are sums of log-probabilities over
// a reply, which at this vocabulary run to a few hundred nats; the sigmoid of their
// difference is where all the precision matters, and doing that in f32 costs
// resolution for no saving worth having.

/// Run knobs. Defaults are the paper's, except where noted.
#[derive(Clone, Copy, Debug)]
pub struct KtoParams {
    /// Strength of the KL anchor. Higher = the tune stays closer to the base model.
    /// The single most important knob: too low and the model drifts into gibberish
    /// that happens to satisfy the labels, too high and nothing moves.
    pub beta: f64,
    /// Weight on desirable examples.
    pub lambda_desirable: f64,
    /// Weight on undesirable examples.
    pub lambda_undesirable: f64,
}

impl Default for KtoParams {
    fn default() -> Self {
        KtoParams {
            beta: 0.1,
            lambda_desirable: 1.0,
            lambda_undesirable: 1.0,
        }
    }
}

impl KtoParams {
    /// Sets the two lambdas so the desirable and undesirable halves carry equal
    /// total weight, given how many of each the dataset actually holds.
    ///
    /// Thumbs data is almost never balanced — people click down on a bad reply far
    /// more readily than up on a fine one, or the reverse, depending on the person.
    /// Left unweighted, the majority class dominates the gradient and the model just
    /// learns "make everything less likely" (or more). The paper recommends keeping
    /// `λ_D·n_D / (λ_U·n_U)` near 1; this puts it exactly at 1.
    ///
    /// The larger class keeps λ = 1 and the smaller is scaled up, rather than
    /// scaling both, so the overall gradient magnitude stays comparable to an
    /// unweighted run and `lr` means the same thing across datasets.
    pub fn balanced(beta: f64, n_desirable: usize, n_undesirable: usize) -> KtoParams {
        let (d, u) = (n_desirable as f64, n_undesirable as f64);
        // A one-sided dataset has nothing to balance against; leave both at 1 and
        // let the caller decide whether such a run is worth doing at all.
        if n_desirable == 0 || n_undesirable == 0 {
            return KtoParams { beta, ..Default::default() };
        }
        if d >= u {
            KtoParams { beta, lambda_desirable: 1.0, lambda_undesirable: d / u }
        } else {
            KtoParams { beta, lambda_desirable: u / d, lambda_undesirable: 1.0 }
        }
    }
}

/// A running estimate of `KL(π ‖ π_ref)`, the reference point KTO measures each
/// example's reward against.
///
/// The paper estimates this from mismatched prompt/response pairs, which costs an
/// extra forward pass per example. This uses the mean of `log π − log π_ref` over
/// the examples already seen, which is free — that difference is computed anyway to
/// get the reward — and tracks the same quantity as the tune drifts. Clamped at
/// zero because a KL divergence cannot be negative, and an early noisy mean can
/// easily go slightly negative before enough examples have accumulated.
///
/// It is deliberately **not** differentiated through: it is a moving baseline, and
/// letting gradient flow into it would let the model lower its loss by dragging the
/// baseline around instead of by improving any individual reply.
pub struct KlEstimate {
    total: f64,
    n: u64,
}

impl KlEstimate {
    pub fn new() -> KlEstimate {
        KlEstimate { total: 0.0, n: 0 }
    }

    pub fn record(&mut self, logprob: f64, ref_logprob: f64) {
        self.total += logprob - ref_logprob;
        self.n += 1;
    }

    pub fn value(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        (self.total / self.n as f64).max(0.0)
    }
}

impl Default for KlEstimate {
    fn default() -> Self {
        KlEstimate::new()
    }
}

/// What one example contributed, for logging and for the run's telemetry.
#[derive(Clone, Copy, Debug)]
pub struct KtoStep {
    /// The KTO loss for this example. In `[0, λ]`; lower is better.
    pub loss: f64,
    /// `log π(y|x)` under the current policy, summed over the reply's tokens.
    /// Carried out so the caller can feed the KL estimate without having to
    /// reconstruct it from `reward` (which would mean dividing by β).
    pub logprob: f64,
    /// The implicit reward `β·(log π − log π_ref)`. Positive means the tune has made
    /// this reply likelier than the base model did.
    pub reward: f64,
    /// The scalar to multiply the mean cross-entropy gradient by. Positive pushes
    /// the reply up, negative pushes it down.
    pub grad_scale: f64,
}

/// The loss, reward, and gradient scale for one example.
///
/// `logprob` and `ref_logprob` are **sums** of `log p(token)` over the reply's
/// tokens — not means. KTO's reward is a property of the whole sequence, and
/// normalizing by length would make the anchor mean something different for a long
/// reply than a short one. (The benchmark's preference stage normalizes, for the
/// opposite and equally deliberate reason: it compares replies of unrelated lengths
/// against each other, where an unnormalized total just ranks by brevity.)
///
/// `active` is the number of reply tokens, and folds into the scale to convert the
/// kernel's *mean* cross-entropy gradient into the *sum* gradient this loss is
/// defined over.
pub fn step(
    logprob: f64,
    ref_logprob: f64,
    desirable: bool,
    kl: f64,
    active: usize,
    params: &KtoParams,
) -> KtoStep {
    let reward = params.beta * (logprob - ref_logprob);

    // A desirable reply should sit above the KL baseline, an undesirable one below,
    // so the two differ only in which way the margin is measured.
    let (margin, lambda) = if desirable {
        (reward - kl, params.lambda_desirable)
    } else {
        (kl - reward, params.lambda_undesirable)
    };

    let sig = sigmoid(margin);
    let loss = lambda * (1.0 - sig);

    // d(loss)/d(margin) = -λ·σ'(margin), and σ' = σ(1-σ).
    //
    // Note this vanishes at both ends: an example the model already has right
    // (σ≈1) and one it has catastrophically wrong (σ≈0) both produce almost no
    // gradient. That saturation is the point — it is what stops a handful of
    // extreme examples from dominating a small dataset.
    let dl_dmargin = -lambda * sig * (1.0 - sig);
    // margin is +reward for desirable, -reward for undesirable.
    let dl_dreward = if desirable { dl_dmargin } else { -dl_dmargin };
    // reward = β·(logπ - ref), so d(loss)/d(logπ) = β·d(loss)/d(reward).
    let dl_dlogprob = params.beta * dl_dreward;

    // The kernel emits d(CE_mean)/d(logits) = (softmax - onehot)/active, and
    // logπ = -CE_sum = -active·CE_mean. So d(logπ)/d(logits) = -active·(kernel), and
    // the scale that turns the kernel's output into d(loss)/d(logits) is
    // -active·d(loss)/d(logπ).
    //
    // Sanity check on the sign: for a desirable reply dl_dlogprob < 0 (more
    // probability is better), so the scale comes out positive — and a positive
    // multiple of the cross-entropy gradient is an ordinary descent step, i.e. the
    // reply gets likelier. For an undesirable one it comes out negative, which
    // ascends instead. That is the entire difference between the two cases.
    let grad_scale = -(active as f64) * dl_dlogprob;

    KtoStep { loss, logprob, reward, grad_scale }
}

fn sigmoid(x: f64) -> f64 {
    // Branch on the sign so the exponential is always of a negative number.
    // exp() of a large positive overflows to inf, and inf/inf is NaN — which would
    // silently poison the gradient scale rather than saturating it as it should.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: KtoParams = KtoParams {
        beta: 0.1,
        lambda_desirable: 1.0,
        lambda_undesirable: 1.0,
    };

    #[test]
    fn a_desirable_reply_is_pushed_up() {
        let s = step(-10.0, -10.0, true, 0.0, 20, &P);
        assert!(s.grad_scale > 0.0, "got {}", s.grad_scale);
    }

    #[test]
    fn an_undesirable_reply_is_pushed_down() {
        let s = step(-10.0, -10.0, false, 0.0, 20, &P);
        assert!(s.grad_scale < 0.0, "got {}", s.grad_scale);
    }

    #[test]
    fn at_the_reference_the_reward_is_zero() {
        // Step 0 of a run: the policy *is* the reference, so nothing has been
        // gained or lost yet and the reward must be exactly 0.
        let s = step(-42.0, -42.0, true, 0.0, 10, &P);
        assert_eq!(s.reward, 0.0);
    }

    #[test]
    fn drifting_above_the_reference_earns_positive_reward() {
        // Less negative logprob = the model finds the reply likelier than the base did.
        let s = step(-8.0, -10.0, true, 0.0, 10, &P);
        assert!(s.reward > 0.0, "got {}", s.reward);
        assert!((s.reward - 0.2).abs() < 1e-12, "got {}", s.reward);
    }

    #[test]
    fn loss_falls_as_a_desirable_reply_gets_likelier() {
        let worse = step(-20.0, -10.0, true, 0.0, 10, &P).loss;
        let better = step(-5.0, -10.0, true, 0.0, 10, &P).loss;
        assert!(better < worse, "{better} vs {worse}");
    }

    #[test]
    fn loss_falls_as_an_undesirable_reply_gets_less_likely() {
        let worse = step(-5.0, -10.0, false, 0.0, 10, &P).loss;
        let better = step(-20.0, -10.0, false, 0.0, 10, &P).loss;
        assert!(better < worse, "{better} vs {worse}");
    }

    #[test]
    fn the_gradient_saturates_at_both_extremes() {
        // Already right, and hopelessly wrong, both produce almost nothing — which
        // is what keeps a couple of outliers from dominating a small dataset.
        let solved = step(500.0, -10.0, true, 0.0, 10, &P).grad_scale.abs();
        let hopeless = step(-5000.0, -10.0, true, 0.0, 10, &P).grad_scale.abs();
        let middling = step(-10.0, -10.0, true, 0.0, 10, &P).grad_scale.abs();
        assert!(solved < middling, "{solved} vs {middling}");
        assert!(hopeless < middling, "{hopeless} vs {middling}");
    }

    #[test]
    fn an_extreme_margin_stays_finite() {
        // The naive sigmoid overflows here; a NaN scale would poison every weight
        // in the model on the next optimizer step.
        for logprob in [1e6, -1e6] {
            let s = step(logprob, 0.0, true, 0.0, 10, &P);
            assert!(s.grad_scale.is_finite(), "{logprob} gave {}", s.grad_scale);
            assert!(s.loss.is_finite());
        }
    }

    #[test]
    fn a_longer_reply_gets_a_proportionally_larger_scale() {
        // The scale converts the kernel's per-token mean gradient into the sum
        // gradient the loss is defined over, so it must track the token count.
        let short = step(-10.0, -10.0, true, 0.0, 10, &P).grad_scale;
        let long = step(-10.0, -10.0, true, 0.0, 20, &P).grad_scale;
        assert!((long - 2.0 * short).abs() < 1e-12, "{long} vs {short}");
    }

    #[test]
    fn balanced_lambdas_equalize_the_two_halves() {
        let p = KtoParams::balanced(0.1, 90, 10);
        assert!((p.lambda_desirable * 90.0 - p.lambda_undesirable * 10.0).abs() < 1e-9);
        // The majority class is the one left at 1.0.
        assert_eq!(p.lambda_desirable, 1.0);
    }

    #[test]
    fn balanced_lambdas_work_in_the_other_direction_too() {
        let p = KtoParams::balanced(0.1, 10, 90);
        assert!((p.lambda_desirable * 10.0 - p.lambda_undesirable * 90.0).abs() < 1e-9);
        assert_eq!(p.lambda_undesirable, 1.0);
    }

    #[test]
    fn a_one_sided_dataset_does_not_divide_by_zero() {
        let p = KtoParams::balanced(0.1, 10, 0);
        assert!(p.lambda_desirable.is_finite() && p.lambda_undesirable.is_finite());
    }

    #[test]
    fn the_kl_estimate_averages_and_never_goes_negative() {
        let mut kl = KlEstimate::new();
        assert_eq!(kl.value(), 0.0, "no observations yet");
        kl.record(-8.0, -10.0); // +2
        kl.record(-9.0, -10.0); // +1
        assert!((kl.value() - 1.5).abs() < 1e-12, "got {}", kl.value());

        let mut negative = KlEstimate::new();
        negative.record(-12.0, -10.0); // -2
        assert_eq!(negative.value(), 0.0, "a KL divergence cannot be negative");
    }
}
