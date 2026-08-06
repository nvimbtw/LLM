//! Automatic thumbs up/down, without a reward model.
//!
//! Preference tuning needs labelled replies, and hand-clicking thumbs produces a
//! few hundred at best. This closes that gap: generate several candidate replies to
//! the same prompt, score them with rules, and label the best one up and the worst
//! one down. The result is a KTO dataset of arbitrary size, produced in one GPU job.
//!
//! **Nothing here is a neural network.** The three signals are:
//!
//!   1. **N-gram fluency** — the order-5 byte model in `eval::ngram`, fit on the
//!      reference corpus. It is a real language model, just not a learned one, and
//!      it is by far the strongest signal available: gibberish scores badly under
//!      real English statistics no matter how it is dressed up.
//!   2. **Surface properties** — real-word rate against the corpus lexicon, and
//!      distinct-3 for repetition. Both from `eval::surface`.
//!   3. **Format** — did the model close its turn, and are its markers balanced.
//!      Exactly decidable, no scoring needed.
//!
//! **Two ways to turn scores into labels, and the second is the one in use.**
//! `label` ranks a prompt's candidates against each other, which needs no
//! calibration but yields only two examples per prompt and can never say "all of
//! these are bad". `label_absolute` compares each candidate to a threshold fitted
//! to replies a person rated — which yields a verdict per candidate, and *can* say
//! that. The threshold is fitted once and then frozen (see `Calibration`): a bar
//! refitted per round would move with the model, and a target that moves with the
//! thing chasing it can never be cleared.
//!
//! **What this can and cannot teach.** It can only teach what it measures: real
//! words, no looping, well-formed turns that end. Those happen to be the actual
//! failure modes of a model this size, which is why it is worth doing — but it will
//! not make the model more helpful or more truthful, and it should not be mistaken
//! for alignment.
//!
//! **On reward hacking.** A model tuned against a fixed rule set will learn to
//! satisfy the rules. Rewarding variety alone would breed varied nonsense; the
//! n-gram term is what pushes back, since nonsense is exactly what it scores down.
//! The real guard is external: auto-labels are written only to the *training*
//! directory, never to the held-out set, so the benchmark's preference margin is
//! computed purely from human thumbs. If the tune games the heuristic, the human
//! margin will not move — and that divergence is the signal to look at.

use serde::{Deserialize, Serialize};

use crate::eval::{ngram, surface};

/// How many bits per byte a uniform random byte stream costs. The "no information
/// at all" anchor the fluency score is normalized against.
const UNIFORM_BITS_PER_BYTE: f64 = 8.0;

/// Component weights. Fluency still leads: it is the one signal a reply cannot
/// satisfy without actually resembling English.
///
/// Format is weighted twice what it was. It used to be half dead — `marker_validity`
/// scanned for characters that never appear in decoded text, so it returned 1.0 for
/// every candidate ever judged, and half of this component was a constant. Now that
/// it reads real markup, "well-formed markers, turn properly closed" is the one
/// judgement here that is *exactly decidable* and *domain-correct* for a chat model,
/// rather than inferred from statistics of a prose corpus. The weight comes off
/// fluency and real-words, which are the two most sensitive to the reference corpus
/// being a different genre from the replies being judged.
const W_FLUENCY: f64 = 0.35;
const W_REAL_WORDS: f64 = 0.20;
const W_VARIETY: f64 = 0.15;
const W_FORMAT: f64 = 0.30;

/// The minimum best-worst score gap for a prompt's candidates to be labelled.
///
/// When every candidate scores about the same, the heuristic has no opinion, and
/// labelling one "good" and another "bad" would inject pure noise into the dataset
/// — teaching the model to prefer one arbitrary reply over an equally good one.
/// Skipping the prompt is strictly better than guessing.
pub const MIN_SCORE_MARGIN: f64 = 0.05;

/// The rule set, built once per run from the reference corpus.
pub struct Judge {
    lexicon: surface::Lexicon,
    ngram: ngram::ByteNgram,
    // The n-gram model's bits/byte on the reference text it was fit on — the "as
    // good as real English gets" anchor. Normalizing against the model's own fit
    // rather than a hard-coded constant is what keeps the fluency score meaningful
    // across different reference corpora.
    reference_bits: f64,
}

/// One reply's score and the parts it came from, kept so a run can be inspected
/// when the labels look wrong.
#[derive(Clone, Copy, Serialize)]
pub struct Score {
    pub total: f64,
    pub fluency: f64,
    pub real_words: f64,
    pub variety: f64,
    pub format: f64,
}

impl Judge {
    /// Fits the rule set to `reference` — the text this judge treats as "what good
    /// output looks like".
    ///
    /// The caller assembles it (see `autolabel::load_reference` on the backend
    /// side) from the benchmark's prose corpus *plus* the assistant turns of the
    /// curated SFT conversations, markers stripped. Prose alone was the original
    /// choice, on the reasoning that the reference should describe English rather
    /// than this model's diet — but the target output is chat replies, and judging
    /// those against children's-story statistics scored a correct technical answer
    /// below a degenerate loop. The prose still supplies nearly all the volume; the
    /// SFT turns supply the vocabulary and register that prose has no examples of.
    ///
    /// **"Children's stories" is history, not the current corpus.** `data/eval/corpus`
    /// has been FineWeb since the re-anchoring (the TinyStories set is parked beside
    /// it as `corpus-tinystories/` and is read by nothing). Measured today the blend is
    /// 8.0 MB of FineWeb against 587 KB of curated assistant turns — **93% web text** —
    /// so the fluency term still describes a register that is not chat replies, just a
    /// much less alien one than the paragraph above implies. Read it before concluding
    /// anything about why the judge prefers what it prefers.
    pub fn build(reference: &str) -> Judge {
        let bytes = reference.as_bytes();
        let ngram = ngram::ByteNgram::fit(bytes, ngram::MAX_ORDER);
        // Measured on the text it was fit on, so this is the model's best case.
        // A reply cannot realistically beat it, which is exactly what makes it the
        // ceiling to normalize against.
        let reference_bits = ngram.bits_per_byte(bytes);
        Judge {
            lexicon: surface::Lexicon::build(reference),
            ngram,
            reference_bits,
        }
    }

    /// Scores one candidate reply in `[0, 1]`.
    ///
    /// `stopped_on_marker` comes from generation, not from the text: once the
    /// terminal marker is stripped, an unterminated reply is indistinguishable from
    /// a terminated one, and "ran to the token cap" is precisely the runaway this
    /// wants to score down.
    pub fn score(&self, reply: &str, stopped_on_marker: bool) -> Score {
        // An empty reply is not a bad reply, it is a non-reply. Scoring its
        // components would divide by zero in several places and, worse, an empty
        // string trivially maximizes variety and real-word rate (no counterexamples).
        if reply.trim().is_empty() {
            return Score { total: 0.0, fluency: 0.0, real_words: 0.0, variety: 0.0, format: 0.0 };
        }

        let fluency = self.fluency(reply);

        // `analyze` pools across samples; handed one, its rates describe that reply.
        let metrics = surface::analyze(std::slice::from_ref(&reply.to_string()), &self.lexicon);
        let real_words = finite_or_zero(metrics.real_word_rate);
        let variety = finite_or_zero(metrics.distinct_3);

        // Two independent format faults, weighted equally: markers that don't nest,
        // and a turn that never ends.
        //
        // A reply containing no markers at all leaves the first unmeasurable
        // (`marker_validity` is NaN, not 0). Scoring that as a fault would punish
        // plain-prose replies for markup they never attempted, so the weight falls
        // to the half that *is* observable — the same "a missing measurement is not
        // a failed one" rule `eval::composite_score` follows.
        let ended = if stopped_on_marker { 1.0 } else { 0.0 };
        let format = if metrics.marker_validity.is_finite() {
            0.5 * metrics.marker_validity + 0.5 * ended
        } else {
            ended
        };

        let total = W_FLUENCY * fluency
            + W_REAL_WORDS * real_words
            + W_VARIETY * variety
            + W_FORMAT * format;

        Score { total, fluency, real_words, variety, format }
    }

    /// How close the reply's byte statistics come to real English, in `[0, 1]`.
    ///
    /// Linear between two anchors: a uniform byte stream (8 bits/byte, the worst
    /// anything can be) scores 0, and the reference corpus's own bits/byte scores 1.
    /// A reply that somehow beats the reference is clamped rather than rewarded
    /// beyond 1 — scoring *below* the reference's entropy means degenerate
    /// repetition, not exceptional prose, and that is variety's job to catch.
    ///
    /// Measured on the reply with its markers stripped, for the same reason
    /// `surface::analyze` strips them: `<|par|>` reaches here as that literal text,
    /// and charging its bytes to a prose n-gram scores correct markup as
    /// disfluency. Markup is judged by `format`, prose by this — one property per
    /// component.
    fn fluency(&self, reply: &str) -> f64 {
        let prose = surface::strip_markers(reply);
        let bits = self.ngram.bits_per_byte(prose.as_bytes());
        if !bits.is_finite() {
            return 0.0;
        }
        let span = UNIFORM_BITS_PER_BYTE - self.reference_bits;
        if span <= 0.0 {
            // A degenerate reference (empty, or one byte repeated) leaves no scale
            // to measure against; fall back to neutral rather than to a divide.
            return 0.0;
        }
        ((UNIFORM_BITS_PER_BYTE - bits) / span).clamp(0.0, 1.0)
    }
}

/// A generated candidate and what it scored.
pub struct Candidate {
    pub text: String,
    pub score: Score,
}

// ── calibration ─────────────────────────────────────────────────────────────
//
// Everything above ranks candidates against each other, which needs no
// calibration but caps the yield at two examples per prompt (a best and a worst)
// and throws away the four in between. Turning the same scores into *absolute*
// verdicts lifts that cap — but an absolute verdict needs a threshold, and a
// hand-picked one would be exactly the unjustified constant the module doc
// refuses to introduce.
//
// The human ratings are where that threshold comes from. Score the replies a
// person already judged, and two questions get answered at once: does this rule
// set agree with the person at all (if not, nothing downstream is worth
// building), and where does the boundary between their "good" and their "bad"
// actually sit on this scale.

/// Agreement below this and the judge is close to a coin flip on the data it will
/// be labelling, so auto-labels would mostly teach the model the judge's mistakes.
/// 0.5 is chance; 0.65 is a modest bar that a rule set tracking real English
/// clears easily on a model whose failure mode is gibberish.
pub const MIN_AUC: f64 = 0.65;

/// Scores within this distance of the threshold are left unlabelled. The boundary
/// is estimated from a few dozen human judgements, so it is not precise to three
/// decimals; a candidate sitting on it is a candidate the judge has no real
/// opinion about, and the pairwise path skips those for the same reason.
pub const UNCERTAIN_BAND: f64 = 0.03;

/// What the human ratings say about this rule set.
///
/// `Deserialize` because this gets **frozen**: fitted once and written to disk,
/// then reused by every later run. A threshold that were refitted each round would
/// move with the model — and a target that moves with the thing chasing it can
/// never be reached, so the loop could neither converge nor report progress.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Calibration {
    /// P(a random approved reply scores above a random rejected one). 0.5 is
    /// chance, 1.0 is perfect agreement. Rank-based, so it measures *ordering*
    /// only — which is all the threshold below needs, and it is immune to the
    /// score scale drifting when the reference corpus changes.
    pub auc: f64,
    /// The score above which a reply is labelled good. Chosen to maximize
    /// (true positive rate − false positive rate) over the human set: the point
    /// that best separates what the person approved from what they rejected.
    pub threshold: f64,
    /// Mean approved score minus mean rejected score. Not used for any decision —
    /// reported because it says how *far* apart the classes sit, which the AUC,
    /// being rank-based, deliberately cannot.
    pub separation: f64,
    pub approved: usize,
    pub rejected: usize,
}

impl Calibration {
    /// Whether auto-labelling on this calibration is worth doing at all.
    pub fn usable(&self) -> bool {
        self.auc >= MIN_AUC
    }
}

/// Fits the threshold from human-rated replies, already scored by the judge.
///
/// Takes scores rather than texts so it stays a pure function over numbers —
/// the caller owns the model and the judge, and this is the part worth testing.
///
/// `None` when either class is empty: with nothing to separate, every threshold is
/// equally good and the AUC is undefined. That is a refusal, not a zero — a
/// calibration computed from one-sided data would claim a boundary it never saw.
pub fn calibrate(approved: &[f64], rejected: &[f64]) -> Option<Calibration> {
    if approved.is_empty() || rejected.is_empty() {
        return None;
    }

    // AUC as the Mann-Whitney statistic: over every (approved, rejected) pair, the
    // fraction where the approved reply scored higher. Ties count as half, which is
    // the standard convention and the honest one — a tie is the judge declining to
    // order the pair, not getting it right.
    let mut wins = 0.0f64;
    for &a in approved {
        for &r in rejected {
            if a > r {
                wins += 1.0;
            } else if (a - r).abs() < f64::EPSILON {
                wins += 0.5;
            }
        }
    }
    let auc = wins / (approved.len() as f64 * rejected.len() as f64);

    // Youden's J over every midpoint between adjacent observed scores. The set of
    // distinct thresholds that can change a classification is finite and small, so
    // this is an exact search rather than an optimizer.
    let mut all: Vec<f64> = approved.iter().chain(rejected.iter()).copied().collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut candidates: Vec<f64> = Vec::with_capacity(all.len() + 1);
    for pair in all.windows(2) {
        if pair[1] > pair[0] {
            candidates.push((pair[0] + pair[1]) / 2.0);
        }
    }
    // A single distinct score everywhere leaves no midpoint; fall back to it so the
    // threshold is still defined (the AUC will be 0.5 and the caller will refuse).
    if candidates.is_empty() {
        candidates.push(all[0]);
    }

    let mut best_threshold = candidates[0];
    let mut best_j = f64::NEG_INFINITY;
    for &t in &candidates {
        let tpr = approved.iter().filter(|&&s| s > t).count() as f64 / approved.len() as f64;
        let fpr = rejected.iter().filter(|&&s| s > t).count() as f64 / rejected.len() as f64;
        let j = tpr - fpr;
        if j > best_j {
            best_j = j;
            best_threshold = t;
        }
    }

    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    Some(Calibration {
        auc,
        threshold: best_threshold,
        separation: mean(approved) - mean(rejected),
        approved: approved.len(),
        rejected: rejected.len(),
    })
}

/// One auto-labelled reply.
pub struct Verdict {
    pub text: String,
    pub good: bool,
    pub score: f64,
}

/// Labels every candidate against the calibrated threshold, rather than only the
/// best and worst of them.
///
/// This is what makes a catalogue rather than a handful of pairs: six candidates
/// per prompt yield up to six examples instead of two, and — more importantly —
/// a prompt where every candidate is bad contributes six rejections instead of
/// being forced to nominate one of them "best". The pairwise path could never say
/// "all of these are bad", because it only ever ranked them against each other.
///
/// Candidates within `UNCERTAIN_BAND` of the threshold are dropped: the boundary
/// is an estimate from a few dozen judgements, and a reply sitting on it is one
/// the judge has no real opinion about.
pub fn label_absolute(candidates: Vec<Candidate>, calibration: &Calibration) -> Vec<Verdict> {
    candidates
        .into_iter()
        .filter_map(|c| {
            // An empty reply is a non-reply, not a bad one — it renders to no reply
            // tokens and the KTO builder would drop it anyway.
            if c.text.trim().is_empty() {
                return None;
            }
            let delta = c.score.total - calibration.threshold;
            if delta.abs() < UNCERTAIN_BAND {
                return None;
            }
            Some(Verdict {
                text: c.text,
                good: delta > 0.0,
                score: c.score.total,
            })
        })
        .collect()
}

/// The best and worst of a prompt's candidates, when the spread is wide enough to
/// mean anything.
pub struct Labelled {
    pub prompt: String,
    pub best: String,
    pub worst: String,
    pub best_score: f64,
    pub worst_score: f64,
}

/// Picks the labelled pair from one prompt's candidates.
///
/// `None` when there are fewer than two candidates, or when best and worst are too
/// close to separate — see `MIN_SCORE_MARGIN`.
///
/// Returning a *pair* rather than a single verdict is deliberate: two examples from
/// one prompt, one desirable and one undesirable, keep the dataset's two halves
/// balanced by construction, and give the benchmark's paired-accuracy metric
/// something to compare. Labelling only "the good one" would produce exactly the
/// one-sided dataset the KTO run refuses to train on.
pub fn label(prompt: &str, mut candidates: Vec<Candidate>) -> Option<Labelled> {
    if candidates.len() < 2 {
        return None;
    }
    candidates.sort_by(|a, b| {
        b.score
            .total
            .partial_cmp(&a.score.total)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best = candidates.first()?;
    let worst = candidates.last()?;
    if best.score.total - worst.score.total < MIN_SCORE_MARGIN {
        return None;
    }
    // An empty best means every candidate was empty; there is no good reply here.
    if best.text.trim().is_empty() {
        return None;
    }

    Some(Labelled {
        prompt: prompt.to_string(),
        best: best.text.clone(),
        worst: worst.text.clone(),
        best_score: best.score.total,
        worst_score: worst.score.total,
    })
}

/// Strips a trailing end-of-turn marker from a generated reply.
///
/// Generation includes the marker it stopped on, but a stored response must not:
/// `chat_template::build_training_example` appends its own `<|/assistant|>` when it
/// renders the example, so leaving one in would train the model on a doubled marker.
pub fn strip_terminal(text: &str) -> String {
    let mut out = text.trim();
    for marker in ["<|/assistant|>", "<|endoftext|>"] {
        if let Some(stripped) = out.strip_suffix(marker) {
            out = stripped.trim_end();
        }
    }
    out.to_string()
}

fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Enough real English for the lexicon and the n-gram model to have something to
    // say. Short, but the assertions below are all comparative.
    const REFERENCE: &str = "The old man walked down to the river and sat on the bank. \
        She opened the letter and read it slowly. In the morning the town was quiet. \
        There was once a small house at the edge of the wood, and a boy lived there \
        with his mother. He could not remember where he had left the key, so he \
        looked in every room of the house until he found it under a chair. The \
        children ran across the field toward the water, laughing as they went. \
        After dinner they sat by the fire and talked about the summer.";

    fn judge() -> Judge {
        Judge::build(REFERENCE)
    }

    fn score(text: &str) -> f64 {
        judge().score(text, true).total
    }

    #[test]
    fn real_english_beats_gibberish() {
        let english = score("The boy walked down to the river and sat on the bank.");
        let gibberish = score("qx zzvv mmkl ppdd xxqq zzvv mmkl bbnn qqxx zzvv");
        assert!(english > gibberish, "{english} vs {gibberish}");
    }

    #[test]
    fn a_loop_scores_below_varied_text() {
        let varied = score("The boy walked down to the river and sat on the bank.");
        let looping = score("the the the the the the the the the the the the the the");
        assert!(varied > looping, "{varied} vs {looping}");
    }

    #[test]
    fn an_unterminated_reply_scores_below_the_same_text_terminated() {
        let j = judge();
        let text = "The boy walked down to the river and sat on the bank.";
        let ended = j.score(text, true).total;
        let ran_on = j.score(text, false).total;
        assert!(ended > ran_on, "{ended} vs {ran_on}");
    }

    #[test]
    fn an_empty_reply_scores_zero() {
        // Specifically guards the degenerate case where "no words" would otherwise
        // give a perfect real-word rate and perfect variety.
        assert_eq!(score(""), 0.0);
        assert_eq!(score("   \n  "), 0.0);
    }

    #[test]
    fn every_score_stays_in_range() {
        for text in [
            "The boy walked down to the river.",
            "qx zzvv mmkl ppdd",
            "the the the the",
            "\u{0}\u{1}\u{2}",
            "a",
        ] {
            let s = judge().score(text, true);
            for part in [s.total, s.fluency, s.real_words, s.variety, s.format] {
                assert!((0.0..=1.0).contains(&part), "{text:?} gave {part}");
            }
        }
    }

    #[test]
    fn a_degenerate_reference_does_not_divide_by_zero() {
        let j = Judge::build("aaaaaaaaaa");
        assert!(j.score("the boy walked", true).total.is_finite());
    }

    fn candidate(text: &str, total: f64) -> Candidate {
        Candidate {
            text: text.to_string(),
            score: Score { total, fluency: total, real_words: total, variety: total, format: total },
        }
    }

    #[test]
    fn labelling_picks_the_extremes() {
        let out = label(
            "hello",
            vec![
                candidate("middling", 0.5),
                candidate("best", 0.9),
                candidate("worst", 0.1),
            ],
        )
        .expect("a 0.8 spread is well past the margin");
        assert_eq!(out.best, "best");
        assert_eq!(out.worst, "worst");
    }

    #[test]
    fn a_flat_field_is_not_labelled() {
        // The heuristic has no opinion here; labelling would be inventing one.
        assert!(label("hello", vec![candidate("a", 0.50), candidate("b", 0.51)]).is_none());
    }

    #[test]
    fn a_single_candidate_is_not_labelled() {
        assert!(label("hello", vec![candidate("only", 0.9)]).is_none());
    }

    #[test]
    fn an_all_empty_field_is_not_labelled() {
        assert!(label("hello", vec![candidate("", 0.9), candidate("", 0.1)]).is_none());
    }

    // ── calibration ─────────────────────────────────────────────────────────

    #[test]
    fn perfect_agreement_scores_one() {
        let cal = calibrate(&[0.7, 0.8, 0.9], &[0.1, 0.2, 0.3]).unwrap();
        assert_eq!(cal.auc, 1.0);
        // The boundary must land between the two classes, or it separates nothing.
        assert!(cal.threshold > 0.3 && cal.threshold < 0.7, "got {}", cal.threshold);
        assert!(cal.usable());
    }

    #[test]
    fn reversed_agreement_scores_zero() {
        // The judge ranking every rejected reply above every approved one is the
        // failure this guard exists to catch — it must not read as "no opinion".
        let cal = calibrate(&[0.1, 0.2], &[0.8, 0.9]).unwrap();
        assert_eq!(cal.auc, 0.0);
        assert!(!cal.usable());
    }

    #[test]
    fn interleaved_scores_land_near_chance() {
        let cal = calibrate(&[0.4, 0.6], &[0.3, 0.7]).unwrap();
        assert!((cal.auc - 0.5).abs() < 1e-9, "got {}", cal.auc);
        assert!(!cal.usable(), "chance agreement must not be usable");
    }

    #[test]
    fn identical_scores_are_ties_not_wins() {
        // Every pair is a tie: half credit each, so exactly chance.
        let cal = calibrate(&[0.5, 0.5], &[0.5, 0.5]).unwrap();
        assert_eq!(cal.auc, 0.5);
    }

    #[test]
    fn one_sided_data_has_no_calibration() {
        // Not a zero and not a default threshold — there is nothing to separate.
        assert!(calibrate(&[0.5, 0.6], &[]).is_none());
        assert!(calibrate(&[], &[0.1]).is_none());
    }

    #[test]
    fn separation_reports_the_gap_the_auc_hides() {
        // Same perfect ordering, very different margins. The AUC cannot tell these
        // apart, which is exactly why separation is reported alongside it.
        let near = calibrate(&[0.51], &[0.49]).unwrap();
        let far = calibrate(&[0.99], &[0.01]).unwrap();
        assert_eq!(near.auc, far.auc);
        assert!(far.separation > near.separation);
    }

    fn cal(threshold: f64) -> Calibration {
        Calibration { auc: 0.9, threshold, separation: 0.3, approved: 10, rejected: 10 }
    }

    #[test]
    fn absolute_labelling_splits_on_the_threshold() {
        let verdicts = label_absolute(
            vec![candidate("good", 0.8), candidate("bad", 0.2)],
            &cal(0.5),
        );
        assert_eq!(verdicts.len(), 2);
        assert!(verdicts[0].good);
        assert!(!verdicts[1].good);
    }

    #[test]
    fn every_candidate_may_land_on_the_same_side() {
        // The whole point of absolute labelling: a prompt where everything the
        // model produced was bad contributes six rejections, where the pairwise
        // path was obliged to call one of them the best.
        let verdicts = label_absolute(
            vec![candidate("a", 0.1), candidate("b", 0.15), candidate("c", 0.2)],
            &cal(0.5),
        );
        assert_eq!(verdicts.len(), 3);
        assert!(verdicts.iter().all(|v| !v.good));
    }

    #[test]
    fn candidates_on_the_boundary_are_dropped() {
        let verdicts = label_absolute(
            vec![
                candidate("just above", 0.5 + UNCERTAIN_BAND / 2.0),
                candidate("just below", 0.5 - UNCERTAIN_BAND / 2.0),
                candidate("clear", 0.9),
            ],
            &cal(0.5),
        );
        assert_eq!(verdicts.len(), 1, "only the unambiguous one survives");
        assert_eq!(verdicts[0].text, "clear");
    }

    #[test]
    fn an_empty_candidate_is_never_labelled() {
        // It would score 0 and read as a confident rejection, but it renders to no
        // reply tokens and teaches nothing.
        let verdicts = label_absolute(vec![candidate("", 0.0)], &cal(0.5));
        assert!(verdicts.is_empty());
    }

    #[test]
    fn strip_terminal_removes_the_marker_and_nothing_else() {
        assert_eq!(strip_terminal("hello there<|/assistant|>"), "hello there");
        assert_eq!(strip_terminal("hello there"), "hello there");
        assert_eq!(strip_terminal("  spaced  <|endoftext|>"), "spaced");
        // Not a suffix, so it stays — stripping mid-text would corrupt the reply.
        assert_eq!(strip_terminal("a<|/assistant|>b"), "a<|/assistant|>b");
    }
}
