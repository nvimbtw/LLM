// Does the model prefer the replies people approved of?
//
// The other four stages measure English: perplexity on held-out prose, a grammar
// forced choice, surface properties of generated text. None of them can see a
// change in *which* reply the model would rather produce, because none of them
// look at chat-shaped text at all — `DEFAULT_PROMPTS` is deliberately plain prose
// so one suite works on a base model and an SFT'd one alike. So a preference tune
// could work perfectly and move nothing in the composite, or move it slightly
// downward, and there would be no way to tell that apart from a failed run.
//
// This stage is that instrument. It reads the thumbs up/down collected in the
// chat UI, renders each rated reply through the same chat template SFT trains on,
// and asks what log-probability the model assigns to the reply *it was actually
// shown*. Two numbers come out:
//
//   * **margin** — mean per-token logprob of approved replies minus that of
//     rejected ones. Positive means the model already leans the right way. This
//     is the number a preference tune should push up, and it works on the
//     unpaired thumbs data we actually have.
//   * **paired accuracy** — restricted to prompts that have *both* an approved
//     and a rejected reply, the fraction where the approved one scores higher.
//     Chance is exactly 50%, like `minimal_pairs`, so it's directly readable.
//     Usually over far fewer examples, hence reported alongside rather than
//     instead of the margin.
//
// Deliberately **not** folded into the composite score. The composite is compared
// across every run ever recorded, and a component that only exists once feedback
// has been collected would make old and new runs incomparable — worse, a model
// would appear to improve merely because someone clicked some thumbs.

use std::path::Path;

use serde::Serialize;

use crate::chat_template::{self, Role, Turn};
use crate::training::infer_session::InferSession;

use super::score;

// One rated reply: the user turn, the model's answer, and the verdict.
pub struct Example {
    pub prompt: String,
    pub response: String,
    // true = thumbs up.
    pub good: bool,
}

#[derive(Serialize)]
pub struct PreferenceMetrics {
    // Mean per-token logprob over approved / rejected replies. Negative numbers
    // (they're log probabilities); closer to zero is more likely.
    pub approved_logprob: f64,
    pub rejected_logprob: f64,
    // approved − rejected. The headline: > 0 means the model already assigns
    // approved replies the higher likelihood.
    pub margin: f64,
    pub approved: usize,
    pub rejected: usize,
    // Forced choice over prompts that have one of each. Chance = 0.5.
    pub paired_accuracy: f64,
    pub pairs: usize,
}

// Reads rated replies from a directory of JSONL files, one example per line:
// `{"prompt": ..., "response": ..., "label": "up" | "down"}`.
//
// A directory of files rather than one fixed path, and read from disk rather than
// straight out of Postgres, for the same reason `corpus/` and `blimp/` are: the
// llm crate has no database awareness and must stay runnable from the `eval`
// binary alone. The backend exports the table into this directory before a
// benchmark job starts.
pub fn load(dir: &Path) -> Vec<Example> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        // No feedback collected yet is the normal case, not an error.
        return Vec::new();
    };

    let mut paths: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .collect();
    // Sorted so a run over the same files is byte-for-byte reproducible; read_dir
    // order is whatever the filesystem feels like.
    paths.sort();

    let mut examples = Vec::new();
    for path in paths {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in text.lines() {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let prompt = value.get("prompt").and_then(|v| v.as_str());
            let response = value.get("response").and_then(|v| v.as_str());
            let label = value.get("label").and_then(|v| v.as_str());
            let (Some(prompt), Some(response), Some(label)) = (prompt, response, label) else {
                continue;
            };
            let good = match label {
                "up" => true,
                "down" => false,
                _ => continue,
            };
            if prompt.trim().is_empty() || response.trim().is_empty() {
                continue;
            }
            examples.push(Example {
                prompt: prompt.to_string(),
                response: response.to_string(),
                good,
            });
        }
    }
    examples
}

// Mean log-probability the model assigns to each token of `response`, given
// `prompt`, under the project's chat template.
//
// **Length-normalized, unlike every other score in this module.** A minimal pair is
// word-count matched precisely so an unnormalized total is comparable; rated replies
// are whatever length the model happened to produce, and total logprob falls
// monotonically with length, so comparing totals would rank short replies above long
// ones no matter what they said. Dividing by the reply's token count asks the
// question that actually matters: per token, how likely was this text.
//
// `None` when the reply contributes no scoreable position (an empty encode, or a
// scoring run that dropped positions) — the caller skips it rather than folding a
// NaN into the mean.
fn mean_reply_logprob(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    example: &Example,
) -> Option<f64> {
    // The same rendering SFT trains on, so the positions scored here are exactly
    // the positions a fine-tune would have taken gradient on. `mask[i] == 1` iff the
    // token at `i + 1` belongs to the assistant's reply.
    //
    // The response is passed through as stored, thinking block and all: the thumbs
    // was given on the whole reply as the user saw it, so scoring a silently
    // stripped version would measure something the user never rated.
    let turns = [
        Turn { role: Role::User, text: example.prompt.clone() },
        Turn { role: Role::Assistant, text: example.response.clone() },
    ];
    // No budget, and nothing to strip: a preference example is a bare
    // prompt/response pair generated unbudgeted (see `labeling::autolabel`), so it
    // carries no thinking block either way and `Keep` is the identity here.
    let (ids, mask) = chat_template::build_training_example(
        None,
        &turns,
        None,
        merges,
        chat_template::Reasoning::Keep,
    );
    if ids.len() < 2 {
        return None;
    }

    let scores = score::token_logprobs(session, &ids);
    // `token_logprobs` drops any non-finite position, which would shift every
    // index after it and silently misalign the mask. Refuse the example instead of
    // scoring the wrong tokens.
    if !scores.aligned_with(ids.len()) {
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
    Some(total / counted as f64)
}

// Scores every example and reduces them to the metrics above.
//
// `cancel` is checked by the caller between examples rather than here, so this
// stays a plain fold over already-scored values.
pub fn summarize(scored: &[(bool, String, f64)]) -> PreferenceMetrics {
    let mut approved_total = 0.0;
    let mut approved = 0usize;
    let mut rejected_total = 0.0;
    let mut rejected = 0usize;

    for (good, _, logprob) in scored {
        if *good {
            approved_total += logprob;
            approved += 1;
        } else {
            rejected_total += logprob;
            rejected += 1;
        }
    }

    let approved_logprob = mean(approved_total, approved);
    let rejected_logprob = mean(rejected_total, rejected);

    let (correct, pairs) = paired_accuracy(scored);

    PreferenceMetrics {
        approved_logprob,
        rejected_logprob,
        // NaN if either side is empty, which is honest: with no rejected replies
        // there is no margin, and reporting the approved mean as one would read as
        // a huge positive result.
        margin: approved_logprob - rejected_logprob,
        approved,
        rejected,
        paired_accuracy: if pairs == 0 {
            f64::NAN
        } else {
            correct as f64 / pairs as f64
        },
        pairs,
    }
}

// The forced-choice half: for every prompt that drew both an approved and a
// rejected reply, does the approved one score higher?
//
// Every approved reply is compared against every rejected reply for the same
// prompt, rather than picking one of each — a prompt with three thumbs-down and one
// thumbs-up carries three independent judgements, and dropping two of them would
// throw away real signal from a dataset that is small to begin with.
//
// Prompts are matched on their trimmed text. Exact matching is the right call here
// because the prompts being compared are literally the same stored string re-sent,
// not paraphrases.
fn paired_accuracy(scored: &[(bool, String, f64)]) -> (usize, usize) {
    let mut correct = 0usize;
    let mut pairs = 0usize;

    for (good, prompt, logprob) in scored {
        if !*good {
            continue;
        }
        for (other_good, other_prompt, other_logprob) in scored {
            if *other_good || other_prompt.trim() != prompt.trim() {
                continue;
            }
            pairs += 1;
            // Strict `>`, matching `minimal_pairs`: an exact tie is a model with no
            // opinion, and counting it as a win would push a flat model above chance.
            if logprob > other_logprob {
                correct += 1;
            }
        }
    }

    (correct, pairs)
}

fn mean(total: f64, n: usize) -> f64 {
    if n == 0 { f64::NAN } else { total / n as f64 }
}

// Scores one example. Split out so the caller can drive progress and cancellation
// between examples without this module knowing about either.
pub fn score_example(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    example: &Example,
) -> Option<f64> {
    mean_reply_logprob(session, merges, example)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(rows: &[(bool, &str, f64)]) -> Vec<(bool, String, f64)> {
        rows.iter()
            .map(|(good, prompt, lp)| (*good, prompt.to_string(), *lp))
            .collect()
    }

    #[test]
    fn margin_is_positive_when_approved_replies_score_higher() {
        let m = summarize(&scored(&[
            (true, "a", -1.0),
            (true, "b", -1.5),
            (false, "a", -3.0),
            (false, "b", -3.5),
        ]));
        assert_eq!(m.approved, 2);
        assert_eq!(m.rejected, 2);
        assert!(m.margin > 0.0, "got {}", m.margin);
    }

    #[test]
    fn margin_is_negative_when_the_model_prefers_the_rejected_replies() {
        let m = summarize(&scored(&[(true, "a", -4.0), (false, "a", -1.0)]));
        assert!(m.margin < 0.0, "got {}", m.margin);
    }

    #[test]
    fn only_shared_prompts_form_pairs() {
        // "b" has an approved reply and "c" a rejected one, but neither prompt has
        // both, so only "a" contributes a judgement.
        let m = summarize(&scored(&[
            (true, "a", -1.0),
            (false, "a", -2.0),
            (true, "b", -1.0),
            (false, "c", -2.0),
        ]));
        assert_eq!(m.pairs, 1);
        assert_eq!(m.paired_accuracy, 1.0);
    }

    #[test]
    fn every_approved_rejected_combination_on_a_prompt_is_judged() {
        // One approved, two rejected on the same prompt => two judgements, one of
        // which the model gets wrong.
        let m = summarize(&scored(&[
            (true, "a", -2.0),
            (false, "a", -3.0),
            (false, "a", -1.0),
        ]));
        assert_eq!(m.pairs, 2);
        assert_eq!(m.paired_accuracy, 0.5);
    }

    #[test]
    fn a_tie_counts_against_the_model() {
        let m = summarize(&scored(&[(true, "a", -1.0), (false, "a", -1.0)]));
        assert_eq!(m.pairs, 1);
        assert_eq!(m.paired_accuracy, 0.0);
    }

    #[test]
    fn one_sided_feedback_yields_no_margin_rather_than_a_flattering_number() {
        let m = summarize(&scored(&[(true, "a", -1.0), (true, "b", -2.0)]));
        assert_eq!(m.approved, 2);
        assert_eq!(m.rejected, 0);
        assert!(m.rejected_logprob.is_nan());
        assert!(m.margin.is_nan());
        assert!(m.paired_accuracy.is_nan());
    }

    #[test]
    fn nothing_scored_is_all_unknown() {
        let m = summarize(&[]);
        assert!(m.approved_logprob.is_nan());
        assert!(m.margin.is_nan());
        assert_eq!(m.pairs, 0);
    }

    #[test]
    fn a_missing_directory_yields_no_examples() {
        assert!(load(Path::new("/nonexistent/preference")).is_empty());
    }
}
