// The markup benchmark: does the model use its own special tokens correctly?
//
// This is the second of the two suites, and it exists because the first one cannot
// see the thing supervised fine-tuning is *for*. Every stage of the language bench
// measures prose — bits per byte on held-out web text, a grammar forced choice,
// surface properties of continuations from plain-prose prompts. A fine-tune that
// successfully taught a model to emit `<|par|>`, close its `<|thinking|>` block and
// end its turn shows up there as, at best, nothing.
//
// Before this, the only marker signal anywhere was `surface::marker_validity`: a
// single stat computed over whichever plain-prose samples happened to contain a
// marker, not scored, and `null` for every base model. It answered "were the markers
// in this handful of samples balanced" — not "does this model know the markup".
//
// Four stages, weighted, with the same redistribution rule the language composite
// uses (a missing measurement is not a failed one):
//
//   1. **Marker pairs** (0.40, `marker_pairs`) — forced choice over token-matched
//      strings. Needs no data at all, so a fresh install still gets a real score, and
//      works on a base model.
//   2. **Forced-position prediction** (0.30) — over held-out conversations rendered
//      through the SFT chat template, at every reply position whose reference next
//      token is a marker: would the model actually have emitted it?
//   3. **Generation structure** (0.20) — generate replies to held-out prompts and
//      check the markup that comes out closes and nests.
//   4. **Restraint** (0.10) — the complement of stage 2. A model that scores well on
//      the first three while spraying `<|bold|>` through ordinary prose is not usable,
//      and nothing else here would notice.
//
// Stages 2-4 need `eval/chat/`, which `jobs::queue` exports from the held-out 15%
// share of `raw/sft/` before a run. Stage 1 does not.

use std::collections::HashSet;
use std::path::Path;

use serde::Serialize;
use tokenize::pretokenize::{NUM_SPECIAL, SPECIAL_TOKENS, special_char_from_id, special_id};

use crate::chat_template::{self, Role, Turn};
use crate::training::infer_session::InferSession;

use super::chat::Conversation;
use super::score;

// The reserved marker ids, as a range. `score::token_predictions` takes this as a
// parameter rather than knowing about markers itself.
pub fn marker_ids() -> std::ops::Range<u32> {
    256..256 + NUM_SPECIAL as u32
}

pub fn is_marker(id: u32) -> bool {
    marker_ids().contains(&id)
}

// Markers a reply must never contain. These are conversation scaffolding — the
// harness emits them, and a model that produces one mid-reply is either impersonating
// the user or ending a turn it should still be inside. Kept as a name list rather
// than an id list so it survives a change to the block layout.
const SCAFFOLD_MARKERS: &[&str] = &[
    "<|user|>",
    "<|/user|>",
    "<|system|>",
    "<|/system|>",
    "<|assistant|>",
];

// ── the report ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ParadigmResult {
    pub name: String,
    pub description: String,
    pub accuracy: f64,
    pub pairs: usize,
}

#[derive(Serialize)]
pub struct PairMetrics {
    // Pooled over every pair in every paradigm. Chance is 0.5.
    pub accuracy: f64,
    pub pairs: usize,
    pub paradigms: Vec<ParadigmResult>,
}

#[derive(Serialize)]
pub struct PositionMetrics {
    // The scored one: at a reply position whose reference next token is a marker,
    // does the model's argmax over the *whole vocabulary* agree? This is the honest
    // "would generation put the marker here" — a model can give a marker a respectable
    // log-probability while confidently predicting something else.
    pub marker_top1: f64,
    // Argmax restricted to the marker block. Separates "picks the right marker" from
    // "knows a marker belongs here"; the two failures want different fixes.
    pub marker_restricted_top1: f64,
    // Mean rank of the correct marker over the whole vocabulary, and its mean
    // log-probability. Reported: they move before `marker_top1` does.
    pub marker_mean_rank: f64,
    pub marker_logprob: f64,
    pub marker_positions: usize,
    // The complement, over reply positions whose reference next token is *not* a
    // marker: how often the model wanted to emit one anyway.
    pub false_marker_rate: f64,
    pub prose_positions: usize,
    pub conversations: usize,
}

#[derive(Serialize)]
pub struct StructureMetrics {
    // Did the reply end its own turn, rather than running to the token budget?
    pub close_rate: f64,
    // Was the emitted marker sequence balanced and properly nested?
    pub balance_rate: f64,
    // Markers per 100 generated tokens, against the same figure over the held-out
    // replies. Reported, not scored: it catches both "emits none" and "floods", and
    // which of those is bad depends on the reference.
    pub marker_density: f64,
    pub reference_marker_density: f64,
    // Fraction of replies that emitted a conversational scaffold marker — `<|user|>`,
    // `<|system|>`, a second `<|assistant|>`. A reply containing one is impersonating
    // the other speaker or reopening a turn it is still inside, and it corrupts the
    // transcript whether or not the markup around it happens to balance. Folded into
    // the structure component, because a reply that does this is not usable however
    // well it nests.
    pub scaffold_leak_rate: f64,
    pub replies: usize,
    pub tokens_generated: usize,
    pub samples: Vec<Sample>,
}

#[derive(Serialize)]
pub struct Sample {
    pub prompt: String,
    pub completion: String,
}

#[derive(Serialize)]
pub struct MarkupMetrics {
    pub pairs: PairMetrics,
    // Absent when no held-out chat set is installed, which is the normal state until
    // one is exported. Stage 1 still scores.
    pub positions: Option<PositionMetrics>,
    pub structure: Option<StructureMetrics>,
}

// ── the token-level balance checker ─────────────────────────────────────────

// Is this id sequence's markup balanced and properly nested?
//
// `surface::markers_are_balanced` answers the same question over decoded *text*, and
// is still the right tool for the language bench's prose samples. This one works over
// ids because that is what the markup bench has: every marker is exactly one reserved
// id, so there is no scanning, no ambiguity about text that merely looks like a
// marker, and no dependence on the decoder round-tripping.
pub fn ids_are_balanced(ids: &[u32]) -> bool {
    let mut stack: Vec<&'static str> = Vec::new();
    for &id in ids {
        let Some(name) = marker_name(id) else {
            continue;
        };
        if let Some(inner) = name.strip_prefix("<|/") {
            let opener = format!("<|{inner}");
            match stack.pop() {
                Some(top) if top == opener => {}
                // Closing something that isn't open, or closing out of order.
                _ => return false,
            }
        } else if has_closer(name) {
            stack.push(name);
        }
    }
    stack.is_empty()
}

pub fn marker_name(id: u32) -> Option<&'static str> {
    special_char_from_id(id).and_then(tokenize::pretokenize::lookup_special)
}

// Does this marker have a `<|/…|>` counterpart? Standalone markers
// (`<|endoftext|>`, `<|divider|>`) constrain nothing.
fn has_closer(marker: &str) -> bool {
    let Some(name) = marker.strip_prefix("<|") else {
        return false;
    };
    let closer = format!("<|/{name}");
    SPECIAL_TOKENS.iter().any(|&(m, _)| m == closer)
}

fn scaffold_ids() -> HashSet<u32> {
    SCAFFOLD_MARKERS
        .iter()
        .filter_map(|name| SPECIAL_TOKENS.iter().find(|&&(m, _)| m == *name))
        .filter_map(|&(_, ch)| special_id(ch))
        .collect()
}

// ── stage 1: marker pairs ───────────────────────────────────────────────────

// One pair costs two full-sequence scores; the model is correct when it assigns the
// well-formed string the higher total log-probability.
//
// The `<|endoftext|>` lead-in is the same fix `eval::lead_in` applies to grammar:
// position 0 is never scored, and several of these paradigms put their contrast
// there (`no_stray_closer` and `thinking_precedes_the_answer` both open on the
// differing marker).
pub fn score_pair(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    pair: &super::marker_pairs::Pair,
) -> bool {
    let good = super::lead_in(&pair.good, merges);
    let bad = super::lead_in(&pair.bad, merges);
    let good_lp = score::sequence_logprob(session, &good);
    let bad_lp = score::sequence_logprob(session, &bad);
    // Strict, so an exact tie counts as wrong — an untrained model is uniform, and
    // scoring ties as successes would push it above 50%.
    good_lp > bad_lp
}

pub fn summarize_pairs(results: &[(String, String, usize, usize)]) -> PairMetrics {
    let mut correct_total = 0usize;
    let mut total = 0usize;
    let paradigms = results
        .iter()
        .map(|(name, description, correct, count)| {
            correct_total += correct;
            total += count;
            ParadigmResult {
                name: name.clone(),
                description: description.clone(),
                accuracy: super::rate(*correct, *count),
                pairs: *count,
            }
        })
        .collect();

    PairMetrics { accuracy: super::rate(correct_total, total), pairs: total, paradigms }
}

// ── stage 2 + 4: forced-position prediction ─────────────────────────────────

// What one conversation contributed, before pooling.
pub struct PositionTally {
    pub marker_positions: usize,
    pub marker_top1: usize,
    pub marker_restricted_top1: usize,
    pub marker_rank_total: f64,
    pub marker_logprob_total: f64,
    pub prose_positions: usize,
    pub false_markers: usize,
}

// Score one held-out conversation's reply positions.
//
// `None` when the conversation contributes nothing scoreable — an unknown role, an
// empty encode, or a scoring pass that dropped a position. Dropped rather than
// counted, so an unscoreable example cannot pull a rate toward zero.
pub fn tally_positions(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    conversation: &Conversation,
) -> Option<PositionTally> {
    let (ids, mask) = render(conversation, merges)?;
    if ids.len() < 2 {
        return None;
    }

    // Only the reply positions — the ones a fine-tune took gradient on. Scoring the
    // scaffold would reward a model for predicting markers the harness supplies.
    let predictions = score::token_predictions(session, &ids, &mask, marker_ids());
    // `token_predictions` drops any non-finite position, which would shift every
    // index after it and silently misalign the mask. Refuse the conversation instead
    // of scoring the wrong tokens — the same guard `chat` and `preference` apply.
    if !predictions.scores.aligned_with(ids.len()) {
        return None;
    }

    let mut tally = PositionTally {
        marker_positions: 0,
        marker_top1: 0,
        marker_restricted_top1: 0,
        marker_rank_total: 0.0,
        marker_logprob_total: 0.0,
        prose_positions: 0,
        false_markers: 0,
    };

    for prediction in &predictions.predictions {
        let target = ids[prediction.position as usize + 1];
        if is_marker(target) {
            tally.marker_positions += 1;
            if prediction.argmax == target {
                tally.marker_top1 += 1;
            }
            if prediction.restricted_argmax == Some(target) {
                tally.marker_restricted_top1 += 1;
            }
            tally.marker_rank_total += prediction.rank as f64;
            tally.marker_logprob_total += prediction.logprob as f64;
        } else {
            tally.prose_positions += 1;
            if is_marker(prediction.argmax) {
                tally.false_markers += 1;
            }
        }
    }

    if tally.marker_positions == 0 && tally.prose_positions == 0 {
        return None;
    }
    Some(tally)
}

// Pooled over positions rather than averaged over conversations: a reply with twenty
// markers is more evidence than one with two, and a per-conversation mean would weight
// them the same.
pub fn summarize_positions(tallies: &[PositionTally]) -> Option<PositionMetrics> {
    if tallies.is_empty() {
        return None;
    }
    let sum = |f: fn(&PositionTally) -> usize| tallies.iter().map(f).sum::<usize>();
    let markers = sum(|t| t.marker_positions);
    let prose = sum(|t| t.prose_positions);
    if markers == 0 && prose == 0 {
        return None;
    }

    let rank_total: f64 = tallies.iter().map(|t| t.marker_rank_total).sum();
    let logprob_total: f64 = tallies.iter().map(|t| t.marker_logprob_total).sum();

    Some(PositionMetrics {
        marker_top1: super::rate(sum(|t| t.marker_top1), markers),
        marker_restricted_top1: super::rate(sum(|t| t.marker_restricted_top1), markers),
        marker_mean_rank: if markers == 0 { f64::NAN } else { rank_total / markers as f64 },
        marker_logprob: if markers == 0 { f64::NAN } else { logprob_total / markers as f64 },
        marker_positions: markers,
        false_marker_rate: super::rate(sum(|t| t.false_markers), prose),
        prose_positions: prose,
        conversations: tallies.len(),
    })
}

// ── stage 3: generation structure ───────────────────────────────────────────

// What one generated reply contributed.
pub struct ReplyTally {
    pub closed: bool,
    pub balanced: bool,
    pub leaked_scaffold: bool,
    pub markers: usize,
    pub tokens: usize,
}

pub fn tally_reply(generated: &[u32], stopped_on_marker: bool) -> ReplyTally {
    let scaffold = scaffold_ids();
    ReplyTally {
        closed: stopped_on_marker,
        balanced: ids_are_balanced(generated),
        // The closing `<|/assistant|>` is how a reply *should* end, so it is not in
        // the scaffold set — only the markers that start someone else's turn are.
        leaked_scaffold: generated.iter().any(|id| scaffold.contains(id)),
        markers: generated.iter().filter(|&&id| is_marker(id)).count(),
        tokens: generated.len(),
    }
}

pub fn summarize_structure(
    replies: &[ReplyTally],
    reference_marker_density: f64,
    samples: Vec<Sample>,
) -> Option<StructureMetrics> {
    if replies.is_empty() {
        return None;
    }
    let tokens: usize = replies.iter().map(|r| r.tokens).sum();
    let markers: usize = replies.iter().map(|r| r.markers).sum();

    Some(StructureMetrics {
        close_rate: super::rate(replies.iter().filter(|r| r.closed).count(), replies.len()),
        balance_rate: super::rate(replies.iter().filter(|r| r.balanced).count(), replies.len()),
        marker_density: if tokens == 0 {
            f64::NAN
        } else {
            100.0 * markers as f64 / tokens as f64
        },
        reference_marker_density,
        scaffold_leak_rate: super::rate(
            replies.iter().filter(|r| r.leaked_scaffold).count(),
            replies.len(),
        ),
        replies: replies.len(),
        tokens_generated: tokens,
        samples,
    })
}

// Markers per 100 tokens over the held-out replies — what the model is being compared
// against. Computed off the same rendering the position stage scores.
pub fn reference_density(conversations: &[Conversation], merges: &[((u32, u32), u32)]) -> f64 {
    let mut markers = 0usize;
    let mut tokens = 0usize;
    for conversation in conversations {
        let Some((ids, mask)) = render(conversation, merges) else {
            continue;
        };
        for (i, &id) in ids.iter().enumerate() {
            // Reply tokens only, matching what the model is asked to generate.
            if i > 0 && mask[i - 1] == 1 {
                tokens += 1;
                if is_marker(id) {
                    markers += 1;
                }
            }
        }
    }
    if tokens == 0 {
        return f64::NAN;
    }
    100.0 * markers as f64 / tokens as f64
}

// The held-out conversation, rendered exactly as `sft_train` would see it.
fn render(conversation: &Conversation, merges: &[((u32, u32), u32)]) -> Option<(Vec<u32>, Vec<u8>)> {
    let turns: Vec<Turn> = conversation
        .turns
        .iter()
        .filter_map(|t| {
            let role = match t.role.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                // An unknown role would shift the mask onto the wrong tokens.
                _ => return None,
            };
            Some(Turn { role, text: t.text.clone() })
        })
        .collect();
    if turns.len() != conversation.turns.len() {
        return None;
    }
    Some(chat_template::build_training_example(
        conversation.system.as_deref(),
        &turns,
        conversation.budget,
        merges,
    ))
}

// The prompt a reply is generated from: the conversation up to its first assistant
// turn, ending on an open `<|assistant|>`. Returns the ids and the user text, the
// latter only so a sample can be shown next to what it produced.
pub fn reply_prompt(
    conversation: &Conversation,
    merges: &[((u32, u32), u32)],
    ctx_window: usize,
    reserve: usize,
) -> Option<(Vec<u32>, String)> {
    let first_user = conversation.turns.iter().find(|t| t.role == "user")?;
    let history = vec![Turn { role: Role::User, text: first_user.text.clone() }];
    let ids = chat_template::build_prompt(
        conversation.system.as_deref(),
        &history,
        conversation.budget,
        merges,
        ctx_window,
        reserve,
    );
    if ids.is_empty() {
        return None;
    }
    Some((ids, first_user.text.clone()))
}

pub fn load(dir: &Path) -> Vec<Conversation> {
    super::chat::load(dir)
}

// ── scoring ─────────────────────────────────────────────────────────────────

const W_PAIRS: f64 = 0.40;
const W_POSITIONS: f64 = 0.30;
const W_STRUCTURE: f64 = 0.20;
const W_RESTRAINT: f64 = 0.10;

#[derive(Serialize)]
pub struct MarkupBreakdown {
    pub version: u32,
    pub pairs: f64,
    pub positions: f64,
    pub structure: f64,
    pub restraint: f64,
    pub score: f64,
    pub score_stderr: f64,
}

// Bump alongside any change to the pair suite or the weights, for the reason
// `eval::SCORE_VERSION` gives.
pub const MARKUP_SCORE_VERSION: u32 = 1;

pub fn composite_score(metrics: &MarkupMetrics) -> MarkupBreakdown {
    // Rescaled from the 0.5 chance floor, like grammar: a model that guesses scores 0.
    let pairs = if metrics.pairs.accuracy.is_finite() {
        ((metrics.pairs.accuracy - 0.5) / 0.5).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };
    let positions = metrics
        .positions
        .as_ref()
        .map(|p| p.marker_top1)
        .filter(|v| v.is_finite())
        .unwrap_or(f64::NAN);
    // Three equally weighted halves-of-a-whole: a reply that never ends, one whose
    // markup does not parse, and one that impersonates the user are all unusable, and
    // there is no reason to rank those failures against each other.
    let structure = metrics
        .structure
        .as_ref()
        .map(|s| (s.close_rate + s.balance_rate + (1.0 - s.scaffold_leak_rate)) / 3.0)
        .filter(|v| v.is_finite())
        .unwrap_or(f64::NAN);
    let restraint = metrics
        .positions
        .as_ref()
        .map(|p| 1.0 - p.false_marker_rate)
        .filter(|v| v.is_finite())
        .unwrap_or(f64::NAN);

    let components = [
        (W_PAIRS, pairs),
        (W_POSITIONS, positions),
        (W_STRUCTURE, structure),
        (W_RESTRAINT, restraint),
    ];
    let present: f64 = components.iter().filter(|(_, v)| v.is_finite()).map(|(w, _)| w).sum();
    let score = if present <= 0.0 {
        f64::NAN
    } else {
        100.0
            * components
                .iter()
                .filter(|(_, v)| v.is_finite())
                .map(|(w, v)| w * v)
                .sum::<f64>()
            / present
    };

    MarkupBreakdown {
        version: MARKUP_SCORE_VERSION,
        pairs,
        positions,
        structure,
        restraint,
        score,
        score_stderr: markup_stderr(metrics, present),
    }
}

// Binomial standard errors of the three counted rates, propagated through their
// weights and added in quadrature — the same construction `eval::score_stderr` uses,
// and for the same purpose.
fn markup_stderr(metrics: &MarkupMetrics, present_weight: f64) -> f64 {
    if present_weight <= 0.0 {
        return f64::NAN;
    }
    let mut variance = 0.0;
    let mut add = |weight: f64, rate: f64, n: usize, slope: f64| {
        if rate.is_finite() && n > 0 {
            let se = (rate * (1.0 - rate) / n as f64).sqrt();
            let points = 100.0 * (weight / present_weight) * slope * se;
            variance += points * points;
        }
    };

    // Grammar-style rescaling doubles the slope.
    add(W_PAIRS, metrics.pairs.accuracy, metrics.pairs.pairs, 2.0);
    if let Some(p) = &metrics.positions {
        add(W_POSITIONS, p.marker_top1, p.marker_positions, 1.0);
        add(W_RESTRAINT, p.false_marker_rate, p.prose_positions, 1.0);
    }
    if let Some(s) = &metrics.structure {
        let third = 1.0 / 3.0;
        add(W_STRUCTURE, s.close_rate, s.replies, third);
        add(W_STRUCTURE, s.balance_rate, s.replies, third);
        add(W_STRUCTURE, s.scaffold_leak_rate, s.replies, third);
    }

    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id_of(marker: &str) -> u32 {
        let &(_, ch) = SPECIAL_TOKENS.iter().find(|&&(m, _)| m == marker).expect("marker");
        special_id(ch).expect("id")
    }

    // ── ids_are_balanced ──
    //
    // Mirrors `surface`'s text-level balance tests, over ids. Prose is represented by
    // ordinary byte ids, which can never collide with the reserved block.

    #[test]
    fn balanced_markup_passes() {
        assert!(ids_are_balanced(&[id_of("<|par|>"), b'x' as u32, id_of("<|/par|>")]));
    }

    #[test]
    fn an_unclosed_block_fails() {
        assert!(!ids_are_balanced(&[id_of("<|par|>"), b'x' as u32]));
    }

    #[test]
    fn a_stray_closer_fails() {
        assert!(!ids_are_balanced(&[b'x' as u32, id_of("<|/par|>")]));
    }

    #[test]
    fn a_mismatched_closer_fails() {
        assert!(!ids_are_balanced(&[id_of("<|par|>"), id_of("<|/item|>")]));
    }

    #[test]
    fn crossed_blocks_fail() {
        assert!(!ids_are_balanced(&[
            id_of("<|par|>"),
            id_of("<|quote|>"),
            id_of("<|/par|>"),
            id_of("<|/quote|>"),
        ]));
    }

    #[test]
    fn nested_blocks_pass() {
        assert!(ids_are_balanced(&[
            id_of("<|list|>"),
            id_of("<|item|>"),
            b'x' as u32,
            id_of("<|/item|>"),
            id_of("<|/list|>"),
        ]));
    }

    #[test]
    fn standalone_markers_constrain_nothing() {
        assert!(ids_are_balanced(&[id_of("<|endoftext|>"), id_of("<|divider|>")]));
    }

    #[test]
    fn plain_prose_is_balanced() {
        assert!(ids_are_balanced(&[b'h' as u32, b'i' as u32]));
        assert!(ids_are_balanced(&[]));
    }

    #[test]
    fn unallocated_block_slots_are_not_markers() {
        // The padding between NUM_SPECIAL and SPECIAL_BLOCK decodes to nothing and
        // must not be treated as markup.
        assert!(marker_name(256 + NUM_SPECIAL as u32).is_none());
        assert!(is_marker(256));
        assert!(!is_marker(255));
        assert!(!is_marker(256 + NUM_SPECIAL as u32));
    }

    #[test]
    fn the_scaffold_markers_all_resolve() {
        // A typo in SCAFFOLD_MARKERS would silently make the set smaller.
        assert_eq!(scaffold_ids().len(), SCAFFOLD_MARKERS.len());
    }

    // ── position pooling ──

    fn tally(markers: usize, top1: usize, prose: usize, false_markers: usize) -> PositionTally {
        PositionTally {
            marker_positions: markers,
            marker_top1: top1,
            marker_restricted_top1: top1,
            marker_rank_total: markers as f64,
            marker_logprob_total: -(markers as f64),
            prose_positions: prose,
            false_markers,
        }
    }

    #[test]
    fn positions_pool_over_tokens_not_conversations() {
        // One long reply at 90/100 and one short at 0/2. Pooled that is 90/102; a
        // per-conversation mean would report 45%.
        let m = summarize_positions(&[tally(100, 90, 0, 0), tally(2, 0, 0, 0)]).expect("metrics");
        assert_eq!(m.marker_positions, 102);
        assert!((m.marker_top1 - 90.0 / 102.0).abs() < 1e-12, "got {}", m.marker_top1);
        assert_eq!(m.conversations, 2);
    }

    #[test]
    fn the_false_marker_rate_is_over_prose_positions_only() {
        let m = summarize_positions(&[tally(10, 10, 90, 9)]).expect("metrics");
        assert!((m.false_marker_rate - 0.1).abs() < 1e-12, "got {}", m.false_marker_rate);
        assert_eq!(m.prose_positions, 90);
    }

    #[test]
    fn a_reply_with_no_markers_leaves_the_marker_rates_unknown() {
        // Not zero: the model was never asked to emit one, so it did not fail to.
        let m = summarize_positions(&[tally(0, 0, 50, 0)]).expect("metrics");
        assert!(m.marker_top1.is_nan());
        assert!(m.marker_mean_rank.is_nan());
        assert_eq!(m.false_marker_rate, 0.0);
    }

    #[test]
    fn summarizing_no_conversations_is_no_metrics() {
        assert!(summarize_positions(&[]).is_none());
        assert!(summarize_positions(&[tally(0, 0, 0, 0)]).is_none());
    }

    // ── structure ──

    fn reply(closed: bool, balanced: bool, markers: usize, tokens: usize) -> ReplyTally {
        ReplyTally { closed, balanced, leaked_scaffold: false, markers, tokens }
    }

    #[test]
    fn structure_rates_count_replies() {
        let m = summarize_structure(
            &[reply(true, true, 4, 100), reply(false, true, 6, 100)],
            5.0,
            Vec::new(),
        )
        .expect("metrics");
        assert!((m.close_rate - 0.5).abs() < 1e-12);
        assert!((m.balance_rate - 1.0).abs() < 1e-12);
        // 10 markers over 200 tokens.
        assert!((m.marker_density - 5.0).abs() < 1e-12, "got {}", m.marker_density);
    }

    #[test]
    fn a_reply_that_emits_no_markup_is_balanced_but_scores_no_density() {
        let m = summarize_structure(&[reply(true, true, 0, 50)], 5.0, Vec::new()).expect("m");
        assert_eq!(m.marker_density, 0.0);
        assert_eq!(m.balance_rate, 1.0);
        // The comparison is what makes a zero legible as a failure.
        assert_eq!(m.reference_marker_density, 5.0);
    }

    #[test]
    fn tally_reply_reads_the_generated_ids() {
        let ids = [id_of("<|par|>"), b'x' as u32, id_of("<|/par|>")];
        let t = tally_reply(&ids, true);
        assert!(t.closed);
        assert!(t.balanced);
        assert_eq!(t.markers, 2);
        assert_eq!(t.tokens, 3);

        let unbalanced = tally_reply(&[id_of("<|par|>"), b'x' as u32], false);
        assert!(!unbalanced.closed);
        assert!(!unbalanced.balanced);
    }

    // ── the composite ──

    fn metrics(
        pair_accuracy: f64,
        positions: Option<PositionMetrics>,
        structure: Option<StructureMetrics>,
    ) -> MarkupMetrics {
        MarkupMetrics {
            pairs: PairMetrics {
                accuracy: pair_accuracy,
                pairs: 1000,
                paradigms: Vec::new(),
            },
            positions,
            structure,
        }
    }

    fn positions_at(top1: f64, false_rate: f64) -> Option<PositionMetrics> {
        Some(PositionMetrics {
            marker_top1: top1,
            marker_restricted_top1: top1,
            marker_mean_rank: 2.0,
            marker_logprob: -1.0,
            marker_positions: 500,
            false_marker_rate: false_rate,
            prose_positions: 5000,
            conversations: 50,
        })
    }

    fn structure_at(close: f64, balance: f64) -> Option<StructureMetrics> {
        Some(StructureMetrics {
            close_rate: close,
            balance_rate: balance,
            marker_density: 5.0,
            reference_marker_density: 5.0,
            scaffold_leak_rate: 0.0,
            replies: 50,
            tokens_generated: 5000,
            samples: Vec::new(),
        })
    }

    #[test]
    fn structure_reports_the_scaffold_leak_rate() {
        let mut leaking = reply(true, true, 2, 50);
        leaking.leaked_scaffold = true;
        let m = summarize_structure(&[leaking, reply(true, true, 2, 50)], 5.0, Vec::new())
            .expect("metrics");
        assert!((m.scaffold_leak_rate - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_perfect_model_scores_one_hundred() {
        let b = composite_score(&metrics(1.0, positions_at(1.0, 0.0), structure_at(1.0, 1.0)));
        assert!((b.score - 100.0).abs() < 1e-9, "got {}", b.score);
    }

    #[test]
    fn a_chance_level_model_scores_zero_on_pairs() {
        let b = composite_score(&metrics(0.5, None, None));
        assert_eq!(b.pairs, 0.0);
        assert!(b.score.abs() < 1e-12, "got {}", b.score);
    }

    // The property that lets this suite run on a fresh install: stage 1 needs no data.
    #[test]
    fn a_fresh_install_still_scores_the_pair_suite() {
        let b = composite_score(&metrics(0.75, None, None));
        assert!(b.positions.is_nan());
        assert!(b.structure.is_nan());
        assert!(b.restraint.is_nan());
        // 0.40 of 0.40 present weight, so the pair component *is* the score.
        assert!((b.score - 50.0).abs() < 1e-9, "got {}", b.score);
    }

    #[test]
    fn a_missing_stage_redistributes_rather_than_scoring_zero() {
        let with = composite_score(&metrics(1.0, positions_at(1.0, 0.0), structure_at(1.0, 1.0)));
        let without = composite_score(&metrics(1.0, None, None));
        assert!((with.score - without.score).abs() < 1e-9, "{} vs {}", with.score, without.score);
    }

    // The failure nothing else in the suite notices: well-formed markup, emitted
    // everywhere it does not belong.
    #[test]
    fn spraying_markers_through_prose_costs_the_restraint_component() {
        let restrained = composite_score(&metrics(0.9, positions_at(0.8, 0.0), None));
        let spraying = composite_score(&metrics(0.9, positions_at(0.8, 0.5), None));
        assert!(restrained.score > spraying.score, "{} vs {}", restrained.score, spraying.score);
        assert_eq!(spraying.restraint, 0.5);
    }

    #[test]
    fn a_reply_that_never_ends_costs_the_structure_component() {
        let closes = composite_score(&metrics(0.9, None, structure_at(1.0, 1.0)));
        let runs_on = composite_score(&metrics(0.9, None, structure_at(0.0, 1.0)));
        assert!(closes.score > runs_on.score);
        // Two of the three structural halves still pass.
        assert!((runs_on.structure - 2.0 / 3.0).abs() < 1e-12, "got {}", runs_on.structure);
    }

    #[test]
    fn a_reply_that_impersonates_the_user_costs_the_structure_component() {
        // Well-formed markup that opens someone else's turn is still a broken reply,
        // and balance alone cannot see it.
        let mut leaking = structure_at(1.0, 1.0).unwrap();
        leaking.scaffold_leak_rate = 1.0;
        let clean = composite_score(&metrics(0.9, None, structure_at(1.0, 1.0)));
        let leaked = composite_score(&metrics(0.9, None, Some(leaking)));
        assert!(clean.score > leaked.score, "{} vs {}", clean.score, leaked.score);
        assert!((leaked.structure - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn tally_reply_sees_a_scaffold_marker() {
        let clean = tally_reply(&[id_of("<|par|>"), b'x' as u32, id_of("<|/par|>")], true);
        assert!(!clean.leaked_scaffold);
        // A reply that starts a user turn mid-answer.
        let leaked = tally_reply(&[b'x' as u32, id_of("<|user|>")], true);
        assert!(leaked.leaked_scaffold);
        // The turn's own closer is how a reply ends and must not count as a leak.
        let closed = tally_reply(&[b'x' as u32, id_of("<|/assistant|>")], true);
        assert!(!closed.leaked_scaffold);
    }

    #[test]
    fn an_empty_report_is_not_a_number() {
        let b = composite_score(&metrics(f64::NAN, None, None));
        assert!(b.score.is_nan());
        assert!(b.score_stderr.is_nan());
    }

    #[test]
    fn the_version_is_recorded() {
        assert_eq!(composite_score(&metrics(0.8, None, None)).version, MARKUP_SCORE_VERSION);
    }

    #[test]
    fn score_stderr_shrinks_as_the_sample_grows() {
        let small = MarkupMetrics {
            pairs: PairMetrics { accuracy: 0.8, pairs: 100, paradigms: Vec::new() },
            positions: None,
            structure: None,
        };
        let large = MarkupMetrics {
            pairs: PairMetrics { accuracy: 0.8, pairs: 400, paradigms: Vec::new() },
            positions: None,
            structure: None,
        };
        let ratio = composite_score(&small).score_stderr / composite_score(&large).score_stderr;
        assert!((ratio - 2.0).abs() < 1e-9, "got {ratio}");
    }

    // ── the pair reduction ──

    #[test]
    fn pair_paradigms_pool_into_the_headline() {
        let m = summarize_pairs(&[
            ("a".to_string(), "".to_string(), 45, 60),
            ("b".to_string(), "".to_string(), 30, 60),
        ]);
        assert_eq!(m.pairs, 120);
        assert!((m.accuracy - 0.625).abs() < 1e-12, "got {}", m.accuracy);
        assert_eq!(m.paradigms.len(), 2);
        assert!((m.paradigms[0].accuracy - 0.75).abs() < 1e-12);
    }

    #[test]
    fn an_empty_pair_suite_is_not_a_number() {
        assert!(summarize_pairs(&[]).accuracy.is_nan());
    }
}
