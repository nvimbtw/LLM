// How well does the model predict a *reply*?
//
// Every other stage that enters the composite measures prose: bits per byte on
// held-out books, a grammar forced choice, surface properties of continuations
// from plain-prose prompts. That is deliberate — one suite has to work on a base
// model and an SFT'd one alike — but it leaves supervised fine-tuning with no
// instrument at all. SFT moves the model toward chat-shaped text, so a *successful*
// fine-tune shows up in the prose stages as a regression and nowhere as a gain.
//
// This stage closes that gap. It reads held-out conversations, renders them
// through the same chat template `sft_train` trains on, and scores exactly the
// positions the fine-tune would have taken gradient on — the assistant's reply
// tokens, per `chat_template::build_training_example`'s mask. The number is bits
// per assistant token: lower is better, and an SFT run that worked moves it down
// against its own base model.
//
// `preference` answers a neighbouring but different question — *which* of two
// replies the model prefers, from thumbs collected in the UI. This one needs no
// feedback and no pairing; it asks how surprised the model is by a good reply.
//
// Deliberately **not** folded into the composite, for the reason spelled out in
// `preference`'s header and one more: a base model has never seen the chat
// scaffold, so folding this in would score every pretrain run down for failing at
// a task it was never trained on — the mirror image of the blind spot this exists
// to fix. Read it as a paired comparison between a base and its fine-tune.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::chat_template::{self, Role, Turn};
use crate::training::infer_session::InferSession;

use super::score;

// One held-out conversation, in the same shape `data/raw/sft/*.jsonl` uses so the
// two are interchangeable — hold a slice of a generated set back from training and
// it is directly usable here.
//
// Declared again rather than shared with the backend's `SftConversation`: the llm
// crate is downstream of nothing and must stay runnable from the `eval` binary
// alone, which is the same reason `preference` reads its examples off disk.
#[derive(Deserialize)]
pub struct Conversation {
    pub system: Option<String>,
    pub turns: Vec<ConversationTurn>,
    // The thinking budget the reply was written under, when the generator recorded
    // one. Scoring has to render it, because a budgeted conversation is *trained*
    // with the marker in front of its reply — score it without and the held-out
    // positions are not the positions a fine-tune took gradient on, which is the
    // one thing this file exists to line up.
    #[serde(default)]
    pub budget: Option<chat_template::ThinkBudget>,
}

#[derive(Deserialize)]
pub struct ConversationTurn {
    pub role: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct ChatMetrics {
    // The headline, and what the assistant score is built on: bits the model spends
    // per **byte** of assistant reply.
    //
    // Per byte rather than per token because a token is not a fixed unit. The BPE
    // fix in `17767d0` took the tokenizer from 2.13 to 3.79 bytes per token, and
    // every bits-per-token figure on record moved with it — the assistant score fell
    // from 42.25 to 0.84 on *unchanged weights*, because a 74% larger token costs
    // proportionally more bits to predict. The model had in fact improved slightly:
    // 1.731 to 1.679 bits per byte. A byte is the same byte under any tokenizer, so
    // this number compares across one; `bits_per_token` below does not.
    //
    // The byte count is the length of what the scored tokens *decode to*, which is
    // the reply text as written — identical however the tokenizer segments it. This
    // is the unit the corpus stage has always reported, and for the same reason.
    pub bits_per_byte: f64,
    // One standard error of `bits_per_byte`, with the **conversation** as the
    // sampling unit rather than the byte. `jobs::cycle` accepts or rejects a round on
    // whether this number moved, so it needs to know how far it moves on its own.
    //
    // Bytes within one reply are nothing like independent — a reply that goes wrong
    // goes wrong for fifty tokens together — so a binomial over `bytes_scored` would
    // understate the error several-fold and the gate would read noise as progress.
    // The unit that *is* independent is the conversation, which is also the unit the
    // held-out set is sampled in.
    //
    // `bits_per_byte` is a ratio of two sums (total bits over total bytes), not a
    // mean, so this is the textbook cluster-robust ratio standard error — see
    // `ratio_stderr`. NaN with fewer than two conversations, where a spread is
    // undefined.
    pub bits_per_byte_stderr: f64,
    pub bytes_scored: usize,
    // Bits per reply *token*, reported but no longer scored.
    //
    // Kept because it is the masked cross-entropy SFT actually minimizes, so it is
    // the number to read when asking whether a fine-tune's own objective moved. It
    // is only comparable between models sharing a tokenizer — see `bits_per_byte`.
    pub bits_per_token: f64,
    // Mean per-token log-probability (natural log) over the same positions, for
    // continuity with `preference`'s numbers.
    pub logprob: f64,
    // One standard error of `bits_per_token`, clustered on the conversation exactly
    // as `bits_per_byte_stderr` is.
    pub bits_per_token_stderr: f64,
    pub conversations: usize,
    pub tokens_scored: usize,
}

/// One conversation's contribution: total log-probability over its reply tokens,
/// and the two denominators that turn it into a rate.
///
/// `bytes` is the UTF-8 length of what those tokens decode to. Carried alongside
/// rather than derived later because it is only recoverable while the ids are in
/// hand — by report time there is nothing left to decode.
#[derive(Clone, Copy)]
pub struct ReplyScore {
    pub logprob: f64,
    pub tokens: usize,
    pub bytes: usize,
}

// Reads held-out conversations from a directory of JSONL files, one per line.
//
// A malformed line is skipped rather than failing the run: this directory is
// hand-maintained, and one bad line should cost its own conversation, not the
// whole benchmark.
pub fn load(dir: &Path) -> Vec<Conversation> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        // No held-out chat set installed is the normal case, not an error.
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

    let mut conversations = Vec::new();
    for path in paths {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let Ok(conversation) = serde_json::from_str::<Conversation>(line) else {
                continue;
            };
            if conversation.turns.is_empty() {
                continue;
            }
            conversations.push(conversation);
        }
    }
    conversations
}

/// Total log-probability over one conversation's assistant tokens, with the token
/// and byte counts to divide it by. The caller accumulates and checks cancellation
/// between calls, in the shape the other stages use.
///
/// `decoder` indexes the same `merges`, passed in rather than built here because the
/// free `bpe::decode` rebuilds its table on every call — 292 us for a table that
/// never changes, which over a few hundred conversations is the whole stage.
///
/// `None` when the conversation contributes no scoreable reply position — an empty
/// encode, a conversation with no assistant turn, or a scoring pass that dropped
/// positions. Dropped rather than counted, so an unscoreable example can't pull the
/// mean toward zero.
pub fn reply_logprob(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    decoder: &tokenize::bpe::Decoder,
    conversation: &Conversation,
    reasoning: chat_template::Reasoning,
) -> Option<ReplyScore> {
    let turns: Vec<Turn> = conversation
        .turns
        .iter()
        .filter_map(|t| {
            let role = match t.role.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                // An unknown role would silently shift the mask onto the wrong
                // tokens, so the conversation is refused below rather than guessed.
                _ => return None,
            };
            Some(Turn { role, text: t.text.clone() })
        })
        .collect();
    if turns.len() != conversation.turns.len() {
        return None;
    }

    // The same rendering SFT trains on, so the positions scored here are exactly
    // the positions a fine-tune would have taken gradient on.
    let (ids, mask) = chat_template::build_training_example(
        conversation.system.as_deref(),
        &turns,
        conversation.budget,
        merges,
        reasoning,
    );
    if ids.len() < 2 {
        return None;
    }

    let scores = score::token_logprobs(session, &ids);
    // `token_logprobs` drops any non-finite position, which would shift every index
    // after it and silently misalign the mask. Refuse the conversation instead of
    // scoring the wrong tokens — the same guard `preference` applies.
    if !scores.aligned_with(ids.len()) {
        return None;
    }

    let mut total = 0.0f64;
    // The ids behind the scored positions, kept so the byte count comes from the
    // same tokens the bits do. `mask[i]` governs the token at `i + 1` — it marks the
    // position that *predicts* a reply token, not the reply token itself.
    let mut scored_ids: Vec<u32> = Vec::new();
    for (i, &logprob) in scores.logprobs.iter().enumerate() {
        // Both accumulate under one guard, so the bits and the bytes can never be
        // taken over different sets of positions. The final position carries no
        // `i + 1` and the mask is 0 there by construction; this refuses it anyway
        // rather than trusting that from here.
        let Some(&id) = ids.get(i + 1) else { continue };
        if mask[i] == 1 {
            total += logprob as f64;
            scored_ids.push(id);
        }
    }

    if scored_ids.is_empty() {
        return None;
    }
    Some(ReplyScore {
        logprob: total,
        tokens: scored_ids.len(),
        bytes: decoder.decode(&scored_ids).len(),
    })
}

/// Reduces scored conversations to the reported metrics.
///
/// Pooled over the denominator rather than averaged over conversations: a long
/// reply is more evidence than a short one, and per-conversation means would weight
/// a three-token answer the same as a three-hundred-token one.
pub fn summarize(scored: &[ReplyScore]) -> Option<ChatMetrics> {
    let tokens_scored: usize = scored.iter().map(|s| s.tokens).sum();
    let bytes_scored: usize = scored.iter().map(|s| s.bytes).sum();
    if tokens_scored == 0 || bytes_scored == 0 {
        return None;
    }
    let total: f64 = scored.iter().map(|s| s.logprob).sum();
    let logprob = total / tokens_scored as f64;
    // Natural log → bits, and negated so both figures are a cost like the corpus
    // stage's: lower is a better prediction.
    let total_bits = -total / std::f64::consts::LN_2;
    let bits_per_token = total_bits / tokens_scored as f64;
    let bits_per_byte = total_bits / bytes_scored as f64;

    let per_token: Vec<(f64, usize)> = scored.iter().map(|s| (bits(s), s.tokens)).collect();
    let per_byte: Vec<(f64, usize)> = scored.iter().map(|s| (bits(s), s.bytes)).collect();

    Some(ChatMetrics {
        bits_per_byte,
        bits_per_byte_stderr: ratio_stderr(&per_byte, bits_per_byte, bytes_scored),
        bytes_scored,
        bits_per_token,
        logprob,
        bits_per_token_stderr: ratio_stderr(&per_token, bits_per_token, tokens_scored),
        conversations: scored.len(),
        tokens_scored,
    })
}

// One conversation's cost in bits. Its log-probability is a natural log and is
// negative; a cost is positive and in bits.
fn bits(score: &ReplyScore) -> f64 {
    -score.logprob / std::f64::consts::LN_2
}

// Cluster-robust standard error of a bits-per-unit ratio, clustering on the
// conversation. `scored` pairs each conversation's bits with its own denominator —
// bytes for the scored rate, tokens for the reported one. See
// `ChatMetrics::bits_per_byte_stderr` for why the conversation is the right unit.
//
// For a ratio `R = Σbits / Σunits`, the delta method gives
// `Var(R) = Σ(bits_i - R·units_i)² / (Σunits)²`, with the usual `n/(n-1)`
// correction for estimating `R` from the same sample.
fn ratio_stderr(scored: &[(f64, usize)], rate: f64, total_units: usize) -> f64 {
    let n = scored.len();
    if n < 2 || total_units == 0 {
        return f64::NAN;
    }
    let sum_squares: f64 = scored
        .iter()
        .map(|&(bits, units)| {
            let residual = bits - rate * units as f64;
            residual * residual
        })
        .sum();
    (sum_squares * n as f64 / (n - 1) as f64).sqrt() / total_units as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // A conversation at `tokens` tokens, `bytes` bytes and a total log-probability of
    // `logprob`. Bytes default to four per token, roughly what the current tokenizer
    // produces, for the tests that are not about the byte rate itself.
    fn scored(logprob: f64, tokens: usize) -> ReplyScore {
        ReplyScore { logprob, tokens, bytes: tokens * 4 }
    }

    // `jobs::cycle` accepts a round when this number falls by more than its own error
    // bar, so an error bar that is silently zero turns every flicker into a decision.
    #[test]
    fn conversations_that_agree_have_no_spread_and_ones_that_disagree_do() {
        // Same rate in every conversation: nothing to be uncertain about.
        let agreeing =
            summarize(&[scored(-10.0, 10), scored(-20.0, 20), scored(-30.0, 30)]).unwrap();
        assert!(
            agreeing.bits_per_byte_stderr.abs() < 1e-12,
            "identical rates should carry no spread, got {}",
            agreeing.bits_per_byte_stderr,
        );

        // Same pooled rate, wildly different per conversation.
        let disagreeing = summarize(&[scored(-1.0, 10), scored(-59.0, 50)]).unwrap();
        assert!(
            disagreeing.bits_per_byte_stderr > 0.0,
            "disagreeing conversations must carry a spread",
        );
    }

    // A spread needs at least two of something. NaN rather than 0.0, because the
    // cycle's fallbacks distinguish "no error bar" from "an error bar of zero" — and
    // reading one conversation as certainty is how a gate starts turning on noise.
    #[test]
    fn a_single_conversation_has_no_error_bar() {
        let one = summarize(&[scored(-10.0, 10)]).unwrap();
        assert!(one.bits_per_byte_stderr.is_nan());
        assert!(one.bits_per_token_stderr.is_nan());
    }

    #[test]
    fn summarizing_nothing_is_no_metrics() {
        assert!(summarize(&[]).is_none());
        // A conversation that scored zero tokens carries no evidence either.
        assert!(summarize(&[scored(0.0, 0)]).is_none());
        // Nor one that decoded to nothing: a zero denominator would make the headline
        // rate infinite rather than absent.
        assert!(summarize(&[ReplyScore { logprob: -10.0, tokens: 5, bytes: 0 }]).is_none());
    }

    #[test]
    fn tokens_are_pooled_rather_than_averaged_per_conversation() {
        // One long conversation at -1.0/token and one short at -3.0/token. Pooled,
        // the long one dominates; a per-conversation mean would report -2.0.
        let metrics = summarize(&[scored(-90.0, 90), scored(-30.0, 10)]).expect("scored");
        assert_eq!(metrics.tokens_scored, 100);
        assert_eq!(metrics.conversations, 2);
        assert!((metrics.logprob - -1.2).abs() < 1e-9, "got {}", metrics.logprob);
    }

    #[test]
    fn bits_per_token_is_the_negated_logprob_in_bits() {
        let metrics =
            summarize(&[ReplyScore { logprob: -std::f64::consts::LN_2, tokens: 1, bytes: 1 }])
                .expect("scored");
        // -ln 2 per token is exactly one bit of surprise.
        assert!((metrics.bits_per_token - 1.0).abs() < 1e-9, "got {}", metrics.bits_per_token);
        assert!((metrics.bits_per_byte - 1.0).abs() < 1e-9, "got {}", metrics.bits_per_byte);
    }

    // The whole point of the unit. The same reply, the same bits, segmented into
    // half as many tokens: bits per token doubles and bits per byte does not move.
    // Version 1 scored the first of these 41 points above the second.
    #[test]
    fn coarser_tokens_move_the_token_rate_and_not_the_byte_rate() {
        let fine = summarize(&[ReplyScore { logprob: -80.0, tokens: 40, bytes: 160 }]).unwrap();
        let coarse = summarize(&[ReplyScore { logprob: -80.0, tokens: 20, bytes: 160 }]).unwrap();

        assert!(
            (coarse.bits_per_token - 2.0 * fine.bits_per_token).abs() < 1e-9,
            "{} should be twice {}",
            coarse.bits_per_token,
            fine.bits_per_token,
        );
        assert!(
            (coarse.bits_per_byte - fine.bits_per_byte).abs() < 1e-12,
            "the byte rate moved with the segmentation: {} vs {}",
            coarse.bits_per_byte,
            fine.bits_per_byte,
        );
    }

    // Both rates are pooled over their own denominator, so a conversation that is
    // byte-heavy relative to its token count pulls them in different directions. The
    // two error bars have to follow their own rate for the same reason.
    #[test]
    fn the_two_rates_use_their_own_denominators() {
        let mixed = summarize(&[
            ReplyScore { logprob: -100.0, tokens: 50, bytes: 100 },
            ReplyScore { logprob: -100.0, tokens: 25, bytes: 300 },
        ])
        .expect("scored");

        assert_eq!(mixed.tokens_scored, 75);
        assert_eq!(mixed.bytes_scored, 400);
        let total_bits = 200.0 / std::f64::consts::LN_2;
        assert!((mixed.bits_per_token - total_bits / 75.0).abs() < 1e-9);
        assert!((mixed.bits_per_byte - total_bits / 400.0).abs() < 1e-9);
        assert!(mixed.bits_per_byte_stderr > 0.0 && mixed.bits_per_token_stderr > 0.0);
    }

    #[test]
    fn loading_a_missing_directory_yields_no_conversations() {
        assert!(load(Path::new("/nonexistent/eval/chat")).is_empty());
    }

    #[test]
    fn a_malformed_line_costs_only_itself() {
        let dir = std::env::temp_dir().join(format!("eval-chat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = r#"{"turns":[{"role":"user","text":"hi"},{"role":"assistant","text":"yo"}]}"#;
        std::fs::write(dir.join("a.jsonl"), format!("{good}\nnot json\n\n{good}\n")).unwrap();

        let conversations = load(&dir);
        assert_eq!(conversations.len(), 2);
        assert_eq!(conversations[0].turns.len(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }
}
