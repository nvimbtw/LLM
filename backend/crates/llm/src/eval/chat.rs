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
use crate::training::transformer::Model;

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
}

#[derive(Deserialize)]
pub struct ConversationTurn {
    pub role: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct ChatMetrics {
    // The headline. Bits the model spends per token of assistant reply — the
    // masked cross-entropy SFT minimizes, in the unit the corpus stage reports.
    pub bits_per_token: f64,
    // Mean per-token log-probability (natural log) over the same positions, for
    // continuity with `preference`'s numbers.
    pub logprob: f64,
    pub conversations: usize,
    pub tokens_scored: usize,
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

/// Total log-probability over one conversation's assistant tokens, and how many
/// there were. The caller accumulates and checks cancellation between calls, in
/// the shape the other stages use.
///
/// `None` when the conversation contributes no scoreable reply position — an empty
/// encode, a conversation with no assistant turn, or a scoring pass that dropped
/// positions. Dropped rather than counted, so an unscoreable example can't pull the
/// mean toward zero.
pub async fn reply_logprob(
    model: &Model,
    merges: &[((u32, u32), u32)],
    conversation: &Conversation,
    layers: usize,
) -> Option<(f64, usize)> {
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
        merges,
    );
    if ids.len() < 2 {
        return None;
    }

    let scores = score::token_logprobs(model, &ids, layers).await;
    // `token_logprobs` drops any non-finite position, which would shift every index
    // after it and silently misalign the mask. Refuse the conversation instead of
    // scoring the wrong tokens — the same guard `preference` applies.
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
    Some((total, counted))
}

/// Reduces scored conversations to the reported metrics. `scored` holds one
/// `(total logprob, token count)` per conversation that produced a score.
///
/// Pooled over tokens rather than averaged over conversations: a long reply is
/// more evidence than a short one, and per-conversation means would weight a
/// three-token answer the same as a three-hundred-token one.
pub fn summarize(scored: &[(f64, usize)]) -> Option<ChatMetrics> {
    let tokens_scored: usize = scored.iter().map(|&(_, n)| n).sum();
    if tokens_scored == 0 {
        return None;
    }
    let total: f64 = scored.iter().map(|&(lp, _)| lp).sum();
    let logprob = total / tokens_scored as f64;

    Some(ChatMetrics {
        // Natural log → bits, and negated so the figure is a cost like the corpus
        // stage's: lower is a better prediction.
        bits_per_token: -logprob / std::f64::consts::LN_2,
        logprob,
        conversations: scored.len(),
        tokens_scored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizing_nothing_is_no_metrics() {
        assert!(summarize(&[]).is_none());
        // A conversation that scored zero tokens carries no evidence either.
        assert!(summarize(&[(0.0, 0)]).is_none());
    }

    #[test]
    fn tokens_are_pooled_rather_than_averaged_per_conversation() {
        // One long conversation at -1.0/token and one short at -3.0/token. Pooled,
        // the long one dominates; a per-conversation mean would report -2.0.
        let metrics = summarize(&[(-90.0, 90), (-30.0, 10)]).expect("scored");
        assert_eq!(metrics.tokens_scored, 100);
        assert_eq!(metrics.conversations, 2);
        assert!((metrics.logprob - -1.2).abs() < 1e-9, "got {}", metrics.logprob);
    }

    #[test]
    fn bits_per_token_is_the_negated_logprob_in_bits() {
        let metrics = summarize(&[(-std::f64::consts::LN_2, 1)]).expect("scored");
        // -ln 2 per token is exactly one bit of surprise.
        assert!((metrics.bits_per_token - 1.0).abs() < 1e-9, "got {}", metrics.bits_per_token);
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
