// Held-out perplexity — the benchmark's primary number, and the one the training
// loop never produced.
//
// Training loss says how well the model fits the text it was shown; it falls even
// while the model is memorizing. Held-out loss says whether it learned English.
// Until now this project had no validation split at all, so there was no way to
// tell those apart.
//
// Reported in **bits per byte**, not per token. Per-token loss is not comparable
// across models with different tokenizers: raising `max_vocab` packs more text
// into each token, which lowers per-token loss without the model improving at all.
// Bytes are the invariant unit — the same passage is the same number of bytes for
// every model — so bits-per-byte is the only form of this number that can be
// charted across runs.

use std::path::Path;

use crate::eval::ngram::ByteNgram;
use crate::eval::score;
use crate::training::transformer::Model;

// Order of the n-gram floor. See `ngram::MAX_ORDER` for why 5 is the ceiling.
const BASELINE_ORDER: usize = 5;

// The fraction of each reference file kept back for scoring. The baseline is fit
// on the other 90% — never on the held-out slice, or the floor would be measured
// on text it had memorized and no model could beat it.
const HELDOUT_FRACTION: f64 = 0.10;

// The reference text, split for evaluation.
pub struct CorpusSplit {
    pub train: String,
    pub heldout: String,
    pub files: usize,
}

// Result of scoring the held-out slice.
pub struct CorpusReport {
    pub bits_per_byte: f64,
    pub bits_per_token: f64,
    pub perplexity: f64,
    pub tokens_scored: usize,
    pub bytes_scored: usize,
    // The order-5 byte n-gram fit on the training slice, scored on the same
    // held-out bytes. A model above this line has not beaten a counting table.
    pub baseline_bits_per_byte: f64,
    // How much of the baseline's cost the model removes, in [0, 1]. Negative
    // values are clamped away — "worse than a lookup table" is one verdict, not a
    // scale — but `bits_per_byte` vs `baseline_bits_per_byte` still shows it.
    pub advantage: f64,
    pub files: usize,
}

// Reads every `.txt` file in `dir` and splits each one by byte offset: the head
// fits the baseline, the tail is held out.
//
// Splitting *within* each file rather than holding whole files back matters when
// the corpus is a handful of long books — a whole-file split would evaluate on a
// different author than it trained on and report a domain shift as a modelling
// failure.
pub fn load(dir: &Path, max_train_bytes: usize, max_heldout_bytes: usize) -> CorpusSplit {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
        .collect();
    // Sorted so the split is identical on every run and across machines —
    // directory iteration order is not.
    paths.sort();

    let mut train = String::new();
    let mut heldout = String::new();
    let mut files = 0;

    for path in paths {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        files += 1;

        let cut = char_boundary(&text, (text.len() as f64 * (1.0 - HELDOUT_FRACTION)) as usize);
        if train.len() < max_train_bytes {
            train.push_str(&text[..cut]);
        }
        if heldout.len() < max_heldout_bytes {
            heldout.push_str(&text[cut..]);
        }
        if train.len() >= max_train_bytes && heldout.len() >= max_heldout_bytes {
            break;
        }
    }

    train.truncate(char_boundary(&train, max_train_bytes.min(train.len())));
    heldout.truncate(char_boundary(&heldout, max_heldout_bytes.min(heldout.len())));

    CorpusSplit { train, heldout, files }
}

// Largest char boundary at or below `index`, so truncating a UTF-8 string can
// never panic on a multi-byte character.
fn char_boundary(text: &str, index: usize) -> usize {
    let mut i = index.min(text.len());
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

// Scores the held-out slice under the model and under the n-gram floor.
pub async fn evaluate(
    model: &Model,
    merges: &[((u32, u32), u32)],
    split: &CorpusSplit,
) -> Option<CorpusReport> {
    if split.heldout.trim().is_empty() {
        return None;
    }

    let tokens = tokenize::bpe::encode(&split.heldout, merges);
    if tokens.len() < 2 {
        return None;
    }

    let scores = score::token_logprobs(model, &tokens, model.layers as usize).await;
    if scores.count() == 0 {
        return None;
    }

    // Natural log → bits.
    let total_bits = -scores.total() / std::f64::consts::LN_2;
    let bits_per_token = total_bits / scores.count() as f64;

    // The first token is context, not a prediction, so its bytes were never
    // scored — subtract them or the per-byte figure is optimistic by that much.
    let unscored = tokenize::bpe::decode(&tokens[..1], merges).len();
    let bytes_scored = split.heldout.len().saturating_sub(unscored).max(1);
    let bits_per_byte = total_bits / bytes_scored as f64;

    let baseline = ByteNgram::fit(split.train.as_bytes(), BASELINE_ORDER)
        .bits_per_byte(split.heldout.as_bytes());

    // With no training slice the baseline degenerates to the uniform byte
    // distribution, which any model beats trivially — report the comparison as
    // unavailable rather than as a flattering advantage.
    let advantage = if baseline.is_finite() && baseline > 0.0 && !split.train.is_empty() {
        (1.0 - bits_per_byte / baseline).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };

    Some(CorpusReport {
        bits_per_byte,
        bits_per_token,
        // Per-token perplexity: the effective number of tokens the model is
        // choosing between at each step.
        perplexity: (bits_per_token * std::f64::consts::LN_2).exp(),
        tokens_scored: scores.count(),
        bytes_scored,
        baseline_bits_per_byte: baseline,
        advantage,
        files: split.files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_boundary_never_splits_a_multibyte_character() {
        let text = "aé"; // 'é' is two bytes, so index 2 is mid-character.
        assert_eq!(char_boundary(text, 2), 1);
        assert_eq!(char_boundary(text, 3), 3);
    }

    #[test]
    fn char_boundary_clamps_past_the_end() {
        assert_eq!(char_boundary("abc", 99), 3);
    }

    #[test]
    fn loading_a_missing_directory_yields_an_empty_split() {
        let split = load(Path::new("/nonexistent/eval/corpus"), 1000, 1000);
        assert_eq!(split.files, 0);
        assert!(split.heldout.is_empty());
    }

    #[test]
    fn the_split_holds_back_the_tail_and_keeps_the_head() {
        let dir = std::env::temp_dir().join(format!("eval-corpus-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let text: String = std::iter::repeat("word ").take(200).collect();
        std::fs::write(dir.join("a.txt"), &text).unwrap();

        let split = load(&dir, 1 << 20, 1 << 20);
        assert_eq!(split.files, 1);
        assert!(!split.train.is_empty());
        assert!(!split.heldout.is_empty());
        // Nothing is lost or duplicated across the cut.
        assert_eq!(split.train.len() + split.heldout.len(), text.len());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn non_txt_files_are_ignored() {
        let dir = std::env::temp_dir().join(format!("eval-corpus-skip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.md"), "ignore me").unwrap();

        let split = load(&dir, 1 << 20, 1 << 20);
        assert_eq!(split.files, 0);

        std::fs::remove_dir_all(&dir).ok();
    }
}
