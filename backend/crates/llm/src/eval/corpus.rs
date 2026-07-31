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

// Cost of a byte under no model at all — a uniform distribution over 256 values.
// The zero point for `floor_progress`: whatever a model knows about text, it is
// worth measuring against having predicted nothing.
const UNMODELLED_BITS_PER_BYTE: f64 = 8.0;

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
    // Where the model sits between predicting nothing (`UNMODELLED_BITS_PER_BYTE`,
    // = 0.0) and matching the n-gram floor (= 1.0), so the composite has a corpus
    // number that still moves below the floor.
    //
    // `advantage` cannot serve that purpose. On a corpus formulaic enough for an
    // order-5 counting table to beat every model at this scale — the TinyStories
    // corpus this used to run against was — it is negative for every checkpoint,
    // clamps to a flat 0.0, and contributes a constant. That is the correct verdict
    // ("has not beaten a lookup table") and a useless coordinate: two models three
    // bits apart score the same.
    //
    // **Neither of them is the composite's corpus term any more**, and the reason is
    // the mirror image. Against the FineWeb slice the eval corpus now holds, every
    // model has *beaten* the floor: `floor_progress` computes to 1.07-1.13 and clamps
    // at the top, pinning a quarter of the score to a constant and hiding a 0.34
    // bits-per-byte improvement. Anchoring a score to the floor failed in both
    // directions, so the composite reads `bits_per_byte` through a fixed logistic
    // instead (see `eval::composite_score`) and these two stay in the report as the
    // diagnostics they always were.
    pub floor_progress: f64,
    pub files: usize,
}

// Reads every `.txt` file in `dir` and splits each one by byte offset: the head
// fits the baseline, the tail is held out.
//
// Splitting *within* each file rather than holding whole files back matters when
// the corpus is a handful of long books — a whole-file split would evaluate on a
// different author than it trained on and report a domain shift as a modelling
// failure.
//
// **Every file contributes its share, rather than the first few filling the budget.**
// This used to walk the sorted list appending whole slices until both caps were met,
// which on the real `eval/corpus` (eight ~1 MB FineWeb files) drew the entire 128 KiB
// held-out slice from the tails of the first *two* — the other six were read for the
// baseline and never scored. That is not a sample of the corpus, it is a sample of two
// documents, and whatever those two happened to be about set the headline number.
// Each file now supplies `max_heldout_bytes / files` from its tail and
// `max_train_bytes / files` from its head, which is the same equal-per-source rule
// `training::plan_bpe_sample` applies for the same reason.
//
// A file shorter than its share contributes all it has and the budget goes unfilled;
// rebalancing would make the split depend on the *sizes* of the other files, and a
// benchmark's held-out slice should not move because an unrelated file grew.
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

    let texts: Vec<String> = paths
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .filter(|t| !t.is_empty())
        .collect();
    if texts.is_empty() {
        return CorpusSplit { train: String::new(), heldout: String::new(), files: 0 };
    }

    let train_share = max_train_bytes / texts.len();
    let heldout_share = max_heldout_bytes / texts.len();

    let mut train = String::new();
    let mut heldout = String::new();
    let mut files = 0;

    for text in &texts {
        let cut = char_boundary(text, (text.len() as f64 * (1.0 - HELDOUT_FRACTION)) as usize);
        let head = &text[..cut];
        let tail = &text[cut..];

        train.push_str(&head[..char_boundary(head, train_share.min(head.len()))]);
        let taken = &tail[..char_boundary(tail, heldout_share.min(tail.len()))];
        heldout.push_str(taken);
        // `files` is what the report shows, and it must mean "files the score was
        // measured on". It used to count files merely *opened*, which read 5 while 2
        // supplied every scored byte.
        if !taken.is_empty() {
            files += 1;
        }
    }

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
    let measurable = baseline.is_finite() && baseline > 0.0 && !split.train.is_empty();
    let advantage = if measurable {
        (1.0 - bits_per_byte / baseline).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };
    // Guarded against a floor at or above the unmodelled cost, which would leave a
    // zero or negative span to place the model on. That means an unusable baseline
    // (an empty or pathological training slice), not a perfect model.
    let span = UNMODELLED_BITS_PER_BYTE - baseline;
    let floor_progress = if measurable && span > 0.0 {
        ((UNMODELLED_BITS_PER_BYTE - bits_per_byte) / span).clamp(0.0, 1.0)
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
        floor_progress,
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

    // A directory of `n` files of `bytes` each, cleaned up by the caller.
    fn corpus_of(tag: &str, n: usize, bytes: usize) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("eval-corpus-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..n {
            // A distinct leading character per file, so a slice can be attributed.
            let ch = (b'a' + i as u8) as char;
            let text: String = std::iter::repeat(ch).take(bytes).collect();
            std::fs::write(dir.join(format!("f{i:02}.txt")), text).unwrap();
        }
        dir
    }

    // The regression this split was rewritten for. Under the old greedy fill the
    // first two files supplied every held-out byte and the rest supplied none.
    #[test]
    fn every_file_contributes_held_out_bytes() {
        let dir = corpus_of("strat", 8, 10_000);
        let split = load(&dir, 40_000, 4_000);

        assert_eq!(split.files, 8, "not every file was scored");
        for i in 0..8u8 {
            let ch = (b'a' + i) as char;
            assert!(
                split.heldout.contains(ch),
                "file {i} contributed nothing to the held-out slice",
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_split_respects_both_caps() {
        let dir = corpus_of("caps", 8, 10_000);
        let split = load(&dir, 40_000, 4_000);
        assert!(split.heldout.len() <= 4_000, "held out {}", split.heldout.len());
        assert!(split.train.len() <= 40_000, "trained on {}", split.train.len());
        std::fs::remove_dir_all(&dir).ok();
    }

    // The train slice must stay disjoint from the scored one, or the n-gram floor is
    // fit on the text it is then measured against.
    #[test]
    fn the_train_and_held_out_slices_do_not_overlap() {
        let dir = std::env::temp_dir().join(format!("eval-corpus-dis-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        // Head is all 'x', tail is all 'y', cut lands at 90%.
        let text = format!("{}{}", "x".repeat(900), "y".repeat(100));
        std::fs::write(dir.join("a.txt"), &text).unwrap();

        let split = load(&dir, 1 << 20, 1 << 20);
        assert!(!split.train.contains('y'), "train slice reached into the held-out tail");
        assert!(!split.heldout.contains('x'), "held-out slice reached into the train head");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_short_file_contributes_what_it_has() {
        let dir = std::env::temp_dir().join(format!("eval-corpus-short-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("big.txt"), "b".repeat(10_000)).unwrap();
        std::fs::write(dir.join("small.txt"), "s".repeat(100)).unwrap();

        // Budget per file far exceeds what `small.txt` can supply.
        let split = load(&dir, 1 << 20, 1 << 20);
        assert_eq!(split.files, 2);
        assert!(split.heldout.contains('s'));
        assert!(split.heldout.contains('b'));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn files_counts_only_what_was_scored() {
        let dir = std::env::temp_dir().join(format!("eval-corpus-count-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "a".repeat(1000)).unwrap();
        // Too short for the 10% tail to survive the char-boundary cut at this budget.
        std::fs::write(dir.join("b.txt"), "b").unwrap();

        let split = load(&dir, 1 << 20, 1 << 20);
        assert!(split.files <= 2);
        assert_eq!(split.files, if split.heldout.contains('b') { 2 } else { 1 });

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
