// Surface metrics over text the model generated: is it made of real words, does
// it stop repeating itself, and is its markup well formed.
//
// These are the numbers that replace "it looks kind of English". They are cheap,
// they need no reference text to compare against, and at this stage they move
// fastest — a model goes from 30% real words to 95% real words long before its
// perplexity looks respectable.
//
// The lexicon is derived from the reference corpus rather than shipped as a
// dictionary, which is worth being precise about: `real_word_rate` therefore
// measures "emits strings that occur as words in real English text", not "emits
// words a dictionary contains". That is the right question here — it separates
// byte soup and invented morphology from actual vocabulary — but it cannot detect
// memorization, and a word the corpus never used counts as unreal.

use std::collections::{HashMap, HashSet};

use tokenize::pretokenize::{lookup_special, SPECIAL_TOKENS};

// How many times a string must appear in the reference corpus to count as a word.
// A threshold of 1 would admit every OCR artifact and typo in the corpus, which
// is exactly the kind of near-miss string a struggling model produces — so the
// metric would score its failures as successes.
const MIN_LEXICON_FREQUENCY: u32 = 3;

// Longest n-gram order used for the repetition measures.
const REPEAT_ORDER: usize = 4;

pub struct Lexicon {
    words: HashSet<String>,
}

impl Lexicon {
    // Builds the lexicon from reference text.
    pub fn build(text: &str) -> Lexicon {
        let mut counts: HashMap<String, u32> = HashMap::new();
        for word in words_in(text) {
            *counts.entry(word).or_insert(0) += 1;
        }
        Lexicon {
            words: counts
                .into_iter()
                .filter(|&(_, n)| n >= MIN_LEXICON_FREQUENCY)
                .map(|(w, _)| w)
                .collect(),
        }
    }

    pub fn contains(&self, word: &str) -> bool {
        self.words.contains(word)
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
}

// What the generation stage reports.
pub struct SurfaceMetrics {
    // Fraction of emitted word-like strings that appear in the reference lexicon.
    pub real_word_rate: f64,
    // Unique n-grams / total n-grams. Falls toward 0 as the model loops.
    pub distinct_1: f64,
    pub distinct_2: f64,
    pub distinct_3: f64,
    // Fraction of 4-grams that occur more than once — the direct "is it stuck"
    // measure, and the complement of distinct-4 in spirit but easier to read.
    pub repeat_rate: f64,
    // Mean words per sentence, over sentences terminated by . ! or ?
    pub mean_sentence_words: f64,
    // Fraction of samples whose special markers are balanced and properly nested.
    pub marker_validity: f64,
    // Word-like strings seen in total, so a rate computed from almost nothing is
    // visible as such rather than reported as a confident 0 or 1.
    pub words_scored: usize,
}

// Analyze a set of generated samples as one pooled body of text (rates), plus
// per-sample for marker validity (a structural property of one output).
pub fn analyze(samples: &[String], lexicon: &Lexicon) -> SurfaceMetrics {
    let pooled: Vec<String> = samples.iter().flat_map(|s| words_in(s)).collect();

    let real = pooled.iter().filter(|w| lexicon.contains(*w)).count();
    let real_word_rate = ratio(real, pooled.len());

    let valid = samples.iter().filter(|s| markers_are_balanced(s)).count();

    SurfaceMetrics {
        real_word_rate,
        distinct_1: distinct_n(&pooled, 1),
        distinct_2: distinct_n(&pooled, 2),
        distinct_3: distinct_n(&pooled, 3),
        repeat_rate: repeat_rate(&pooled, REPEAT_ORDER),
        mean_sentence_words: mean_sentence_words(samples),
        marker_validity: ratio(valid, samples.len()),
        words_scored: pooled.len(),
    }
}

// Lowercased word-like strings: runs of letters, with internal apostrophes kept so
// `don't` is one word rather than two fragments. Marker PUA characters are not
// letters, so they drop out here without special handling.
fn words_in(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphabetic() {
            current.extend(ch.to_lowercase());
        } else if (ch == '\'' || ch == '\u{2019}') && !current.is_empty() {
            // Only mid-word; a leading quote starts no word and a trailing one is
            // trimmed by the flush below.
            current.push('\'');
        } else if !current.is_empty() {
            words.push(current.trim_end_matches('\'').to_string());
            current.clear();
        }
    }
    if !current.is_empty() {
        words.push(current.trim_end_matches('\'').to_string());
    }
    words.retain(|w| !w.is_empty());
    words
}

// Unique n-grams over total n-grams. 1.0 means nothing repeated; low values mean
// the model is cycling.
fn distinct_n(words: &[String], n: usize) -> f64 {
    if words.len() < n || n == 0 {
        return f64::NAN;
    }
    let grams: Vec<&[String]> = words.windows(n).collect();
    let unique: HashSet<Vec<&str>> = grams
        .iter()
        .map(|g| g.iter().map(|s| s.as_str()).collect())
        .collect();
    ratio(unique.len(), grams.len())
}

// Fraction of n-grams that appear more than once anywhere in the pooled text.
fn repeat_rate(words: &[String], n: usize) -> f64 {
    if words.len() < n || n == 0 {
        return f64::NAN;
    }
    let mut counts: HashMap<Vec<&str>, usize> = HashMap::new();
    for gram in words.windows(n) {
        *counts
            .entry(gram.iter().map(|s| s.as_str()).collect())
            .or_insert(0) += 1;
    }
    let total: usize = counts.values().sum();
    let repeated: usize = counts.values().filter(|&&c| c > 1).sum();
    ratio(repeated, total)
}

fn mean_sentence_words(samples: &[String]) -> f64 {
    let mut lengths = Vec::new();
    for sample in samples {
        for sentence in sample.split(['.', '!', '?']) {
            let n = words_in(sentence).len();
            // A trailing empty fragment after the final terminator isn't a
            // sentence; neither is whitespace between two terminators.
            if n > 0 {
                lengths.push(n);
            }
        }
    }
    if lengths.is_empty() {
        return f64::NAN;
    }
    lengths.iter().sum::<usize>() as f64 / lengths.len() as f64
}

// Are the sample's paired markers balanced and properly nested?
//
// Structural correctness is separable from fluency and worth its own number: a
// model can produce good prose while emitting `<|par|>` without ever closing it,
// and the frontend's renderer cares about exactly that.
pub fn markers_are_balanced(text: &str) -> bool {
    let mut stack: Vec<&str> = Vec::new();
    for ch in text.chars() {
        let Some(marker) = lookup_special(ch) else {
            continue;
        };
        // A closer is spelled `<|/name|>`; its opener is the same name without the
        // slash. Markers with no counterpart (`<|endoftext|>`, `<|divider|>`) are
        // standalone and constrain nothing.
        if let Some(name) = marker.strip_prefix("<|/") {
            let opener = format!("<|{name}");
            match stack.pop() {
                Some(top) if top == opener => {}
                // Closing something that isn't open, or closing out of order.
                _ => return false,
            }
        } else if has_closer(marker) {
            stack.push(marker);
        }
    }
    stack.is_empty()
}

// Does this marker have a `<|/…|>` counterpart in the protected vocabulary?
fn has_closer(marker: &str) -> bool {
    let Some(name) = marker.strip_prefix("<|") else {
        return false;
    };
    let closer = format!("<|/{name}");
    SPECIAL_TOKENS.iter().any(|&(m, _)| m == closer)
}

// Guards every rate against a zero denominator: a metric computed from no
// observations is unknown, not zero, and NaN serializes to null for the frontend
// to skip rather than plot as a real reading.
fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        return f64::NAN;
    }
    part as f64 / whole as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexicon() -> Lexicon {
        // Each word repeated to clear MIN_LEXICON_FREQUENCY.
        Lexicon::build("the cat sat on the mat the cat sat on the mat the cat sat on the mat")
    }

    #[test]
    fn lexicon_ignores_words_below_the_frequency_threshold() {
        let lex = Lexicon::build("common common common rare");
        assert!(lex.contains("common"));
        assert!(!lex.contains("rare"));
    }

    #[test]
    fn words_are_lowercased_and_keep_internal_apostrophes() {
        assert_eq!(words_in("Don't STOP, now"), vec!["don't", "stop", "now"]);
    }

    #[test]
    fn words_drop_trailing_apostrophes_and_punctuation() {
        assert_eq!(words_in("'quoted' words."), vec!["quoted", "words"]);
    }

    #[test]
    fn real_word_rate_counts_only_lexicon_hits() {
        let metrics = analyze(&["the cat qxzt".to_string()], &lexicon());
        assert!((metrics.real_word_rate - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(metrics.words_scored, 3);
    }

    #[test]
    fn distinct_n_falls_when_the_text_loops() {
        let varied: Vec<String> = "a b c d e f".split(' ').map(String::from).collect();
        let looped: Vec<String> = "a b a b a b".split(' ').map(String::from).collect();
        assert!(distinct_n(&varied, 2) > distinct_n(&looped, 2));
    }

    #[test]
    fn repeat_rate_is_zero_when_nothing_repeats() {
        let words: Vec<String> = "a b c d e f g h".split(' ').map(String::from).collect();
        assert_eq!(repeat_rate(&words, 4), 0.0);
    }

    #[test]
    fn repeat_rate_is_one_when_everything_repeats() {
        // Three full cycles: every 4-gram window occurs at least twice.
        let words: Vec<String> = "a b c d a b c d a b c d".split(' ').map(String::from).collect();
        assert_eq!(repeat_rate(&words, 4), 1.0);
    }

    #[test]
    fn repeat_rate_counts_only_the_windows_that_recur() {
        // Two cycles gives 5 windows, of which only `a b c d` occurs twice.
        let words: Vec<String> = "a b c d a b c d".split(' ').map(String::from).collect();
        assert!((repeat_rate(&words, 4) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn rates_over_no_observations_are_not_a_number() {
        let metrics = analyze(&[], &lexicon());
        assert!(metrics.real_word_rate.is_nan());
        assert!(metrics.marker_validity.is_nan());
    }

    #[test]
    fn mean_sentence_words_ignores_empty_fragments() {
        let mean = mean_sentence_words(&["One two. Three four five.".to_string()]);
        assert!((mean - 2.5).abs() < 1e-9, "got {mean}");
    }

    // ── marker validity ──

    #[test]
    fn balanced_markers_pass() {
        // <|par|> … <|/par|>
        assert!(markers_are_balanced("\u{E001}hello\u{E025}"));
    }

    #[test]
    fn an_unclosed_marker_fails() {
        assert!(!markers_are_balanced("\u{E001}hello"));
    }

    #[test]
    fn a_stray_closer_fails() {
        assert!(!markers_are_balanced("hello\u{E025}"));
    }

    #[test]
    fn crossed_markers_fail() {
        // <|par|> <|quote|> <|/par|> <|/quote|> — overlapping, not nested.
        assert!(!markers_are_balanced("\u{E001}\u{E01F}\u{E025}\u{E020}"));
    }

    #[test]
    fn nested_markers_pass() {
        // <|par|> <|quote|> … <|/quote|> <|/par|>
        assert!(markers_are_balanced("\u{E001}\u{E01F}x\u{E020}\u{E025}"));
    }

    #[test]
    fn standalone_markers_constrain_nothing() {
        // <|endoftext|> and <|divider|> have no closing form.
        assert!(markers_are_balanced("a\u{E000}b\u{E026}c"));
    }

    #[test]
    fn plain_text_is_trivially_balanced() {
        assert!(markers_are_balanced("just some ordinary prose."));
    }
}
