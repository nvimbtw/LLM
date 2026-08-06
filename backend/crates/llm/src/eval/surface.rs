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

use tokenize::pretokenize::SPECIAL_TOKENS;

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
//
// The word-based rates are measured on marker-*stripped* text. `<|par|>` decodes
// to that literal string (see `strip_markers`), so leaving it in would feed `par`
// to `words_in` as a word — and since the reference lexicon is prose, every marker
// the model emits would count against `real_word_rate`. Markup correctness is
// `marker_validity`'s job; vocabulary is this one's, and the two must not be
// measured on the same characters.
pub fn analyze(samples: &[String], lexicon: &Lexicon) -> SurfaceMetrics {
    let prose: Vec<String> = samples.iter().map(|s| strip_markers(s)).collect();
    let pooled: Vec<String> = prose.iter().flat_map(|s| words_in(s)).collect();

    let real = pooled.iter().filter(|w| lexicon.contains(*w)).count();
    let real_word_rate = ratio(real, pooled.len());

    // Only samples that actually contain a marker are evidence about markup. A
    // sample with none is trivially "balanced", and counting it as a success let
    // this metric report a perfect score for models emitting no markup at all —
    // and, worse, for models whose every marker was malformed, diluted by the
    // ones that attempted none.
    let attempted: Vec<&String> = samples.iter().filter(|s| contains_marker(s)).collect();
    let valid = attempted.iter().filter(|s| markers_are_balanced(s)).count();

    SurfaceMetrics {
        real_word_rate,
        distinct_1: distinct_n(&pooled, 1),
        distinct_2: distinct_n(&pooled, 2),
        distinct_3: distinct_n(&pooled, 3),
        repeat_rate: repeat_rate(&pooled, REPEAT_ORDER),
        mean_sentence_words: mean_sentence_words(&prose),
        marker_validity: ratio(valid, attempted.len()),
        words_scored: pooled.len(),
    }
}

// Lowercased word-like strings: runs of letters, with internal apostrophes kept so
// `don't` is one word rather than two fragments.
//
// This is deliberately marker-unaware: markers reach it spelled out, so `<|par|>`
// would yield the word `par`. Callers strip them first (`analyze` does); anything
// else scoring raw generation output through here would be counting markup as
// vocabulary.
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

// Every protected marker in `text`, in order, as (byte offset, marker spelling).
//
// Markers arrive here in their *literal* spelling (`<|par|>`), not as the private-use
// character the tokenizer packs them into: `bpe::decode` runs `replace_special_chars`
// before any of this sees the text. Scanning for PUA characters instead — as this
// module did — matched nothing on real generation output, which silently pinned
// `marker_validity` at 1.0 for every model ever benchmarked.
pub(super) fn markers_in(text: &str) -> Vec<(usize, &'static str)> {
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] != b'<' || bytes[i + 1] != b'|' {
            i += 1;
            continue;
        }
        // No marker is a prefix of another (they all end in `|>`), so the first
        // match is the only match.
        match SPECIAL_TOKENS.iter().find(|(m, _)| text[i..].starts_with(m)) {
            Some(&(marker, _)) => {
                found.push((i, marker));
                i += marker.len();
            }
            // `<|foo|>` is ordinary text that merely looks like a marker. Step past
            // the `<` alone, so a real marker starting inside this run is still seen.
            None => i += 1,
        }
    }
    found
}

// Does the sample attempt any markup at all? Distinguishes "no markers" from
// "markers, all of them well formed" — two very different things that
// `markers_are_balanced` alone reports identically.
pub fn contains_marker(text: &str) -> bool {
    !markers_in(text).is_empty()
}

// The sample with every protected marker replaced by a space, leaving the prose.
// A space rather than nothing so `<|item|>a<|/item|><|item|>b<|/item|>` yields two
// words rather than the single fused `ab`.
pub fn strip_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for (at, marker) in markers_in(text) {
        out.push_str(&text[last..at]);
        out.push(' ');
        last = at + marker.len();
    }
    out.push_str(&text[last..]);
    out
}

// Are the sample's paired markers balanced and properly nested?
//
// Structural correctness is separable from fluency and worth its own number: a
// model can produce good prose while emitting `<|par|>` without ever closing it,
// and the frontend's renderer cares about exactly that.
//
// Text containing no markers is vacuously balanced, so callers that mean "is this
// model's markup any good" must gate on `contains_marker` first — `analyze` does.
pub fn markers_are_balanced(text: &str) -> bool {
    let mut stack: Vec<&str> = Vec::new();
    for (_, marker) in markers_in(text) {
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
    fn marker_names_are_not_counted_as_words() {
        // `<|par|>` decodes to that literal text, so an unstripped scorer reads
        // `par` as a word and — the lexicon being prose — as a *fake* one.
        let metrics = analyze(&["<|par|>the cat<|/par|>".to_string()], &lexicon());
        assert_eq!(metrics.words_scored, 2);
        assert!((metrics.real_word_rate - 1.0).abs() < 1e-9, "got {}", metrics.real_word_rate);
    }

    #[test]
    fn markup_only_output_scores_no_words_rather_than_all_fake_ones() {
        let metrics = analyze(&["<|par|><|/par|>".to_string()], &lexicon());
        assert_eq!(metrics.words_scored, 0);
        assert!(metrics.real_word_rate.is_nan());
    }

    #[test]
    fn stripping_markers_does_not_fuse_adjacent_words() {
        assert_eq!(words_in(&strip_markers("<|item|>the<|/item|><|item|>cat<|/item|>")),
                   vec!["the", "cat"]);
    }

    #[test]
    fn mean_sentence_words_ignores_empty_fragments() {
        let mean = mean_sentence_words(&["One two. Three four five.".to_string()]);
        assert!((mean - 2.5).abs() < 1e-9, "got {mean}");
    }

    // ── marker validity ──

    // These run on markers in their literal spelling, which is how they reach this
    // module: `bpe::decode` expands the PUA characters before the benchmark sees
    // the text. The PUA-shaped tests these replace passed against input production
    // never produces, which is what let the metric read 1.0 for three consecutive
    // fine-tunes whose every sample was malformed.

    #[test]
    fn balanced_markers_pass() {
        assert!(markers_are_balanced("<|par|>hello<|/par|>"));
    }

    #[test]
    fn an_unclosed_marker_fails() {
        assert!(!markers_are_balanced("<|par|>hello"));
    }

    #[test]
    fn a_stray_closer_fails() {
        assert!(!markers_are_balanced("hello<|/par|>"));
    }

    #[test]
    fn crossed_markers_fail() {
        assert!(!markers_are_balanced("<|par|><|quote|><|/par|><|/quote|>"));
    }

    #[test]
    fn nested_markers_pass() {
        assert!(markers_are_balanced("<|par|><|quote|>x<|/quote|><|/par|>"));
    }

    #[test]
    fn standalone_markers_constrain_nothing() {
        // <|endoftext|> and <|divider|> have no closing form.
        assert!(markers_are_balanced("a<|endoftext|>b<|divider|>c"));
    }

    #[test]
    fn text_that_merely_looks_like_a_marker_is_ignored() {
        // Not in the protected vocabulary, so it is prose, not markup.
        assert!(markers_are_balanced("compare a <|foo|> against b"));
        assert!(!contains_marker("compare a <|foo|> against b"));
        // …and a real marker starting inside such a run is still found.
        assert!(contains_marker("<|x<|par|>y<|/par|>"));
    }

    #[test]
    fn plain_text_is_not_evidence_of_valid_markup() {
        // Vacuously balanced, but it attempted nothing — `analyze` must not read
        // that as a success.
        assert!(markers_are_balanced("just some ordinary prose."));
        assert!(!contains_marker("just some ordinary prose."));
    }

    #[test]
    fn marker_validity_ignores_samples_that_attempted_no_markup() {
        let samples = vec![
            "plain prose with no markup at all".to_string(),
            "<|par|>closed properly<|/par|>".to_string(),
            "<|par|>never closed".to_string(),
        ];
        let metrics = analyze(&samples, &lexicon());
        // 1 of the 2 samples that tried, not 2 of 3.
        assert!((metrics.marker_validity - 0.5).abs() < 1e-9, "got {}", metrics.marker_validity);
    }

    #[test]
    fn marker_validity_is_unknown_when_nothing_attempted_markup() {
        let metrics = analyze(&["all prose".to_string()], &lexicon());
        assert!(metrics.marker_validity.is_nan());
    }
}
