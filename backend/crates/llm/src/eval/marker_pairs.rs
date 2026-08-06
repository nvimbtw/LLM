// Minimal pairs for markup, in the shape `minimal_pairs` uses for grammar.
//
// The question this answers is "does the model know how its own markers compose",
// asked the cheapest way there is: a forced choice between two strings that contain
// the *same prose and the same number of markers*, differing only in which markers
// and in what order. Chance is exactly 50%, it needs no generation and no held-out
// data, and — the reason it leads the markup suite rather than trailing it — it works
// on a base model. A pretrain sees the marker vocabulary through the chat share of
// its corpus, so "has this pretrain learned the markup" is a real question with a
// real answer, which bits-per-assistant-token cannot ask.
//
// **Every pair is token-count matched, and that is exact rather than approximate.**
// `minimal_pairs` can only match on *words*, because its two sentences contain
// different words and BPE may segment them differently. Here both sides carry
// identical prose and swap one reserved marker for another, and every marker is a
// single id by construction (`pretokenize::SPECIAL_BLOCK`) — so equal marker counts
// plus identical stripped prose is equal token counts, with no tokenizer needed to
// verify it. `Paradigm::build` enforces exactly that and a test asserts it holds for
// the whole suite.
//
// The other invariant a grammar suite cannot have: every `good` side is well formed
// under `markup::ids_are_balanced` and every `bad` side is not. The suite checks
// itself, so a paradigm cannot quietly become two grammatical sentences the way
// `minimal_pairs`'s first draft of `embedded_question_inversion` did.

use crate::eval::surface;

pub struct Pair {
    pub good: String,
    pub bad: String,
}

pub struct Paradigm {
    pub name: String,
    pub description: String,
    pub pairs: Vec<Pair>,
}

// Matches `minimal_pairs::PAIRS_PER_PARADIGM`, for the same reason: enough pairs that
// a paradigm's own reported accuracy means something.
const PAIRS_PER_PARADIGM: usize = 180;

impl Paradigm {
    fn build(name: &str, description: &str, pairs: Vec<(String, String)>) -> Paradigm {
        let pairs = pairs
            .into_iter()
            .filter(|(good, bad)| matched(good, bad))
            .take(PAIRS_PER_PARADIGM)
            .map(|(good, bad)| Pair { good, bad })
            .collect();
        Paradigm { name: name.to_string(), description: description.to_string(), pairs }
    }
}

// Same prose, same number of markers — which is what makes the two sides the same
// number of tokens under any tokenizer.
fn matched(good: &str, bad: &str) -> bool {
    surface::strip_markers(good) == surface::strip_markers(bad)
        && surface::markers_in(good).len() == surface::markers_in(bad).len()
}

// Prose fillers. Deliberately plain and short: the sentence is scaffolding for the
// markers around it, and anything distinctive would let a model win on the prose.
const FILLERS: &[&str] = &[
    "the water was cold",
    "she went to the market",
    "the train arrived late",
    "he opened the window",
    "the bread had gone stale",
    "they walked along the river",
    "the letter was never sent",
    "the garden needed weeding",
    "his coat was too thin",
    "the meeting ran over",
    "the road was closed",
    "she found the missing key",
    "the lamp flickered twice",
    "the harvest came in early",
    "the door would not shut",
    "they waited for the bus",
    "the soup needed salt",
    "the clock had stopped",
    "the path led uphill",
    "she folded the map",
    "the fire burned low",
    "the message was clear",
    "the boat drifted out",
    "he counted the coins",
    "the roof began to leak",
    "the field lay empty",
    "the shop closed at six",
    "the story made no sense",
    "the ink had dried up",
    "the stairs were steep",
];

pub fn suite() -> Vec<Paradigm> {
    vec![
        closer_matches_opener(),
        block_is_closed(),
        no_stray_closer(),
        nesting_order(),
        turn_closes_its_own_role(),
        thinking_precedes_the_answer(),
    ]
}

// ── the paradigms ───────────────────────────────────────────────────────────

// A closer must name the block it closes. The commonest real failure: the model
// learns that *a* closer follows and picks the wrong one.
fn closer_matches_opener() -> Paradigm {
    let blocks = [
        ("<|par|>", "<|/par|>", "<|/item|>"),
        ("<|item|>", "<|/item|>", "<|/par|>"),
        ("<|bold|>", "<|/bold|>", "<|/italic|>"),
        ("<|italic|>", "<|/italic|>", "<|/bold|>"),
        ("<|quote|>", "<|/quote|>", "<|/heading|>"),
        ("<|heading|>", "<|/heading|>", "<|/quote|>"),
    ];
    let mut pairs = Vec::new();
    for (open, close, wrong) in blocks {
        for filler in FILLERS {
            pairs.push((format!("{open}{filler}{close}"), format!("{open}{filler}{wrong}")));
        }
    }
    Paradigm::build(
        "closer_matches_opener",
        "A closing marker must name the block it closes.",
        pairs,
    )
}

// A block gets closed, not reopened. Repeating the opener keeps the marker count and
// the prose identical while leaving the block unterminated.
fn block_is_closed() -> Paradigm {
    let blocks = [
        ("<|heading|>", "<|/heading|>"),
        ("<|par|>", "<|/par|>"),
        ("<|codeblock|>", "<|/codeblock|>"),
        ("<|quote|>", "<|/quote|>"),
        ("<|title|>", "<|/title|>"),
        ("<|bold|>", "<|/bold|>"),
    ];
    let mut pairs = Vec::new();
    for (open, close) in blocks {
        for filler in FILLERS {
            pairs.push((format!("{open}{filler}{close}"), format!("{open}{filler}{open}")));
        }
    }
    Paradigm::build(
        "block_is_closed",
        "A block is terminated by its closer, not by reopening it.",
        pairs,
    )
}

// Open before close. Swapping the two keeps every token and reverses the structure,
// so this is a pure ordering probe — nothing can be won from the marker vocabulary.
fn no_stray_closer() -> Paradigm {
    let blocks = [
        ("<|bold|>", "<|/bold|>"),
        ("<|italic|>", "<|/italic|>"),
        ("<|par|>", "<|/par|>"),
        ("<|quote|>", "<|/quote|>"),
        ("<|link|>", "<|/link|>"),
        ("<|warn|>", "<|/warn|>"),
    ];
    let mut pairs = Vec::new();
    for (open, close) in blocks {
        for filler in FILLERS {
            pairs.push((format!("{open}{filler}{close}"), format!("{close}{filler}{open}")));
        }
    }
    Paradigm::build(
        "no_stray_closer",
        "A block opens before it closes.",
        pairs,
    )
}

// Nested blocks close innermost first. The bad side closes them in the order they
// were opened, which is the natural-looking mistake and the one that breaks the
// frontend's renderer.
fn nesting_order() -> Paradigm {
    let nests = [
        ("<|list|>", "<|/list|>", "<|item|>", "<|/item|>"),
        ("<|par|>", "<|/par|>", "<|bold|>", "<|/bold|>"),
        ("<|par|>", "<|/par|>", "<|italic|>", "<|/italic|>"),
        ("<|quote|>", "<|/quote|>", "<|par|>", "<|/par|>"),
        ("<|item|>", "<|/item|>", "<|bold|>", "<|/bold|>"),
        ("<|warn|>", "<|/warn|>", "<|par|>", "<|/par|>"),
    ];
    let mut pairs = Vec::new();
    for (outer_open, outer_close, inner_open, inner_close) in nests {
        for filler in FILLERS {
            pairs.push((
                format!("{outer_open}{inner_open}{filler}{inner_close}{outer_close}"),
                format!("{outer_open}{inner_open}{filler}{outer_close}{inner_close}"),
            ));
        }
    }
    Paradigm::build(
        "nesting_order",
        "Nested blocks close innermost first.",
        pairs,
    )
}

// Each conversational turn closes the role it opened. The bad side crosses the two
// closers, which is the scaffold-level version of `closer_matches_opener` and the one
// that actually corrupts a chat transcript.
fn turn_closes_its_own_role() -> Paradigm {
    let mut pairs = Vec::new();
    for filler in FILLERS {
        for reply in ["yes that is right", "I can help with that", "here is the answer"] {
            pairs.push((
                format!("<|user|>{filler}<|/user|><|assistant|>{reply}<|/assistant|>"),
                format!("<|user|>{filler}<|/assistant|><|assistant|>{reply}<|/user|>"),
            ));
        }
    }
    Paradigm::build(
        "turn_closes_its_own_role",
        "A turn closes the role it opened.",
        pairs,
    )
}

// The thinking block comes before the answer and is closed before it starts. This is
// the marker pair the frontend depends on most directly — it hides everything between
// the two — and the one the tiered `think-*` training data exists to teach.
fn thinking_precedes_the_answer() -> Paradigm {
    let mut pairs = Vec::new();
    for filler in FILLERS {
        for answer in ["so the answer is no", "that should work", "it depends on the case"] {
            pairs.push((
                format!("<|thinking|>{filler}<|/thinking|>{answer}"),
                format!("<|/thinking|>{filler}<|thinking|>{answer}"),
            ));
        }
    }
    Paradigm::build(
        "thinking_precedes_the_answer",
        "A thinking block opens, closes, and then the answer follows.",
        pairs,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::markup;

    #[test]
    fn every_pair_carries_identical_prose_and_marker_counts() {
        // What makes the pair token-count matched without consulting a tokenizer.
        for paradigm in suite() {
            for pair in &paradigm.pairs {
                assert_eq!(
                    surface::strip_markers(&pair.good),
                    surface::strip_markers(&pair.bad),
                    "{}: prose differs",
                    paradigm.name,
                );
                assert_eq!(
                    surface::markers_in(&pair.good).len(),
                    surface::markers_in(&pair.bad).len(),
                    "{}: marker counts differ",
                    paradigm.name,
                );
            }
        }
    }

    // The self-consistency check a grammar suite cannot perform: the benchmark's own
    // notion of well-formedness must agree with which side is labelled good.
    #[test]
    fn every_good_side_is_well_formed_and_every_bad_side_is_not() {
        let merges = Vec::new();
        for paradigm in suite() {
            for pair in &paradigm.pairs {
                let good = tokenize::bpe::encode(&pair.good, &merges);
                let bad = tokenize::bpe::encode(&pair.bad, &merges);
                assert!(
                    markup::ids_are_balanced(&good),
                    "{}: good side is malformed: {:?}",
                    paradigm.name,
                    pair.good,
                );
                assert!(
                    !markup::ids_are_balanced(&bad),
                    "{}: bad side is well formed: {:?}",
                    paradigm.name,
                    pair.bad,
                );
            }
        }
    }

    // The claim the header makes, verified through the real tokenizer path rather
    // than argued from the marker counts.
    #[test]
    fn every_pair_encodes_to_the_same_number_of_tokens() {
        // Byte-level fallback: deterministic, and markers still come out as single
        // reserved ids, which is the property under test.
        let merges = Vec::new();
        for paradigm in suite() {
            for pair in &paradigm.pairs {
                let good = tokenize::bpe::encode(&pair.good, &merges);
                let bad = tokenize::bpe::encode(&pair.bad, &merges);
                assert_eq!(
                    good.len(),
                    bad.len(),
                    "{}: {:?} vs {:?}",
                    paradigm.name,
                    pair.good,
                    pair.bad,
                );
            }
        }
    }

    #[test]
    fn every_marker_in_the_suite_is_a_single_token() {
        let merges = Vec::new();
        for paradigm in suite() {
            for pair in &paradigm.pairs {
                for (_, marker) in surface::markers_in(&pair.good) {
                    assert_eq!(
                        tokenize::bpe::encode(marker, &merges).len(),
                        1,
                        "{marker} is not one token",
                    );
                }
            }
        }
    }

    #[test]
    fn the_two_sides_of_a_pair_always_differ() {
        for paradigm in suite() {
            for pair in &paradigm.pairs {
                assert_ne!(pair.good, pair.bad, "{}", paradigm.name);
            }
        }
    }

    #[test]
    fn every_paradigm_has_pairs_and_respects_the_cap() {
        for paradigm in suite() {
            assert!(paradigm.pairs.len() >= 60, "{} is too small", paradigm.name);
            assert!(paradigm.pairs.len() <= PAIRS_PER_PARADIGM, "{}", paradigm.name);
        }
    }

    #[test]
    fn paradigm_names_are_unique() {
        let mut names: Vec<String> = suite().into_iter().map(|p| p.name).collect();
        let before = names.len();
        names.sort();
        names.dedup();
        assert_eq!(before, names.len());
    }

    #[test]
    fn the_suite_is_deterministic() {
        let of = || -> Vec<String> {
            suite().iter().flat_map(|p| p.pairs.iter().map(|x| x.good.clone())).collect()
        };
        assert_eq!(of(), of());
    }

    #[test]
    fn the_suite_is_large_enough_to_resolve_a_round() {
        let pairs: usize = suite().iter().map(|p| p.pairs.len()).sum();
        assert!(pairs >= 800, "only {pairs} pairs");
    }

    #[test]
    fn build_drops_a_pair_whose_prose_differs() {
        let paradigm = Paradigm::build(
            "test",
            "",
            vec![
                ("<|par|>a<|/par|>".to_string(), "<|par|>a<|/item|>".to_string()),
                ("<|par|>a<|/par|>".to_string(), "<|par|>b<|/par|>".to_string()),
            ],
        );
        assert_eq!(paradigm.pairs.len(), 1);
    }

    #[test]
    fn build_drops_a_pair_whose_marker_counts_differ() {
        let paradigm = Paradigm::build(
            "test",
            "",
            vec![("<|par|>a<|/par|>".to_string(), "<|par|>a".to_string())],
        );
        assert!(paradigm.pairs.is_empty());
    }
}
