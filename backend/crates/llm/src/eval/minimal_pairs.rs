// A BLiMP-style grammar benchmark: minimal pairs of sentences that differ in one
// place, one grammatical and one not.
//
// Why this and not MMLU / HellaSwag / ARC. Those measure knowledge and reasoning,
// which a 2-layer, 64-dimension model does not have — they would return chance
// forever and tell us nothing. A minimal pair asks something a small model *can*
// answer: does it assign higher probability to `the cats annoy Tim` than to
// `the cats annoys Tim`? That is a forced choice between two strings, scored by
// log-probability, so it needs no generation, no parsing of the model's output,
// and no judge. Chance is exactly 50%, and every point above it is grammar the
// model has actually acquired. This is the same construction the BabyLM challenge
// uses to evaluate models trained on a child-sized corpus.
//
// The pairs are generated from templates rather than downloaded, so the suite is
// deterministic, offline, and versioned with the code. `load_external` accepts
// real BLiMP JSONL if you want the full 67-paradigm set later.
//
// **Every pair is word-count matched.** Total log-probability is not normalized by
// length, so a pair whose sentences differ in length would be decided by the
// missing word rather than by the grammatical violation. `Paradigm::build` drops
// any pair that isn't matched, and a test asserts the whole built-in suite is.

use std::path::Path;

// One minimal pair. `good` is grammatical, `bad` differs in exactly one place and
// is not.
pub struct Pair {
    pub good: String,
    pub bad: String,
}

// A named group of pairs probing one grammatical phenomenon. Reported separately
// as well as pooled, because *which* grammar a model has is more informative than
// the aggregate: subject-verb agreement usually lands long before filler-gap.
pub struct Paradigm {
    pub name: String,
    pub description: String,
    pub pairs: Vec<Pair>,
}

// Pairs per paradigm. The templates' cross products are larger than this; taking a
// deterministic prefix keeps a run to a predictable length without an RNG (a
// benchmark that samples differently each run can't be compared across runs).
const PAIRS_PER_PARADIGM: usize = 60;

impl Paradigm {
    // Builds a paradigm, enforcing the length-matching invariant and the cap.
    fn build(name: &str, description: &str, pairs: Vec<(String, String)>) -> Paradigm {
        let pairs = pairs
            .into_iter()
            .filter(|(good, bad)| word_count(good) == word_count(bad))
            .take(PAIRS_PER_PARADIGM)
            .map(|(good, bad)| Pair { good, bad })
            .collect();
        Paradigm {
            name: name.to_string(),
            description: description.to_string(),
            pairs,
        }
    }
}

// Words, ignoring attached punctuation — what the length-matching check compares.
fn word_count(sentence: &str) -> usize {
    sentence.split_whitespace().count()
}

// The full built-in suite.
pub fn suite() -> Vec<Paradigm> {
    vec![
        subject_verb_agreement(),
        determiner_noun_agreement(),
        anaphor_gender_agreement(),
        anaphor_number_agreement(),
        irregular_past_tense(),
        argument_structure(),
        filler_gap(),
        npi_licensing(),
        pronoun_case(),
        adjective_order(),
    ]
}

// ── the paradigms ───────────────────────────────────────────────────────────

// Does the verb agree in number with its subject? The single most basic
// long-ish-distance dependency in English, and typically the first one a small
// model acquires — so it doubles as a "is this model alive at all" check.
fn subject_verb_agreement() -> Paradigm {
    // (singular noun, plural noun)
    let nouns = [
        ("girl", "girls"), ("boy", "boys"), ("man", "men"), ("woman", "women"),
        ("dog", "dogs"), ("cat", "cats"), ("teacher", "teachers"), ("farmer", "farmers"),
        ("child", "children"), ("doctor", "doctors"),
    ];
    // (third-person singular, plural/base) + a completing phrase
    let verbs = [
        ("runs", "run", "every morning"),
        ("walks", "walk", "to the river"),
        ("sings", "sing", "in the garden"),
        ("waits", "wait", "by the door"),
        ("sleeps", "sleep", "near the fire"),
        ("works", "work", "on the farm"),
    ];

    let mut pairs = Vec::new();
    for (sg, pl) in nouns {
        for (v_sg, v_pl, tail) in verbs {
            // Singular subject wants the -s form.
            pairs.push((
                format!("The {sg} {v_sg} {tail}."),
                format!("The {sg} {v_pl} {tail}."),
            ));
            // Plural subject wants the bare form — the same contrast, flipped, so
            // the paradigm can't be passed by a model that simply prefers "-s".
            pairs.push((
                format!("The {pl} {v_pl} {tail}."),
                format!("The {pl} {v_sg} {tail}."),
            ));
        }
    }
    Paradigm::build(
        "subject_verb_agreement",
        "The verb must match the number of its subject.",
        pairs,
    )
}

// Demonstratives agree in number with the noun they select: this/that take a
// singular, these/those a plural.
fn determiner_noun_agreement() -> Paradigm {
    let nouns = [
        ("book", "books"), ("chair", "chairs"), ("house", "houses"), ("letter", "letters"),
        ("story", "stories"), ("window", "windows"), ("apple", "apples"), ("river", "rivers"),
        ("picture", "pictures"), ("garden", "gardens"),
    ];
    let frames = [
        ("She wanted", "very much"),
        ("He remembered", "quite clearly"),
        ("They described", "to the class"),
    ];
    let determiners = [("this", "these"), ("that", "those")];

    let mut pairs = Vec::new();
    for (sg, pl) in nouns {
        for (head, tail) in frames {
            for (det_sg, det_pl) in determiners {
                pairs.push((
                    format!("{head} {det_sg} {sg} {tail}."),
                    format!("{head} {det_sg} {pl} {tail}."),
                ));
                pairs.push((
                    format!("{head} {det_pl} {pl} {tail}."),
                    format!("{head} {det_pl} {sg} {tail}."),
                ));
            }
        }
    }
    Paradigm::build(
        "determiner_noun_agreement",
        "A demonstrative must match the number of its noun.",
        pairs,
    )
}

// A reflexive must match the gender of its antecedent. Tests that the model is
// binding the pronoun to a specific earlier noun, not just emitting a plausible
// reflexive.
fn anaphor_gender_agreement() -> Paradigm {
    // (noun, correct reflexive, incorrect reflexive)
    let subjects = [
        ("man", "himself", "herself"), ("boy", "himself", "herself"),
        ("father", "himself", "herself"), ("brother", "himself", "herself"),
        ("king", "himself", "herself"), ("woman", "herself", "himself"),
        ("girl", "herself", "himself"), ("mother", "herself", "himself"),
        ("sister", "herself", "himself"), ("queen", "herself", "himself"),
    ];
    let verbs = ["hurt", "blamed", "watched", "described", "questioned", "trusted"];

    let mut pairs = Vec::new();
    for (noun, good, bad) in subjects {
        for verb in verbs {
            pairs.push((
                format!("The {noun} {verb} {good} in the mirror."),
                format!("The {noun} {verb} {bad} in the mirror."),
            ));
        }
    }
    Paradigm::build(
        "anaphor_gender_agreement",
        "A reflexive pronoun must match its antecedent's gender.",
        pairs,
    )
}

// The same binding dependency, on number instead of gender.
fn anaphor_number_agreement() -> Paradigm {
    let subjects = [
        ("boy", "boys"), ("girl", "girls"), ("man", "men"), ("woman", "women"),
        ("student", "students"), ("soldier", "soldiers"), ("child", "children"),
        ("farmer", "farmers"),
    ];
    let verbs = ["hurt", "blamed", "watched", "defended", "questioned"];

    let mut pairs = Vec::new();
    for (sg, pl) in subjects {
        for verb in verbs {
            pairs.push((
                format!("The {pl} {verb} themselves that evening."),
                format!("The {pl} {verb} itself that evening."),
            ));
            pairs.push((
                format!("The {sg} {verb} itself that evening."),
                format!("The {sg} {verb} themselves that evening."),
            ));
        }
    }
    Paradigm::build(
        "anaphor_number_agreement",
        "A reflexive pronoun must match its antecedent's number.",
        pairs,
    )
}

// Irregular past tenses versus the regular -ed the model would produce if it had
// only learned the rule. This is the classic over-regularization probe: getting it
// right means the lexicon has beaten the pattern.
fn irregular_past_tense() -> Paradigm {
    // (correct past, over-regularized past, object phrase)
    let verbs = [
        ("wrote", "writed", "a long letter"),
        ("took", "taked", "the small key"),
        ("gave", "gived", "the boy a coin"),
        ("ate", "eated", "the last apple"),
        ("came", "comed", "home very late"),
        ("knew", "knowed", "the answer already"),
        ("thought", "thinked", "about it often"),
        ("brought", "bringed", "a heavy bag"),
        ("bought", "buyed", "a new coat"),
        ("caught", "catched", "the morning train"),
        ("taught", "teached", "the class quietly"),
        ("found", "finded", "the missing ring"),
        ("held", "holded", "the door open"),
        ("spoke", "speaked", "to her mother"),
        ("broke", "breaked", "the kitchen window"),
        ("drove", "drived", "into the town"),
        ("chose", "choosed", "the shorter road"),
        ("saw", "seed", "the whole thing"),
        ("went", "goed", "into the woods"),
        ("ran", "runned", "across the field"),
    ];
    let subjects = ["She", "He", "The girl", "The old man"];

    let mut pairs = Vec::new();
    for (good, bad, object) in verbs {
        for subject in subjects {
            pairs.push((
                format!("{subject} {good} {object} yesterday."),
                format!("{subject} {bad} {object} yesterday."),
            ));
        }
    }
    Paradigm::build(
        "irregular_past_tense",
        "Irregular verbs resist the regular -ed rule.",
        pairs,
    )
}

// Can the verb take a direct object at all? Both sentences are well-formed
// sequences of English words in a normal order; only the verb's argument
// structure separates them, so surface n-gram statistics don't give it away.
fn argument_structure() -> Paradigm {
    // (transitive verb, object phrase)
    let transitives = [
        ("ate", "the ripe apple"), ("read", "the long letter"), ("cleaned", "the small room"),
        ("opened", "the heavy door"), ("carried", "the wooden box"), ("painted", "the old fence"),
        ("bought", "the blue coat"), ("watched", "the evening train"),
    ];
    // Strictly intransitive — these cannot take a direct object.
    let intransitives = ["slept", "arrived", "laughed", "coughed", "waited", "vanished", "snored"];
    let subjects = ["The girl", "The farmer", "My brother"];

    let mut pairs = Vec::new();
    for (verb, object) in transitives {
        for (i, intransitive) in intransitives.iter().enumerate() {
            let subject = subjects[i % subjects.len()];
            pairs.push((
                format!("{subject} {verb} {object} this morning."),
                format!("{subject} {intransitive} {object} this morning."),
            ));
        }
    }
    Paradigm::build(
        "argument_structure",
        "Only a transitive verb can take a direct object.",
        pairs,
    )
}

// A wh-phrase leaves a gap; filling that gap as well is ungrammatical. This is a
// genuinely long-distance dependency — the model has to connect `what` at the
// start to an absent object several words later — and is usually one of the last
// things a small model gets, which makes it a useful ceiling probe.
fn filler_gap() -> Paradigm {
    // (verb, the illicit filler that occupies the gap, a legitimate adjunct of the
    // same length so the pair stays word-matched)
    let verbs = [
        ("ate", "the bread", "very quickly"),
        ("read", "the paper", "last night"),
        ("bought", "the coat", "in town"),
        ("carried", "the box", "up here"),
        ("opened", "the door", "quite slowly"),
        ("painted", "the wall", "this week"),
        ("wrote", "the letter", "by hand"),
        ("cleaned", "the room", "on Sunday"),
    ];
    let subjects = ["the boy", "the farmer", "my sister", "the teacher"];
    let frames = ["I know what", "She asked what", "He forgot what"];

    let mut pairs = Vec::new();
    for (verb, filler, adjunct) in verbs {
        for subject in subjects {
            for frame in frames {
                pairs.push((
                    format!("{frame} {subject} {verb} {adjunct}."),
                    format!("{frame} {subject} {verb} {filler}."),
                ));
            }
        }
    }
    Paradigm::build(
        "filler_gap",
        "A wh-question leaves a gap that must not also be filled.",
        pairs,
    )
}

// `ever`, `any`, and `yet` are negative polarity items: they need a licensing
// negative or downward-entailing environment above them. Swapping the licensor for
// a plain determiner is a one-word change that flips grammaticality.
fn npi_licensing() -> Paradigm {
    // (licensor, non-licensor) — both single words, so length matches.
    let licensors = [("No", "The"), ("Few", "Some"), ("Neither", "Either")];
    let bodies = [
        "student has ever finished the exam",
        "teacher had ever seen the letter",
        "farmer has ever entered the barn",
        "child had ever opened the box",
        "soldier has ever crossed the river",
        "doctor had ever read the report",
        "writer has ever printed the story",
    ];

    let mut pairs = Vec::new();
    for (good, bad) in licensors {
        for body in bodies {
            pairs.push((format!("{good} {body}."), format!("{bad} {body}.")));
        }
    }
    // `Neither`/`Either` want a plural body; the plural variants keep the paradigm
    // from being a single template repeated.
    for body in [
        "students have ever finished the exam",
        "teachers had ever seen the letter",
        "farmers have ever entered the barn",
    ] {
        pairs.push((format!("Few {body}."), format!("Some {body}.")));
    }
    Paradigm::build(
        "npi_licensing",
        "`ever` needs a negative licensor above it.",
        pairs,
    )
}

// Subject position takes nominative case, object position accusative. English
// marks this only on pronouns, so it is a clean one-word contrast.
fn pronoun_case() -> Paradigm {
    // (nominative, accusative)
    let pronouns = [("He", "Him"), ("She", "Her"), ("They", "Them"), ("We", "Us"), ("I", "Me")];
    let predicates = [
        "arrived at the station early",
        "walked into the quiet room",
        "waited by the old gate",
        "answered the question at once",
        "opened the door very slowly",
        "carried the box down stairs",
    ];
    // Object-position frames use the lowercase accusative/nominative forms.
    let objects = [("him", "he"), ("her", "she"), ("them", "they"), ("us", "we"), ("me", "I")];
    let transitives = ["The teacher saw", "My brother told", "The farmer helped"];

    let mut pairs = Vec::new();
    for (nom, acc) in pronouns {
        for predicate in predicates {
            pairs.push((format!("{nom} {predicate}."), format!("{acc} {predicate}.")));
        }
    }
    for (acc, nom) in objects {
        for frame in transitives {
            pairs.push((
                format!("{frame} {acc} this morning."),
                format!("{frame} {nom} this morning."),
            ));
        }
    }
    Paradigm::build(
        "pronoun_case",
        "Subjects take nominative case, objects accusative.",
        pairs,
    )
}

// An attributive adjective precedes its noun. A word-order violation with the same
// words in both sentences — nothing is added or removed, so a model can only get
// this from position, not from vocabulary.
fn adjective_order() -> Paradigm {
    let adjectives = ["tall", "quiet", "angry", "young", "clever", "tired", "small", "kind"];
    let nouns = ["man", "girl", "farmer", "teacher", "soldier", "doctor", "boy", "woman"];
    let predicates = ["walked home alone", "opened the gate", "waited for hours"];

    let mut pairs = Vec::new();
    for adjective in adjectives {
        for noun in nouns {
            for predicate in predicates {
                pairs.push((
                    format!("The {adjective} {noun} {predicate}."),
                    format!("The {noun} {adjective} {predicate}."),
                ));
            }
        }
    }
    Paradigm::build(
        "adjective_order",
        "An attributive adjective comes before its noun.",
        pairs,
    )
}

// ── external BLiMP ──────────────────────────────────────────────────────────

// Loads real BLiMP paradigms from a directory of JSONL files, one paradigm per
// file, each line `{"sentence_good": ..., "sentence_bad": ...}` — the published
// BLiMP format. Drop the dataset in and it is picked up alongside the built-in
// suite; the file stem names the paradigm.
//
// Pairs that aren't word-count matched are dropped, same as the built-ins: BLiMP
// is mostly matched but not entirely, and an unmatched pair measures length rather
// than grammar under an unnormalized score.
pub fn load_external(dir: &Path) -> Vec<Paradigm> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        // No external set installed is the normal case, not an error.
        Err(_) => return Vec::new(),
    };

    let mut paradigms = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("external")
            .to_string();

        let mut pairs = Vec::new();
        for line in text.lines() {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let good = value.get("sentence_good").and_then(|v| v.as_str());
            let bad = value.get("sentence_bad").and_then(|v| v.as_str());
            if let (Some(good), Some(bad)) = (good, bad) {
                pairs.push((good.to_string(), bad.to_string()));
            }
        }
        if pairs.is_empty() {
            continue;
        }
        paradigms.push(Paradigm::build(&name, "External BLiMP paradigm.", pairs));
    }
    paradigms.sort_by(|a, b| a.name.cmp(&b.name));
    paradigms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_built_in_pair_is_word_count_matched() {
        // The whole benchmark rests on this: total log-probability is not length
        // normalized, so an unmatched pair would be decided by the extra word.
        for paradigm in suite() {
            for pair in &paradigm.pairs {
                assert_eq!(
                    word_count(&pair.good),
                    word_count(&pair.bad),
                    "{}: {:?} vs {:?}",
                    paradigm.name,
                    pair.good,
                    pair.bad,
                );
            }
        }
    }

    #[test]
    fn every_paradigm_has_pairs_and_respects_the_cap() {
        for paradigm in suite() {
            assert!(!paradigm.pairs.is_empty(), "{} is empty", paradigm.name);
            assert!(
                paradigm.pairs.len() <= PAIRS_PER_PARADIGM,
                "{} has {} pairs",
                paradigm.name,
                paradigm.pairs.len(),
            );
        }
    }

    #[test]
    fn the_two_sentences_in_a_pair_always_differ() {
        for paradigm in suite() {
            for pair in &paradigm.pairs {
                assert_ne!(pair.good, pair.bad, "{} has an identical pair", paradigm.name);
            }
        }
    }

    #[test]
    fn paradigm_names_are_unique() {
        let mut names: Vec<String> = suite().into_iter().map(|p| p.name).collect();
        let before = names.len();
        names.sort();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate paradigm name");
    }

    #[test]
    fn the_suite_is_deterministic() {
        let first: Vec<String> = suite().iter().flat_map(|p| p.pairs.iter().map(|x| x.good.clone())).collect();
        let second: Vec<String> = suite().iter().flat_map(|p| p.pairs.iter().map(|x| x.good.clone())).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn build_drops_length_mismatched_pairs() {
        let paradigm = Paradigm::build(
            "test",
            "",
            vec![
                ("a b c".to_string(), "a b d".to_string()),
                ("a b c".to_string(), "a b c d".to_string()),
            ],
        );
        assert_eq!(paradigm.pairs.len(), 1);
    }

    #[test]
    fn missing_external_directory_yields_nothing() {
        assert!(load_external(Path::new("/nonexistent/blimp")).is_empty());
    }
}
