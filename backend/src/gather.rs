// Drives a signal-gathering run: generate one short reply per prompt and hand each
// one straight out to be rated by a person.
//
// **Why this exists.** KTO trains on thumbs, and thumbs are a by-product of
// chatting — so a freshly trained model can never be preference-tuned, because
// nobody has talked to it enough to have rated anything. Waiting for organic
// ratings to accumulate is not a plan; generating replies whose only purpose is to
// be rated is.
//
// **How it differs from `autolabel`.** Autolabel generates several candidates per
// prompt and has `llm::autolabel::Judge` pick a best and a worst by rule. This
// generates *one* reply per prompt and asks a human. The rule-based judge scores
// surface properties — fluency, real words, whether the turn closed — and is
// deliberately independent of taste; a person rating a reply is the taste. The two
// produce different data on purpose and both land in `raw/preference/`.
//
// **The output does not go to a file here.** Unlike autolabel, this writes nothing:
// a generated reply is not training data until somebody has judged it. The samples
// go to `signal_samples` (written by the queue runner, which owns the pool), and
// only the rated ones reach `raw/preference/` via `queue::export_preference_data`.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// **Two different bars, for two different jobs.**
//
// Hand-rated replies are the *seed*: enough of them to calibrate the judge and to
// give the benchmark a held-out slice, and no more, because clicking thumbs is the
// slow step. The catalogue is what a tune actually trains on, and it is built by
// auto-labelling — so its bar is two orders of magnitude higher and costs GPU time
// rather than attention.
//
// Conflating the two is what produced a tune on thirty examples: the seed bar was
// cleared, so training was allowed, and thirty replies run five times narrowed the
// model onto them.

// Human ratings needed before the judge can be calibrated. Matching
// `autolabel::MIN_HUMAN_RATINGS` — the threshold is fit to these, and below a few
// dozen it moves with every new rating.
pub const MIN_SIGNALS: i64 = 32;

// And at least this many human ratings of each verdict. Calibration needs examples
// of both to find a boundary between them at all; the AUC is undefined on one-sided
// data.
pub const MIN_PER_CLASS: i64 = 8;

// Examples — human and auto-labelled together — before a KTO run is worth starting.
//
// **Why 300.** KTO's gradient saturates at both ends (see `llm::kto::step`), so
// each example contributes a small nudge and the tune is an average over the
// catalogue's variety. At thirty examples there is no variety to average and the
// run memorizes. 300 is the point where an auto-labelling pass is still minutes
// rather than an hour and there is real variety to average over — a working
// number, deliberately not a comfortable one. Raise it once a pass is cheap: the
// epoch cap below scales with it, so nothing else needs touching.
pub const MIN_CATALOGUE: i64 = 300;

// And at least this many of each verdict in the catalogue. Same reasoning as
// `KtoParams::balanced`: with one class empty there is nothing to balance against
// and the tune degenerates into "make everything likelier" (or less likely).
pub const MIN_CATALOGUE_PER_CLASS: i64 = 60;

// Examples per epoch a run is allowed. A hard backstop under the catalogue bar:
// repeated passes over a small set is how a preference tune narrows a model, and
// the bar above can be lowered by editing one constant while this stays true.
const EXAMPLES_PER_EPOCH: i64 = 100;

/// The most epochs a catalogue this size should be run for. Always at least 1 —
/// the catalogue bar is what refuses a small run, not this. Capped at 5 however
/// large the catalogue gets: past that a preference tune is drifting from the base
/// model rather than nudging it, which is the KL anchor's job to prevent and not
/// something more epochs improve.
pub fn max_epochs_for(catalogue: i64) -> i64 {
    (catalogue / EXAMPLES_PER_EPOCH).clamp(1, 5)
}

// Whether there is enough *human* signal to calibrate the judge — the seed bar.
//
// The message names the specific shortfall rather than a generic "not enough":
// "rate more" and "you have thirty thumbs up and no thumbs down" call for
// different actions, and the second is invisible from a total.
pub fn check_signals(desirable: i64, undesirable: i64) -> Result<(), String> {
    let total = desirable + undesirable;
    if total < MIN_SIGNALS {
        return Err(format!(
            "not enough rated replies to calibrate the judge: {total} of {MIN_SIGNALS}. \
             Gather some signals first."
        ));
    }
    if desirable < MIN_PER_CLASS {
        return Err(format!(
            "only {desirable} approved replies — calibration needs at least {MIN_PER_CLASS} \
             of each verdict to find the boundary between them."
        ));
    }
    if undesirable < MIN_PER_CLASS {
        return Err(format!(
            "only {undesirable} rejected replies — calibration needs at least {MIN_PER_CLASS} \
             of each verdict to find the boundary between them."
        ));
    }
    Ok(())
}

// Whether the catalogue is big enough to train on — the training bar. Checked
// *after* `check_signals`, so a caller that fails both reports the seed shortfall
// first: rating is the step that unblocks auto-labelling, which is the step that
// fills the catalogue.
pub fn check_catalogue(approved: i64, rejected: i64) -> Result<(), String> {
    let total = approved + rejected;
    if total < MIN_CATALOGUE {
        return Err(format!(
            "the preference catalogue holds {total} examples, and a tune wants at least \
             {MIN_CATALOGUE}. Training on this few narrows the model onto them rather than \
             teaching it what good English looks like — build the catalogue first."
        ));
    }
    if approved < MIN_CATALOGUE_PER_CLASS || rejected < MIN_CATALOGUE_PER_CLASS {
        return Err(format!(
            "the catalogue is one-sided ({approved} approved, {rejected} rejected) and needs \
             at least {MIN_CATALOGUE_PER_CLASS} of each. KTO balances the two halves against \
             each other; with one side thin the tune just pushes everything one way."
        ));
    }
    Ok(())
}

// Replies per gather run. Sized so a batch is finishable in one sitting: rating is
// the scarce resource here, not generation, and a queue of 200 unrated replies is
// a chore nobody returns to.
const DEFAULT_BATCH: i64 = 16;
// Sampling temperature. Above the chat default for the same reason autolabel's is:
// near-greedy decoding produces the same reply to similar prompts, and rating
// sixteen variations of one answer teaches the tune about one answer.
const DEFAULT_TEMPERATURE: f32 = 0.9;
const DEFAULT_TOP_K: usize = 40;
// Deliberately short. A rating is a snap judgement, and a 400-token wall of text
// gets rated on its first sentence anyway — so generating the rest costs GPU time
// to produce signal about text nobody read.
const MAX_TOKENS: usize = 80;

#[derive(Deserialize, Serialize)]
pub struct GatherRequest {
    // Which model writes the replies. Normally the one about to be tuned:
    // preference data is on-policy for the model that produced it.
    pub id_model: Uuid,
    #[serde(default = "default_batch")]
    pub batch: i64,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_batch() -> i64 {
    DEFAULT_BATCH
}
fn default_temperature() -> f32 {
    DEFAULT_TEMPERATURE
}

#[derive(Serialize)]
pub struct GatherStats {
    pub generated: usize,
    // Replies that came out empty and were dropped. Nothing to rate, and an empty
    // reply renders to no reply tokens, so the KTO builder would drop it anyway.
    pub skipped_empty: usize,
}

pub type ProgressCallback = Box<dyn Fn(&str, f64) + Send>;
// Called with each finished reply, as it finishes. The runner persists it and
// broadcasts it, so the panel can be rating sample 3 while sample 4 generates.
pub type SampleCallback = Box<dyn Fn(&str, &str) + Send>;

// Prompts to fall back on when the chat history is too thin to fill a batch.
//
// Real prompts are always preferred — `db::message::distinct_user_prompts` first,
// for the reason autolabel gives: preference data should describe the distribution
// the model is actually used on. But a brand-new install has no history at all,
// and "gather signals" that answers "you have nothing to gather from" is the same
// dead end as the one this feature exists to remove. Short, broad, and open-ended,
// so the reply has room to be good or bad rather than merely right or wrong.
const SEED_PROMPTS: &[&str] = &[
    "What is the sun made of?",
    "Tell me a short story about a lost dog.",
    "Why do people dream?",
    "Explain what a river is to a small child.",
    "What happens when water freezes?",
    "Describe the sound of rain on a roof.",
    "Who was Marie Curie?",
    "How do birds know where to fly in winter?",
    "What is the difference between a town and a city?",
    "Write two sentences about the ocean at night.",
    "Why is the sky blue?",
    "What makes bread rise?",
    "Tell me something surprising about honey.",
    "How does a library work?",
    "What is a good reason to learn to cook?",
    "Describe autumn in one short paragraph.",
    "What do roots do for a plant?",
    "Explain why boats float.",
    "What is your favourite kind of weather, and why?",
    "Tell me about a animal that lives in the desert.",
    "How do you make a paper aeroplane?",
    "What is thunder?",
    "Why do cats purr?",
    "Describe a busy market street.",
];

// Fills `batch` prompts, preferring the real ones and topping up from the seed list
// in order. Real prompts come first so a run with plenty of history never touches
// the seeds, and the seeds are taken in a stable order so successive top-ups do not
// keep asking the same first question.
pub fn choose_prompts(history: Vec<String>, batch: usize, seed_offset: usize) -> Vec<String> {
    let mut prompts = history;
    prompts.truncate(batch);

    // Walks the whole seed list once from the offset, so skipping a duplicate costs
    // nothing and the batch still fills. It stops there rather than wrapping twice:
    // a batch larger than the seed list comes up short, which is right — the missing
    // slots could only be filled by repeating a prompt, and two ratings of the same
    // question are one rating's worth of signal.
    for i in 0..SEED_PROMPTS.len() {
        if prompts.len() >= batch {
            break;
        }
        let seed = SEED_PROMPTS[(seed_offset + i) % SEED_PROMPTS.len()];
        if prompts.iter().any(|p| p.trim() == seed) {
            continue;
        }
        prompts.push(seed.to_string());
    }
    prompts
}

// Generates the batch. Blocking and GPU-bound, so the caller gives it its own OS
// thread — the same constraint training, benchmarking and auto-labelling have.
//
// `Ok(None)` means cancelled partway. Everything generated before the stop has
// already been handed to `on_sample` and persisted, so a cancelled gather is a
// shorter batch rather than a wasted one.
pub fn run(
    model_dir: &Path,
    prompts: &[String],
    request: &GatherRequest,
    on_progress: ProgressCallback,
    on_sample: SampleCallback,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<Option<GatherStats>, String> {
    let merges = tokenize::bpe::load_merges(&model_dir.join("merges.bin").to_string_lossy())
        .map_err(|e| format!("could not load tokenizer: {e}"))?;
    let model = llm::training::transformer::Model::load(&model_dir.to_string_lossy())
        .map_err(|e| format!("could not load model: {e}"))?;

    let ctx_window = model.positional_table.len();
    let reserve = (ctx_window / 4).max(8);

    // Seeded from the clock rather than the prompt index. Autolabel seeds
    // deterministically so a labelling pass can be reproduced and debugged; here the
    // opposite is wanted — a second gather over the same prompts must produce
    // *different* replies, or the second batch is the first one again and rating it
    // adds no information.
    let run_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);

    let mut generated = 0usize;
    let mut skipped_empty = 0usize;

    for (i, prompt) in prompts.iter().enumerate() {
        if cancelled(&cancel) {
            return Ok(None);
        }
        on_progress(
            &format!("Generating ({}/{})", i + 1, prompts.len()),
            i as f64 / prompts.len() as f64,
        );

        let turns = [llm::chat_template::Turn {
            role: llm::chat_template::Role::User,
            text: prompt.clone(),
        }];
        let seq = llm::chat_template::build_prompt(None, &turns, &merges, ctx_window, reserve);

        let config = llm::eval::generate::SamplingConfig {
            max_tokens: MAX_TOKENS,
            temperature: request.temperature,
            top_k: DEFAULT_TOP_K,
            seed: run_seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        };
        let continuation =
            llm::eval::generate::continue_tokens_blocking(&model, &merges, seq, config);
        // The terminal marker is scaffolding, not part of the reply — showing it in
        // the rating panel would have people rating the template.
        let text = llm::autolabel::strip_terminal(&continuation.text);

        if text.trim().is_empty() {
            skipped_empty += 1;
            continue;
        }

        on_sample(prompt, text.trim());
        generated += 1;
    }

    on_progress("Done", 1.0);
    Ok(Some(GatherStats { generated, skipped_empty }))
}

fn cancelled(cancel: &Option<Arc<AtomicBool>>) -> bool {
    cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_seed_bar_and_the_training_bar_are_different_bars() {
        // The exact confusion that produced a tune on thirty examples: enough human
        // ratings to calibrate is not enough data to train on.
        assert!(check_signals(20, 20).is_ok());
        assert!(check_catalogue(20, 20).is_err());
    }

    #[test]
    fn a_one_sided_catalogue_is_refused_however_large() {
        assert!(check_catalogue(5000, 3).is_err());
        assert!(check_catalogue(200, 200).is_ok());
    }

    #[test]
    fn epochs_are_capped_by_catalogue_size() {
        // Thirty examples must not be runnable five times, whatever was configured.
        assert_eq!(max_epochs_for(30), 1);
        assert_eq!(max_epochs_for(300), 3);
        // And a huge catalogue does not license an unbounded run.
        assert_eq!(max_epochs_for(1_000_000), 5);
    }

    #[test]
    fn history_alone_fills_the_batch() {
        let history: Vec<String> = (0..10).map(|i| format!("prompt {i}")).collect();
        let chosen = choose_prompts(history, 4, 0);
        assert_eq!(chosen.len(), 4);
        // No seed prompt should appear while real ones are available.
        assert!(chosen.iter().all(|p| p.starts_with("prompt ")));
    }

    #[test]
    fn seeds_top_up_a_thin_history() {
        let chosen = choose_prompts(vec!["real one".to_string()], 5, 0);
        assert_eq!(chosen.len(), 5);
        assert_eq!(chosen[0], "real one");
        assert_eq!(chosen[1], SEED_PROMPTS[0]);
    }

    #[test]
    fn an_empty_history_still_produces_a_batch() {
        // The whole point of the fallback: a fresh install can gather.
        let chosen = choose_prompts(Vec::new(), 6, 0);
        assert_eq!(chosen.len(), 6);
    }

    #[test]
    fn the_offset_rotates_which_seeds_are_used() {
        let first = choose_prompts(Vec::new(), 3, 0);
        let second = choose_prompts(Vec::new(), 3, 3);
        assert_ne!(first, second, "a second run should ask different questions");
    }

    #[test]
    fn a_seed_already_covered_by_history_is_not_repeated() {
        let history = vec![SEED_PROMPTS[0].to_string()];
        let chosen = choose_prompts(history, 3, 0);
        assert_eq!(chosen.iter().filter(|p| *p == SEED_PROMPTS[0]).count(), 1);
        // Skipping the duplicate must not cost a slot.
        assert_eq!(chosen.len(), 3);
    }

    #[test]
    fn a_batch_larger_than_the_seed_list_comes_up_short_rather_than_repeating() {
        let chosen = choose_prompts(Vec::new(), SEED_PROMPTS.len() + 5, 0);
        assert_eq!(chosen.len(), SEED_PROMPTS.len());
        let mut unique = chosen.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), chosen.len(), "no prompt should be asked twice");
    }
}
