// Plain-English explanations for the built-in minimal-pair paradigms
// (backend/src/llm/src/eval/minimal_pairs.rs).
//
// The backend already ships a one-line `description` per paradigm, but it's
// written for someone who already knows the term — "`ever` needs a negative
// licensor above it" doesn't help if "licensor" is the unfamiliar word. These
// entries say what the phenomenon is in ordinary language and then show the exact
// contrast the suite tests, because a good/bad pair explains a grammar rule faster
// than a definition does.
//
// Keyed by the paradigm's backend `name`. External BLiMP paradigms aren't listed —
// there are 67 of them and their names are self-describing — so lookups fall back
// to whatever description the backend sent.

export interface ParadigmGloss {
  /** What the paradigm tests, in ordinary language. */
  summary: string
  /** The grammatical sentence, as the suite actually generates it. */
  good: string
  /** Its minimal pair: one word changed or moved, now ungrammatical. */
  bad: string
}

export const PARADIGM_GLOSSARY: Record<string, ParadigmGloss> = {
  subject_verb_agreement: {
    summary:
      "A verb has to match whether its subject is one thing or many. Usually the first grammar a small model picks up, so it doubles as an is-this-model-alive check.",
    good: "The cat runs every morning.",
    bad: "The cat run every morning.",
  },
  determiner_noun_agreement: {
    summary:
      "this/that point at one thing, these/those at several — the word has to match the noun it introduces.",
    good: "She wanted this book very much.",
    bad: "She wanted this books very much.",
  },
  anaphor_gender_agreement: {
    summary:
      "himself/herself has to match the gender of the person it refers back to. Passing means the model is tying the pronoun to a specific earlier noun, not just picking a reflexive that sounds plausible.",
    good: "The man blamed himself in the mirror.",
    bad: "The man blamed herself in the mirror.",
  },
  anaphor_number_agreement: {
    summary:
      "The same reference back to an earlier noun, but on number instead of gender: one person takes itself, several take themselves.",
    good: "The boys watched themselves that evening.",
    bad: "The boys watched itself that evening.",
  },
  irregular_past_tense: {
    summary:
      "Irregular verbs don't take -ed. A model that has learned only the rule says \"writed\"; getting this right means the individual word has beaten the pattern.",
    good: "She wrote a long letter yesterday.",
    bad: "She writed a long letter yesterday.",
  },
  argument_structure: {
    summary:
      "Some verbs take an object and some can't. Both sentences are ordinary English word sequences in a normal order — only the verb's requirements separate them, so common word pairs don't give it away.",
    good: "The girl ate the ripe apple this morning.",
    bad: "The girl slept the ripe apple this morning.",
  },
  filler_gap: {
    summary:
      "A what-question leaves a hole later in the sentence, and that hole must stay empty. The model has to connect \"what\" at the start to something missing several words away — a genuinely long-distance dependency, and usually the last thing a small model gets.",
    good: "I know what the boy ate very quickly.",
    bad: "I know what the boy ate the bread.",
  },
  npi_licensing: {
    summary:
      "Words like ever, any, and yet only work under a negative — no, few, neither. Swap the negative for a plain the and the same sentence stops being English.",
    good: "No student has ever finished the exam.",
    bad: "The student has ever finished the exam.",
  },
  pronoun_case: {
    summary:
      "English changes pronouns by their job in the sentence: he/she/they do the acting, him/her/them are acted on.",
    good: "He arrived at the station early.",
    bad: "Him arrived at the station early.",
  },
  adjective_order: {
    summary:
      "A describing word comes before the noun it describes. Both sentences use exactly the same words — nothing added or removed — so the model can only get this from position, never from vocabulary.",
    good: "The tall man walked home alone.",
    bad: "The man tall walked home alone.",
  },
}

/**
 * The gloss for a paradigm, or `null` for one we have no entry for (any external
 * BLiMP paradigm) — callers fall back to the backend's own description.
 */
export function paradigmGloss(name: string): ParadigmGloss | null {
  return PARADIGM_GLOSSARY[name] ?? null
}
