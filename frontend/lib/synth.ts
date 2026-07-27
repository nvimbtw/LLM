import { apiFetch } from "./api"

// The SFT corpus, measured (backend routes/synth.rs).
//
// Generation itself is a job, so it lives in `lib/jobs.ts` with the other kinds;
// this is only the read side — what is on disk now, and whether the model that
// will train on it can hold it.
//
// The token counts are real: the backend renders every conversation through the
// same chat template and tokenizer `build_sft_dataset` uses. A character estimate
// would answer a different and much less useful question.

/** One `.jsonl` file in the SFT directory. */
export interface CorpusFile {
  file: string
  conversations: number
  turns: number
  with_thinking: number
  /** Rendered token length of every conversation, for the distribution. */
  tokens: number[]
  /** How many of them exceed the context window. */
  over_budget: number
  user_chars_median: number
  assistant_chars_median: number
}

export interface CorpusStats {
  files: CorpusFile[]
  conversations: number
  /** Absent when no model was named — there is then no window to measure against. */
  ctx_window: number | null
  over_budget: number
}

/**
 * Reads the corpus. `modelId` names the model whose tokenizer and context window
 * to measure against; without it the counts still come back and the token
 * distribution is simply absent rather than guessed.
 */
export async function getCorpus(modelId?: string): Promise<CorpusStats> {
  const query = modelId ? `?id_model=${encodeURIComponent(modelId)}` : ""
  const res = await apiFetch(`/synth/corpus${query}`)
  if (!res.ok) {
    throw new Error((await res.text()) || "Failed to read the SFT corpus.")
  }
  return res.json()
}
