import { apiFetch } from "./api"
import type { DatasetSelection } from "./training"

// Built vocabularies: the tokenizers a training run can be pointed at.
//
// A vocabulary is learned from a selection of corpora and then kept, so several
// runs can share one — and so a run's packed corpus can be cached, since a
// corpus is only meaningful under the vocabulary that encoded it.
//
// Building one is a job (`queueTokenizer` in `lib/jobs.ts`); this is the reading
// half.

/** One corpus a vocabulary was learned from, as the LLM engine records it
 *  (`corpus.remote.Selected.record`): what was asked for — the cap, null for
 *  every document — beside what answered at build time. */
export interface TokenizerDataset {
  id: string
  name: string
  count: number | null
  documents: number
  updated_at: string
}

export interface TokenizerInfo {
  id: string
  name: string
  // Learned by a training run for itself rather than built by name here. Listed,
  // because it is a real build other runs will reuse, but not offered as a choice.
  implicit: boolean
  vocabSize: number | null
  merges: number | null
  maxVocab: number | null
  datasets: TokenizerDataset[]
  files: number | null
  bytes: number
  builtAt: string | null
  // False when the row outlived its directory. Kept in the list so the models
  // naming it are still explicable, but never offered to a new run.
  hasVocab: boolean
}

/** What a build is asked for: the corpora, and how large a vocabulary to learn. */
export interface TokenizerBuild {
  tokenizer_id: string
  max_vocab: number
  datasets: DatasetSelection[]
}

interface RawTokenizer {
  tokenizer_id: string
  name: string
  implicit: boolean
  vocab_size: number | null
  merges: number | null
  max_vocab: number | null
  datasets: TokenizerDataset[]
  files: number | null
  bytes: number
  built_at: string | null
  has_vocab: boolean
}

function fromRow(raw: RawTokenizer): TokenizerInfo {
  return {
    id: raw.tokenizer_id,
    name: raw.name,
    implicit: raw.implicit,
    vocabSize: raw.vocab_size,
    merges: raw.merges,
    maxVocab: raw.max_vocab,
    datasets: raw.datasets ?? [],
    files: raw.files,
    bytes: raw.bytes ?? 0,
    builtAt: raw.built_at,
    hasVocab: raw.has_vocab,
  }
}

export async function listTokenizers(): Promise<TokenizerInfo[]> {
  const res = await apiFetch("/tokenizers")
  if (!res.ok) throw new Error("Unable to load tokenizers.")
  const data = (await res.json()) as { tokenizers: RawTokenizer[] }
  return data.tokenizers.map(fromRow)
}

export async function deleteTokenizer(id: string): Promise<number> {
  const res = await apiFetch(`/tokenizers/${id}`, { method: "DELETE" })
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to delete the tokenizer.")
  // Deleting a vocabulary takes the corpora packed under it: without it they are
  // token ids nothing can read. The count is reported so that is visible.
  const data = (await res.json()) as { corpora_deleted?: number }
  return data.corpora_deleted ?? 0
}

/** What a vocabulary was learned from, for a list row. */
export function describeDatasets(tokenizer: TokenizerInfo): string {
  if (tokenizer.datasets.length === 0) return "—"
  return tokenizer.datasets
    .map(({ name, count }) => (count ? `${name} (${count})` : name))
    .join(", ")
}
