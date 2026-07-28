import { apiFetch } from "./api"

// Wire types for the benchmark API (backend routes/benchmark.rs). The report is
// the serialized `llm::eval::Report`; every metric there is an f64, and a metric
// that could not be measured serializes as `null` rather than 0 — so every number
// below is nullable and the UI renders "—" for a null instead of a confident zero.

export type BenchmarkStatus = "running" | "done" | "failed"

export interface ModelInfo {
  dimensions: number
  layers: number
  heads: number
  context: number
  vocab: number
  parameters: number
}

export interface CorpusMetrics {
  bits_per_byte: number | null
  bits_per_token: number | null
  perplexity: number | null
  tokens_scored: number
  bytes_scored: number
  baseline_bits_per_byte: number | null
  // Fraction of the n-gram floor's bits the model removes. Stays 0 for a model
  // that has not beaten the floor, which at this scale is all of them — it is a
  // verdict, not a scale. `floor_progress` is the one that moves.
  advantage: number | null
  // How far the model has come from predicting nothing (0) to matching the n-gram
  // floor (1). This is the corpus term the composite score reads.
  floor_progress: number | null
  files: number
}

export interface ParadigmResult {
  name: string
  description: string
  accuracy: number | null
  pairs: number
}

export interface GrammarMetrics {
  accuracy: number | null
  pairs: number
  paradigms: ParadigmResult[]
}

export interface GenerationSample {
  prompt: string
  completion: string
}

export interface GenerationMetrics {
  real_word_rate: number | null
  distinct_1: number | null
  distinct_2: number | null
  distinct_3: number | null
  repeat_rate: number | null
  mean_sentence_words: number | null
  marker_validity: number | null
  words_scored: number
  lexicon_size: number
  samples: GenerationSample[]
}

// Bits per assistant-reply token over held-out conversations, scored through the
// SFT loss mask. Kept out of the composite score: a base model has never seen the
// chat scaffold, so this is read as a paired comparison between a base model and
// its fine-tune, not as a number to rank every run by.
export interface ChatMetrics {
  bits_per_token: number | null
  logprob: number | null
  conversations: number
  tokens_scored: number
}

export interface BenchmarkReport {
  score: number | null
  model: ModelInfo
  corpus: CorpusMetrics | null
  grammar: GrammarMetrics
  generation: GenerationMetrics
  // Null until a held-out conversation set is installed under `eval/chat/`.
  chat: ChatMetrics | null
  duration_secs: number
}

export interface BenchmarkRun {
  id_run: string
  id_model: string
  model_name: string
  status: BenchmarkStatus
  score: number | null
  error: string | null
  // Tokens in the corpus the model was trained on, snapshotted when the run
  // started (so retraining the model doesn't relabel past runs). `null` for a model
  // whose training config never recorded a corpus size — such a run is left off the
  // scaling chart rather than plotted at zero.
  train_tokens: number | null
  created_at: string
  completed_at: string | null
}

export interface BenchmarkRunDetail extends BenchmarkRun {
  report: BenchmarkReport | null
}

// Whether the reference corpus is installed. Surfaced on the page so the absence
// of perplexity numbers reads as "no corpus installed" rather than as a bug.
export interface SuiteInfo {
  corpus_dir: string
  corpus_files: number
  blimp_dir: string
  blimp_files: number
  chat_dir: string
  chat_files: number
}

export async function listRuns(): Promise<BenchmarkRun[]> {
  const res = await apiFetch("/benchmark/runs")
  if (!res.ok) throw new Error("Failed to load benchmark history.")
  return res.json()
}

/**
 * One model's benchmark record: its latest completed report, and every run of it.
 *
 * `latest` is the most recent run that produced a report — deliberately not "the
 * most recent run", which may be a failure with nothing in it to show. The failed
 * run is still in `runs`, where its error belongs.
 */
export interface ModelBenchmarks {
  latest: BenchmarkRunDetail | null
  runs: BenchmarkRun[]
}

export async function getModelRuns(idModel: string): Promise<ModelBenchmarks> {
  const res = await apiFetch(`/benchmark/models/${idModel}/runs`)
  if (!res.ok) throw new Error("Failed to load this model's benchmarks.")
  return res.json()
}

export async function getRun(id: string): Promise<BenchmarkRunDetail> {
  const res = await apiFetch(`/benchmark/runs/${id}`)
  if (!res.ok) throw new Error("Failed to load benchmark run.")
  return res.json()
}

export async function getSuite(): Promise<SuiteInfo> {
  const res = await apiFetch("/benchmark/suite")
  if (!res.ok) throw new Error("Failed to load suite info.")
  return res.json()
}

export async function deleteRun(id: string): Promise<void> {
  const res = await apiFetch(`/benchmark/runs/${id}`, { method: "DELETE" })
  if (!res.ok) throw new Error("Failed to delete benchmark run.")
}

// ── formatting ──────────────────────────────────────────────────────────────

/** Em dash for an unmeasured metric — never 0, which would read as a real value. */
export const NO_VALUE = "—"

export function formatNumber(
  value: number | null | undefined,
  digits = 2
): string {
  if (value === null || value === undefined || !Number.isFinite(value))
    return NO_VALUE
  return value.toFixed(digits)
}

export function formatPercent(
  value: number | null | undefined,
  digits = 1
): string {
  if (value === null || value === undefined || !Number.isFinite(value))
    return NO_VALUE
  return `${(value * 100).toFixed(digits)}%`
}

export function formatCompact(n: number): string {
  if (n >= 1_000_000_000) return `${trim(n / 1_000_000_000)}B`
  if (n >= 1_000_000) return `${trim(n / 1_000_000)}M`
  if (n >= 1_000) return `${trim(n / 1_000)}K`
  return `${n}`
}

function trim(n: number) {
  return n % 1 ? n.toFixed(1) : `${n}`
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${Math.round(seconds % 60)}s`
}

/**
 * A duration at its shortest unambiguous length, for axis ticks where
 * `formatDuration`'s "45.0s" is a decimal place too many: `45s`, `2m`, `1m 40s`,
 * `1h 5m`. Every value formats without a decimal point, so a tick reads as a round
 * time even when the number that produced it isn't round.
 */
export function formatDurationCompact(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60)
    const rest = Math.round(seconds % 60)
    return rest === 0 ? `${minutes}m` : `${minutes}m ${rest}s`
  }
  const hours = Math.floor(seconds / 3600)
  const rest = Math.round((seconds % 3600) / 60)
  return rest === 0 ? `${hours}h` : `${hours}h ${rest}m`
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

/** Turns `subject_verb_agreement` into `Subject verb agreement`. */
export function humanizeParadigm(name: string): string {
  const spaced = name.replace(/_/g, " ")
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}
