import { apiFetch } from "./api"
import type { Tier } from "./jobs"

// Wire types for the benchmark API (backend routes/benchmark.rs). The report is
// the serialized `llm::eval::Report`; every metric there is an f64, and a metric
// that could not be measured serializes as `null` rather than 0 — so every number
// below is nullable and the UI renders "—" for a null instead of a confident zero.

export type BenchmarkStatus = "running" | "done" | "failed"

/**
 * Which of the three benchmarks a run belongs to.
 *
 * `language` measures whether the model can model English, `markup` whether it uses
 * its own special tokens correctly, and `assistant` whether the replies themselves
 * are any good. Separate runs with separate scores because a supervised fine-tune is
 * expected to move them in *different* directions, and one blended number could not
 * say so — a fine-tune that improved on three of four language components used to
 * read as a regression.
 *
 * The three scores share a 0–100 range and nothing else. Never compare across them,
 * and never average them: the post-train cycle judges a round by requiring that none
 * regressed and at least one improved, which a mean would quietly defeat.
 */
export type BenchmarkSuite = "language" | "markup" | "assistant"

/**
 * A thinking budget, as the marker names it. Re-exported from `lib/jobs` rather than
 * redeclared: the same three the chat composer sends, the teacher files by and the
 * SFT corpus is tagged with — one vocabulary across the whole system, so a benchmark
 * at `high` is the same request as a chat turn at `high`, and two identical unions
 * would be two places for that to stop being true.
 */
export type { Tier } from "./jobs"

/** Sweep order: least thinking to most. Not alphabetical, and not the enum order. */
export const TIER_ORDER: Tier[] = ["low", "med", "high"]

/**
 * Below this many held-out conversations a score is drawn as provisional.
 *
 * Not a statistical threshold so much as an honesty one: the high tier currently has
 * three conversations behind it, and its error bar is several times the differences a
 * sweep is looking for. A point that thin must not look like one resting on fifteen.
 */
export const MIN_CONVERSATIONS = 10

/** Which suites a thinking-budget sweep can move. See `EvalConfig::budget`. */
export function suiteTakesBudget(suite: BenchmarkSuite): boolean {
  return suite !== "language"
}

export const BENCHMARK_SUITES: {
  id: BenchmarkSuite
  label: string
  blurb: string
}[] = [
  {
    id: "language",
    label: "Language",
    blurb: "Held-out text, grammar, and generated prose.",
  },
  {
    id: "markup",
    label: "Markup",
    blurb: "Special-token use: pairs, placement, and structure.",
  },
  {
    id: "assistant",
    label: "Assistant",
    blurb:
      "Held-out replies: how many bits the model spends per token of answer.",
  },
]

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

/**
 * The score with its parts shown.
 *
 * `score_stderr` is one standard error of the score as an estimate of the model's
 * ability, from the finite pair and word samples. It is the number to compare a
 * run-to-run delta against: a change smaller than this is the benchmark's sampling
 * noise, not the model.
 */
export interface ScoreBreakdown {
  version: number
  language: number | null
  grammar: number | null
  words: number | null
  variety_gate: number | null
  score: number | null
  score_stderr: number | null
}

export interface BenchmarkReport {
  score: number | null
  breakdown: ScoreBreakdown
  model: ModelInfo
  corpus: CorpusMetrics | null
  grammar: GrammarMetrics
  generation: GenerationMetrics
  // Agreement with the replies you rated good and bad. Reported, never scored.
  preference: PreferenceMetrics | null
  duration_secs: number
  // Tokens the model forwarded to produce this report — the run's cost in the unit
  // the model works in. Lifted onto the run row as `eval_tokens`, which is what the
  // charts read; here it is the report's own copy.
  tokens_used: number
}

// Thumbs collected in the chat UI, scored. Absent until some have been given.
export interface PreferenceMetrics {
  approved_logprob: number | null
  rejected_logprob: number | null
  margin: number | null
  approved: number
  rejected: number
  paired_accuracy: number | null
  pairs: number
}

// ── the markup report ───────────────────────────────────────────────────────

export interface MarkupParadigmResult {
  name: string
  description: string
  accuracy: number | null
  pairs: number
}

export interface MarkupPairMetrics {
  accuracy: number | null
  pairs: number
  paradigms: MarkupParadigmResult[]
}

export interface MarkupPositionMetrics {
  marker_top1: number | null
  marker_restricted_top1: number | null
  marker_mean_rank: number | null
  marker_logprob: number | null
  marker_positions: number
  false_marker_rate: number | null
  prose_positions: number
  conversations: number
}

export interface MarkupStructureMetrics {
  close_rate: number | null
  balance_rate: number | null
  marker_density: number | null
  reference_marker_density: number | null
  scaffold_leak_rate: number | null
  replies: number
  tokens_generated: number
  samples: GenerationSample[]
}

export interface MarkupMetrics {
  pairs: MarkupPairMetrics
  // Both null until the held-out chat set exists. The pair stage still scores.
  positions: MarkupPositionMetrics | null
  structure: MarkupStructureMetrics | null
}

export interface MarkupBreakdown {
  version: number
  pairs: number | null
  positions: number | null
  structure: number | null
  restraint: number | null
  score: number | null
  score_stderr: number | null
}

export interface MarkupReport {
  score: number | null
  breakdown: MarkupBreakdown
  model: ModelInfo
  markup: MarkupMetrics
  chat: ChatMetrics | null
  duration_secs: number
  tokens_used: number
}

// ── the assistant report ────────────────────────────────────────────────────

export interface AssistantBreakdown {
  version: number
  replies: number | null
  score: number | null
  score_stderr: number | null
}

/**
 * Bits per assistant-reply token over the held-out conversations, mapped to 0–100.
 *
 * **The score inverts the measurement.** Bits per token is a *cost* — lower is
 * better — while every suite reports higher-is-better, so `chat.bits_per_token`
 * falling is `score` rising. Read the score for direction and the bits for
 * magnitude; showing the raw bits without saying which way is up is how a
 * 3.6875 → 3.8796 regression once got read as progress.
 */
export interface AssistantReport {
  score: number | null
  breakdown: AssistantBreakdown
  model: ModelInfo
  chat: ChatMetrics | null
  duration_secs: number
  tokens_used: number
}

/** A run's report is one of three, told apart by the run's `suite`. */
export type AnyReport = BenchmarkReport | MarkupReport | AssistantReport

export function isMarkupReport(report: AnyReport): report is MarkupReport {
  return "markup" in report
}

// Discriminated on the one field only this report has. `chat` will not do — the
// markup report carries one too.
export function isAssistantReport(
  report: AnyReport
): report is AssistantReport {
  return "breakdown" in report && "replies" in report.breakdown
}

/**
 * Positive rather than `!isMarkupReport(...)`.
 *
 * With two suites, "not markup" was a sound way to reach the language report. With
 * three it silently includes the assistant one, which has none of the fields the
 * language panels read — so the narrowing has to name what it wants.
 */
export function isLanguageReport(report: AnyReport): report is BenchmarkReport {
  return "grammar" in report
}

export interface BenchmarkRun {
  id_run: string
  id_model: string
  model_name: string
  status: BenchmarkStatus
  suite: BenchmarkSuite
  /**
   * Which revision of the suite produced this score. Bumped whenever a scoring
   * constant, the pair set, the prompt set or the corpus split changes — all of which
   * move the number without the model having changed. Runs of different versions are
   * not comparable and must not share a trend line.
   */
  suite_version: number
  score: number | null
  error: string | null
  // Tokens in the corpus the model was trained on, snapshotted when the run
  // started (so retraining the model doesn't relabel past runs). A property of the
  // model rather than of this run, and nothing plots it any more — the charts ask
  // what the measurement cost, which is `eval_tokens`.
  train_tokens: number | null
  // Tokens the model forwarded to complete the run: the cost of the benchmark in
  // the unit the model works in, and the x of every score chart. `null` for a run
  // older than the counter — those are left off rather than plotted at zero.
  eval_tokens: number | null
  /**
   * The thinking budget every held-out conversation was rendered under. `null` is
   * not a fourth value: it means each conversation kept the budget it was written
   * with, which is every run before the sweep existed and every run with the switch
   * off.
   */
  think_budget: Tier | null
  /**
   * The sweep this run belongs to. Three runs queued by the split switch share one,
   * which is what the chart groups on to draw them as a connected line. `null` for a
   * run queued on its own — a series of one, drawn as a lone dot.
   */
  id_series: string | null
  /**
   * Held-out conversations scored. What a budget sweep varies over, and small enough
   * to matter — the current holdout is 15 low, 15 med, 3 high — so the chart marks a
   * point below `MIN_CONVERSATIONS` as provisional. `null` on a language run, which
   * scores no conversations at all.
   */
  eval_conversations: number | null
  created_at: string
  completed_at: string | null
}

export interface BenchmarkRunDetail extends BenchmarkRun {
  report: AnyReport | null
}

// Whether the reference corpus is installed. Surfaced on the page so the absence
// of perplexity numbers reads as "no corpus installed" rather than as a bug.
export interface SuiteInfo {
  corpus_dir: string
  corpus_files: number
  blimp_dir: string
  blimp_files: number
  chat_dir: string
  // Held-out conversations, which the markup suite's position and structure stages
  // need. Written from the 15% eval share at the start of every benchmark, so this
  // reads 0 only before the first run on a fresh install.
  chat_files: number
  // Pairs in the built-in marker suite. Templated in code, not a file count.
  markup_pairs: number
}

export async function listRuns(suite: BenchmarkSuite): Promise<BenchmarkRun[]> {
  const res = await apiFetch(`/benchmark/runs?suite=${suite}`)
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

export async function getModelRuns(
  idModel: string,
  suite: BenchmarkSuite = "language"
): Promise<ModelBenchmarks> {
  const res = await apiFetch(`/benchmark/models/${idModel}/runs?suite=${suite}`)
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

// ── the suite, as a running order ───────────────────────────────────────────

/**
 * The five sections a run scores, in the order it scores them, with the share of
 * the composite each one carries. Mirrors the weights in `crates/llm/src/eval/mod.rs`
 * — if those move, these have to move with them.
 *
 * This exists so a benchmark that has not finished still has something true to
 * show. A percentage alone says how far along it is; the running order says what
 * it is doing and what is still to come, which is the difference between a
 * progress bar and knowing whether the number you are waiting for is close.
 */
export interface BenchmarkSection {
  name: string
  /** Share of the composite score, 0–1. Doubles as this section's slice of progress. */
  weight: number
  /** What it measures, in a line. */
  blurb: string
}

const LANGUAGE_SECTIONS: BenchmarkSection[] = [
  {
    name: "Held-out text",
    weight: 0.36,
    blurb: "Perplexity on text the model never trained on.",
  },
  {
    name: "Grammar",
    weight: 0.4,
    blurb: "Minimal pairs: does it prefer the well-formed sentence?",
  },
  {
    name: "Generation",
    weight: 0.13,
    blurb: "Twelve prompts continued, scored on surface quality.",
  },
  {
    name: "Preference",
    weight: 0.11,
    blurb: "Agreement with the replies you rated good and bad.",
  },
]

const MARKUP_SECTIONS: BenchmarkSection[] = [
  {
    name: "Marker pairs",
    weight: 0.55,
    blurb: "Forced choice: does it prefer well-formed markup?",
  },
  {
    name: "Marker positions",
    weight: 0.2,
    blurb: "At each marker in a held-out reply, would it emit that one?",
  },
  {
    name: "Structure",
    weight: 0.2,
    blurb: "Generated replies: do they close and nest correctly?",
  },
  {
    name: "Reply bits",
    weight: 0.05,
    blurb: "Bits per assistant token, reported but not scored.",
  },
]

export function benchmarkSections(suite: BenchmarkSuite): BenchmarkSection[] {
  return suite === "markup" ? MARKUP_SECTIONS : LANGUAGE_SECTIONS
}

/**
 * Which section a fraction of overall progress falls in.
 *
 * The runner reports one number for the whole suite, so the section boundaries are
 * reconstructed here from the same weights it advances by.
 */
export function sectionIndex(
  fraction: number,
  suite: BenchmarkSuite = "language"
): number {
  const sections = benchmarkSections(suite)
  let base = 0
  for (let i = 0; i < sections.length; i++) {
    const weight = sections[i].weight
    if (fraction < base + weight) return i
    base += weight
  }
  return sections.length - 1
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

/**
 * The fewest decimals that print every one of these values exactly.
 *
 * Not derived from the gap, which is the obvious rule and a wrong one: the round-step
 * ladder includes 2.5, so a 0.25 step needs two decimals while `-log10(0.25)` asks for
 * one — and 56.5 · 56.8 · 57.0 · 57.3 is a ladder whose labels imply gaps of 0.3 and
 * 0.2 alternating. The axis then reads as uneven when it is perfectly even, which is
 * worse than a decimal too many.
 */
export function exactDecimals(values: number[], max = 3): number {
  for (let digits = 0; digits < max; digits++) {
    const scale = 10 ** digits
    if (
      values.every((v) => Math.abs(v * scale - Math.round(v * scale)) < 1e-9)
    ) {
      return digits
    }
  }
  return max
}

/**
 * A compact formatter carrying enough decimals to tell *these* ticks apart.
 *
 * `formatCompact` rounds to one decimal, which is right for an axis spanning decades
 * and wrong for one spanning a fraction of a percent: five ticks fifty tokens apart all
 * print as "308.6K", and an axis whose labels are five copies of one number is worse
 * than no axis, because it reads as a bug rather than as a tight cluster.
 *
 * So the decimals come from the gap between adjacent ticks, and the unit is fixed
 * across all of them — mixing "980K" with "1.0M" on one axis hides that they are
 * neighbours. Both only apply within a decade; a wider axis is exactly what
 * `formatCompact` is already good at, and keeps it.
 */
export function compactAxisFormat(ticks: number[]): (value: number) => string {
  const values = ticks.filter((t) => Number.isFinite(t) && t > 0)
  if (values.length < 2) return formatCompact

  const max = Math.max(...values)
  const min = Math.min(...values)
  if (max / min >= 10) return formatCompact

  const [unit, suffix] =
    max >= 1e9
      ? [1e9, "B"]
      : max >= 1e6
        ? [1e6, "M"]
        : max >= 1e3
          ? [1e3, "K"]
          : [1, ""]

  // Enough decimals to print every tick exactly in its own unit, capped: past three
  // the label is longer than the difference it is describing is worth.
  const digits = exactDecimals(
    values.map((v) => v / unit),
    3
  )
  return (value: number) => `${(value / unit).toFixed(digits)}${suffix}`
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
