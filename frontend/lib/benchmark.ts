import { apiFetch } from "./api"
import type { Tier } from "./jobs"

// Wire types for the benchmark API. The report is
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

/**
 * Which model a report describes.
 *
 * `id` is the weights directory and what an API call names it by; `name` is
 * what a person calls it, filled in by the engine's handler because the
 * pipeline that measured the model has never heard of the row.
 */
export interface ReportModel {
  id: string
  name: string | null
  layers: number
  dimensions: number
  heads: number
  context: number
  vocab: number
  parameters: number
}

/**
 * Every suite's score, with the components that produced it.
 *
 * `score` is the 0–100 composite; the rest are the 0–1 components behind it,
 * and a component that could not be measured is `null` rather than 0 — its
 * weight redistributes across the ones that ran. `version` is what makes two
 * scores comparable: a scoring change moves the number without the model
 * having changed, so runs of different versions must not share a trend line.
 */
interface ScoreBase {
  version: number
  score: number | null
  score_stderr: number | null
}

export interface LanguageScore extends ScoreBase {
  language: number | null
  /** Null throughout: the minimal-pair suite is not ported yet. */
  grammar: number | null
  words: number | null
  /** Zeroes the score for a model whose samples are too repetitive to read. */
  variety_gate: number | null
}

export interface MarkupScore extends ScoreBase {
  pairs: number | null
  positions: number | null
  structure: number | null
  restraint: number | null
  pair_count: number
}

export interface AssistantScore extends ScoreBase {
  replies: number | null
}

export interface CorpusMetrics {
  bits_per_byte: number | null
  bits_per_token: number | null
  perplexity: number | null
  tokens_scored: number
  baseline_bits_per_byte: number | null
  /** Fraction of the n-gram floor's bits removed. A verdict, not a scale. */
  advantage: number | null
  /** Nothing predicted (0) to matching the floor (1). The scored component. */
  floor_progress: number | null
}

export interface SurfaceMetrics {
  words: number
  real_word_rate: number | null
  distinct_1: number | null
  distinct_2: number | null
  distinct_3: number | null
  repeat_3: number | null
  mean_sample_words: number | null
}

export interface Sample {
  prompt: string
  text: string
}

/** One sublayer silenced, and what the model lost by it. */
export interface AblationUnit {
  unit: string
  kind: "attention" | "ffn"
  layer: number
  cross_entropy: number
  ce_delta: number
  kl_per_query: number
  top1_agreement: number
}

/**
 * What each sublayer contributes, from silencing them one at a time.
 *
 * `learned_nats` is how far below a uniform guess the intact model sits. Near
 * zero means the table is measuring noise.
 */
export interface AblationReport {
  layers: number
  window_tokens: number
  intact_ce: number
  chance_ce: number
  learned_nats: number
  units: AblationUnit[]
}

export interface PositionMetrics {
  marker_top1: number | null
  marker_restricted_top1: number | null
  marker_mean_rank: number | null
  marker_logprob: number | null
  marker_positions: number
  /** How often a marker was wanted where the reference has prose. */
  false_marker_rate: number | null
  prose_positions: number
  conversations: number
}

export interface StructureMetrics {
  close_rate: number | null
  balance_rate: number | null
  marker_density: number | null
  scaffold_leak_rate: number | null
  replies: number
  tokens_generated: number
}

/**
 * Bits per assistant-reply *byte* over held-out conversations, through the same
 * loss mask SFT trains on.
 *
 * `bits_per_byte` is the scored figure. `bits_per_token` is reported alongside
 * but only comparable between models sharing a tokenizer — the BPE fix moved
 * the vocabulary from 2.13 to 3.79 bytes per token and every per-token figure
 * on record moved ~74% with it, on unchanged weights.
 */
export interface ChatMetrics {
  bits_per_byte: number | null
  bits_per_byte_stderr: number | null
  bits_per_token: number | null
  logprob: number | null
  bytes_scored: number
  tokens_scored: number
  conversations: number
}

interface ReportBase {
  model: ReportModel
  total_secs: number
}

export interface LanguageReport extends ReportBase {
  suite: "language"
  score: LanguageScore
  corpus: CorpusMetrics | null
  surface: SurfaceMetrics | null
  ablation: AblationReport | null
  samples: Sample[]
  /** True when the run stopped at a stage boundary; later stages are absent. */
  cancelled: boolean
}

export interface MarkupReport extends ReportBase {
  suite: "markup"
  score: MarkupScore
  positions: PositionMetrics | null
  structure: StructureMetrics | null
  conversations: number
}

export interface AssistantReport extends ReportBase {
  suite: "assistant"
  score: AssistantScore
  chat: ChatMetrics | null
  conversations: number
}

/** A run's report is one of three, told apart by the `suite` it carries. */
export type AnyReport = LanguageReport | MarkupReport | AssistantReport

export function isMarkupReport(report: AnyReport): report is MarkupReport {
  return report.suite === "markup"
}

export function isAssistantReport(
  report: AnyReport
): report is AssistantReport {
  return report.suite === "assistant"
}

export function isLanguageReport(report: AnyReport): report is LanguageReport {
  return report.suite === "language"
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
  chat_dir: string
  // Held-out conversations, which the markup suite's position and structure stages
  // need. Written from the 15% eval share at the start of every benchmark, so this
  // reads 0 only before the first run on a fresh install.
  chat_files: number
}

/**
 * The wire shape of a run. `think_budget` and `id_series` are absent: the budget
 * sweep is parked with the data module that will rebuild the corpus it varies
 * over, and null is already what the charts read as "no budget was forced".
 */
interface BenchmarkRow extends Omit<BenchmarkRun, "id_run" | "id_model"> {
  benchmark_id: string
  model_id: string
}

function fromRow(row: BenchmarkRow): BenchmarkRun {
  const { benchmark_id, model_id, ...rest } = row
  return {
    ...rest,
    id_run: benchmark_id,
    id_model: model_id,
    think_budget: null,
    id_series: null,
  }
}

export async function listRuns(suite: BenchmarkSuite): Promise<BenchmarkRun[]> {
  const res = await apiFetch(`/benchmarks?suite=${suite}`)
  if (!res.ok) throw new Error("Failed to load benchmark history.")
  const { benchmarks } = (await res.json()) as { benchmarks: BenchmarkRow[] }
  return benchmarks.map(fromRow)
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
  const res = await apiFetch(`/models/${idModel}/benchmarks`)
  if (!res.ok) throw new Error("Failed to load this model's benchmarks.")
  const { benchmarks } = (await res.json()) as { benchmarks: BenchmarkRow[] }
  const runs = benchmarks.map(fromRow).filter((run) => run.suite === suite)
  // The most recent run that produced a report — deliberately not the most
  // recent run, which may be a failure with nothing in it to show.
  const scored = runs.find((run) => run.status === "done" && run.score !== null)
  return {
    latest: scored ? await getRun(scored.id_run) : null,
    runs,
  }
}

export async function getRun(id: string): Promise<BenchmarkRunDetail> {
  const res = await apiFetch(`/benchmarks/${id}`)
  if (!res.ok) throw new Error("Failed to load benchmark run.")
  const row = (await res.json()) as BenchmarkRow & { report: AnyReport | null }
  return { ...fromRow(row), report: row.report }
}

export async function getSuite(): Promise<SuiteInfo> {
  const res = await apiFetch("/benchmarks/suite")
  if (!res.ok) throw new Error("Failed to load suite info.")
  return res.json()
}

export async function deleteRun(id: string): Promise<void> {
  const res = await apiFetch(`/benchmarks/${id}`, { method: "DELETE" })
  if (!res.ok) throw new Error("Failed to delete benchmark run.")
}

// ── the suite, as a running order ───────────────────────────────────────────

/**
 * The stages a run scores, in order, with the share of the composite each
 * carries. Mirrors the weights in `engine/llm/eval/score.py` and `markup.py` —
 * if those move, these have to move with them.
 *
 * This exists so a benchmark that has not finished still has something true to
 * show. A percentage says how far along it is; the running order says what it is
 * doing and what is still to come.
 */
export interface BenchmarkSection {
  name: string
  /** Share of the composite score, 0–1. Doubles as this section's slice of progress. */
  weight: number
  /** What it measures, in a line. */
  blurb: string
}

// Grammar's 0.40 is listed because the score still reserves it; the minimal-pair
// suite is not ported, so its weight redistributes across the stages that ran.
const LANGUAGE_SECTIONS: BenchmarkSection[] = [
  {
    name: "Held-out text",
    weight: 0.45,
    blurb: "Compression of text the model never trained on.",
  },
  {
    name: "Grammar",
    weight: 0.4,
    blurb: "Minimal pairs — not ported yet, so its weight redistributes.",
  },
  {
    name: "Generation",
    weight: 0.15,
    blurb: "Fixed prompts continued, scored on surface quality.",
  },
]

const MARKUP_SECTIONS: BenchmarkSection[] = [
  {
    name: "Marker pairs",
    weight: 0.4,
    blurb: "Forced choice — not ported yet, so its weight redistributes.",
  },
  {
    name: "Marker positions",
    weight: 0.3,
    blurb: "At each marker in a held-out reply, would it emit that one?",
  },
  {
    name: "Structure",
    weight: 0.2,
    blurb: "Generated replies: do they close and nest correctly?",
  },
  {
    name: "Restraint",
    weight: 0.1,
    blurb: "Markers where the reference has ordinary prose.",
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
