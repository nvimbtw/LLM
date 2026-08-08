import { apiFetch } from "./api"
import type { BenchmarkSuite } from "./benchmark"
import type { TrainConfig } from "./training"

// Wire types for the GPU queue (backend routes/jobs.rs).
//
// Everything that uses the device — a training run, a benchmark, a chat turn — is
// a row in `jobs`, executed one at a time in queue order. The board is the whole
// of what the /jobs page renders, and `config` travels with each row so a finished
// job can seed a new one without a second request.

export type JobKind =
  | "train"
  | "benchmark"
  | "chat"
  | "synth"

/**
 * Which runner a job stands in front of. `device` is the GPU line — one job at a
 * time, and everything that touches the model is in it. `network` is its own line
 * for work that only waits on somebody else's API, so a generation run never
 * occupies the device it does not use.
 */
export type JobLane = "device" | "network"

export type JobStatus =
  // Queued behind another job and not yet runnable: it consumes the model that job
  // will produce, and that model does not exist yet. It becomes `queued` on its own
  // when the job named by `depends_on` finishes — see the backend's `jobs::chain`.
  | "waiting"
  | "queued"
  | "running"
  // Cancel requested, worker not yet at a step boundary where it can stop. A real
  // interval, not a client-side fiction, so it gets its own state.
  | "cancelling"
  | "done"
  | "cancelled"
  | "failed"

export interface Job {
  id_job: string
  kind: JobKind
  lane: JobLane
  status: JobStatus
  label: string
  // The job's own config: a TrainConfig, a { id_model }, or a chat request. Typed
  // as unknown because the shape depends on `kind`; the readers below narrow it.
  config: unknown
  progress: number
  stage: string | null
  // What the job produced: a model id, a benchmark run id, or a message id.
  ref_id: string | null
  /**
   * What the run produced, for a kind with no table of its own to read back from —
   * a teaching run's stats, today. It is why a finished teaching run still draws its
   * figures: the socket frame that carried them only reached whoever was watching
   * live. Null for every other kind.
   */
  result: unknown
  /**
   * The job this one runs after, for a `waiting` row. Null for every ordinary job.
   * The board draws a chain from it: a follow-up is not "queued in position 4", it
   * is "after that one", and a rank would be a lie about when it runs.
   */
  depends_on: string | null
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface JobBoard {
  // Running and queued, in the order they will execute.
  active: Job[]
  // One page of finished jobs, newest first — the slice named by the three fields
  // below. The live band is never paged: it is short by construction.
  history: Job[]
  historyOffset: number
  historyLimit: number
  historyTotal: number
  paused: boolean
}

/** The wire shape of the board, before the history fields are camel-cased. */
interface BoardPayload {
  active: Job[]
  history: Job[]
  history_offset: number
  history_limit: number
  history_total: number
  paused: boolean
}

export const TERMINAL: JobStatus[] = ["done", "cancelled", "failed"]

export function isTerminal(job: Job): boolean {
  return TERMINAL.includes(job.status)
}

export function isRunning(job: Job): boolean {
  return job.status === "running" || job.status === "cancelling"
}

// ── reading a job's config ──────────────────────────────────────────────────

/** A train job's config, or null when this isn't one. */
export function trainConfig(job: Job): TrainConfig | null {
  if (job.kind !== "train") return null
  return job.config as TrainConfig
}

/** The model a benchmark job scores. */
export function benchmarkModelId(job: Job): string | null {
  if (job.kind !== "benchmark") return null
  const config = job.config as { id_model?: string }
  return config?.id_model ?? null
}

/**
 * Which benchmark a benchmark job runs.
 *
 * Absent on rows queued before the suites were split, which all meant the language
 * one — so the fallback is not a guess, it is what those jobs did.
 */
export function benchmarkSuiteOf(job: Job): BenchmarkSuite {
  const config = job.config as { suite?: BenchmarkSuite }
  if (config?.suite === "markup") return "markup"
  if (config?.suite === "assistant") return "assistant"
  return "language"
}

/**
 * The forced thinking budget a benchmark job was queued with, if any.
 *
 * Null is the unbudgeted run — each held-out conversation keeps the marker it was
 * written with — which is what every benchmark queued before the sweep existed asked
 * for, and what the switch being off still means.
 */
export function benchmarkBudgetOf(job: Job): Tier | null {
  if (job.kind !== "benchmark") return null
  const config = job.config as { budget?: Tier | null }
  return config?.budget ?? null
}

/** The chat a chat job replies in, so the row can link back to the conversation. */
export function chatId(job: Job): string | null {
  if (job.kind !== "chat") return null
  const config = job.config as { id_chat?: string }
  return config?.id_chat ?? null
}

/**
 * Where a job's row should link to.
 *
 * Everything gets its own page — except a chat turn, which is never queued from
 * the jobs menu in the first place: it is a reply inside a conversation you are
 * already reading, and the conversation is where the reply belongs. Sending you to
 * /jobs/{id} for one is a detour to a page whose only useful control is a link
 * back, so the row goes straight to the chat instead.
 *
 * A chat job with no conversation on its config (older rows) still falls back to
 * the job page rather than to a broken link.
 */
export function jobHref(job: Job): string {
  const chat = chatId(job)
  return chat ? `/chat/${chat}` : `/jobs/${job.id_job}`
}

// ── requests ────────────────────────────────────────────────────────────────

/** How many finished jobs a history page holds. Mirrors the server's default. */
export const HISTORY_PAGE_SIZE = 25

export async function getBoard(offset = 0): Promise<JobBoard> {
  const params = new URLSearchParams({
    history_offset: String(Math.max(0, offset)),
    history_limit: String(HISTORY_PAGE_SIZE),
  })
  const res = await apiFetch(`/jobs?${params}`)
  if (!res.ok) throw new Error("Failed to load the queue.")
  const payload: BoardPayload = await res.json()
  return {
    active: payload.active,
    history: payload.history,
    historyOffset: payload.history_offset,
    historyLimit: payload.history_limit,
    historyTotal: payload.history_total,
    paused: payload.paused,
  }
}

export async function getJob(id: string): Promise<Job> {
  const res = await apiFetch(`/jobs/${id}`)
  if (!res.ok) throw new Error("Failed to load the job.")
  return res.json()
}

/**
 * A job to run *after* the one it hangs off, against the model that one produces.
 *
 * Recursive, and that recursion is what makes a chain a chain: `then` on a step is
 * "and after *that* one". The composers build a flat list and nest it on the way
 * out (see `queueChain`), because a flat list is what "then, then, then" means to
 * a person, and nesting is what it means to the server.
 *
 * Only `train` and `benchmark` can be follow-ups — they are the two kinds that can
 * take their model from the job before them (`base_model_id` and `id_model`
 * respectively, filled in server-side at promotion). The config carries **no**
 * model: naming one is a 400, because it would be two answers to the one question
 * a follow-up exists to leave open.
 */
export interface FollowUp {
  kind: "train" | "benchmark"
  config: unknown
  then?: FollowUp[]
}

async function enqueue(
  kind: JobKind,
  config: unknown,
  then?: FollowUp[]
): Promise<Job> {
  const res = await apiFetch("/jobs", {
    method: "POST",
    body: JSON.stringify(
      then && then.length > 0 ? { kind, config, then } : { kind, config }
    ),
  })
  if (!res.ok) {
    // A 400 here carries the specific validation failure (a bad hyperparameter, a
    // missing dataset, a model with no tokenizer), which is exactly what the
    // composer needs to show — so it is surfaced verbatim.
    throw new Error((await res.text()) || "Failed to queue the job.")
  }
  return res.json()
}

export function queueTraining(
  config: TrainConfig,
  steps: FollowUp[] = []
): Promise<Job> {
  return enqueue("train", config, nest(steps))
}

/**
 * Turns the composer's flat list of steps into the nested chain the server wants.
 *
 * The distinction matters and is easy to get backwards. Sent flat, all three steps
 * would hang off the *root* — so a benchmark listed after a fine-tune would score
 * the model the root produced, not the fine-tune's. Nested, each step follows the
 * one above it, which is what "then" says on the screen.
 *
 * Fan-out (two follow-ups off one job) is expressible in the wire format and
 * deliberately not in this UI: one list reads as one sequence, and a tree drawn as
 * a list is a tree somebody will misread.
 */
export function nest(steps: FollowUp[]): FollowUp[] {
  if (steps.length === 0) return []
  const [head, ...rest] = steps
  const nested = nest(rest)
  return [nested.length > 0 ? { ...head, then: nested } : head]
}

/**
 * Queues a benchmark, or a whole thinking-budget sweep.
 *
 * With `splitThinking` the backend queues **three** jobs — one per budget, sharing an
 * `id_series` — and returns the first. The other two are on the board; the series id
 * is what ties them together on the chart. Refused by the server for the language
 * suite, whose stages render no chat prompt for a marker to attach to.
 */
export function queueBenchmark(
  modelId: string,
  suite: BenchmarkSuite = "language",
  splitThinking = false,
  steps: FollowUp[] = []
): Promise<Job> {
  return enqueue(
    "benchmark",
    {
      id_model: modelId,
      suite,
      split_thinking: splitThinking,
    },
    nest(steps)
  )
}



/**
 * What an auto-labelling run did. Mirrors the backend `AutolabelStats`.
 *
 * `calibration` is the part worth reading: it is the judge measured against the
 * replies *you* rated, before it was allowed to label anything. An `auc` of 0.5 is
 * a coin flip and the run refuses; 1.0 is perfect agreement with your thumbs.
 */
export interface Calibration {
  auc: number
  threshold: number
  separation: number
  approved: number
  rejected: number
}

/** One auto-labelled reply, as it is decided. */
export interface Verdict {
  prompt: string
  response: string
  good: boolean
  score: number
}

export interface CatalogueReport {
  prompts: number
  calibration: Calibration
  examples_written: number
  approved: number
  rejected: number
  skipped_uncertain: number
  mean_score: number
}

/** Config for a synthetic SFT generation run. Mirrors the backend `SynthRequest`. */
export interface SynthConfig {
  /** An OpenAI-compatible API base, e.g. https://openrouter.ai/api/v1. */
  base_url: string
  /** The provider's model id — the one writing the conversations, not ours. */
  model: string
  /** How many valid conversations to add before stopping. */
  count: number
  /** Conversations per call. Small: a long batch arrives truncated. */
  batch?: number
  temperature?: number
  max_tokens?: number
  /** Thinking blocks shorter than this are rejected. */
  min_think_words?: number
  think_frac?: number
  system_frac?: number
  timeout_secs?: number
  max_calls?: number
  /**
   * Our model, whose tokenizer and context window a conversation has to fit.
   * Without it nothing checks length, which is how a corpus of conversations three
   * times too long to train on gets written.
   */
  id_model?: string
  /** Name of the server-side env var holding the key. Never the key itself. */
  api_key_env?: string
  exemplar_file?: string
  out_file?: string
  seed?: number
}

export function queueSynth(config: SynthConfig): Promise<Job> {
  return enqueue("synth", config)
}

/** One model call in a generation run, as it completes. */
export interface SynthCall {
  call: number
  kept: number
  duplicates: number
  /** [reason, count] for everything this call's output lost. */
  rejected: [string, number][]
  elapsed_ms: number
  response_chars: number
  total: number
  target: number
}

/** One conversation that survived validation. */
export interface SynthAccepted {
  opener: string
  reply: string
  turns: number
  think_words: number
  /** Rendered length in real tokens, when a base model was named. */
  tokens: number | null
  user_chars: number
  assistant_chars: number
}

/** What a generation run did, emitted once at the end. */
export interface SynthReport {
  added: number
  calls: number
  duplicates: number
  with_thinking: number
  rejected: [string, number][]
  out_file: string
  corpus_total: number
}

// ── teaching: generation from a local model ─────────────────────────────────

/** The three thinking budgets. Mirrors the backend `synth::teach::Tier`. */
export type Tier = "low" | "med" | "high"

export const TIER_LABEL: Record<Tier, string> = {
  low: "Short",
  med: "Medium",
  high: "Long",
}

/** Which GGUF to run, and how much of the device to give it. */
export interface TeacherConfig {
  /** A bare file name inside the teacher directory, never a path. */
  model_file: string
  /**
   * Context window to allocate. The binding VRAM cost after the weights: the KV
   * cache scales linearly with it, so this is a real allocation rather than a free
   * upper bound.
   */
  n_ctx?: number
  /** Layers to offload. Larger than any small model has, meaning "all of them". */
  n_gpu_layers?: number
  n_threads?: number
}

/** How to sample. Mirrors the backend `teacher::SampleOpts`. */
export interface SampleOpts {
  temperature: number
  top_p: number
  min_p: number
  top_k: number
  max_tokens: number
  seed: number
  /**
   * DRY repetition penalty. Not a flat repetition penalty on purpose: that pushes
   * down every token already seen, which on our markup means fighting the closing
   * tags. DRY penalizes repeated *sequences*, so it suppresses a second answer that
   * merely rewords the first and leaves `<|/codeblock|>` alone.
   */
  dry_multiplier?: number
  dry_base?: number
  dry_allowed_length?: number
}

/** Config for a local-teacher run. Mirrors the backend `TeachRequest`. */
export interface TeachConfig {
  teacher: TeacherConfig
  sample?: SampleOpts
  /** Triples to accept before stopping. A triple is one question, three answers. */
  count: number
  /**
   * Upper bound on thinking tokens for the short tier, and for the medium one.
   * Anything above the second is long. Measured in *our* model's tokenizer, so
   * these do nothing without `id_model`.
   */
  low_max_tokens?: number
  med_max_tokens?: number
  /** Our model, whose tokenizer measures the budget and whose window bounds it. */
  id_model?: string
  exemplar_file?: string
  max_calls?: number
  seed?: number
}


/** One model call in a teaching run, as it completes. */
export interface TeachCall {
  call: number
  /** Conversations this call wrote: 0 to 3, one per surviving answer. */
  answers_kept: number
  /** How many of the three tiers those answers covered. 3 is a real triple. */
  tiers_covered: number
  duplicates: number
  rejected: [string, number][]
  elapsed_ms: number
  tokens_generated: number
  /**
   * Running totals in the three units the run has. `total_questions` is what
   * `target` bounds; the tier bars add up to `total_answers`. Reporting only the
   * first of these is what made "45 triples" sit beside bars summing to 75.
   */
  total_questions: number
  total_answers: number
  total_complete: number
  /** Questions by tiers covered: `[one tier, two tiers, all three]`. */
  total_by_tiers: number[]
  target: number
}

/** One tier's answer within an accepted triple. */
export interface AnsweredTier {
  requested: Tier
  /**
   * Where it was actually filed. Assigned by measuring the thinking block, not by
   * what the prompt asked for — a teacher told to write 40 words routinely writes
   * 70, and the label has to describe the content to be trainable.
   */
  filed_as: Tier
  think_tokens: number
  reply_preview: string
}

/** One question and the answers that survived for it — up to three, often fewer. */
export interface TeachAccepted {
  group: string
  opener: string
  answers: AnsweredTier[]
}

/** What a teaching run did, emitted once at the end. */
export interface TeachReport {
  /** Questions with at least one answer kept. What `count` bounds and stops on. */
  questions: number
  /** Questions whose answers covered all three tiers — actual triples. */
  complete_triples: number
  /** Conversations written; the sum of `per_tier`, in the bars' own unit. */
  answers: number
  /**
   * Questions by how many distinct budgets they covered: `[one, two, all three]`.
   * Sums to `questions`; its last entry is `complete_triples`. The *shape* is the
   * diagnosis — a run that is mostly one-tier questions is failing differently from
   * one that is mostly two-tier, and the second is a nudge away from working.
   */
  questions_by_tiers: number[]
  calls: number
  duplicates: number
  rejected: [string, number][]
  /** Answers filed per tier, `[tier, count]`. The headline number of a run. */
  per_tier: [string, number][]
  /** How often the measured tier disagreed with the requested one. */
  relabelled: number
  /** False when no model was named, so tiers are requested rather than measured. */
  measured: boolean
  /** False when the build could not offload — a correct run at ~1% the speed. */
  gpu: boolean
}

/**
 * Moves a queued job to sit directly after `afterId`, or to the front when that is
 * null. Expressed as a neighbour rather than an index so a drop lands where it was
 * dropped even if the board shifted underneath the drag.
 */
export async function reorderJob(
  id: string,
  afterId: string | null
): Promise<void> {
  const res = await apiFetch(`/jobs/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ after: afterId }),
  })
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to reorder the job.")
}

export async function cancelJob(id: string): Promise<void> {
  const res = await apiFetch(`/jobs/${id}/cancel`, { method: "POST" })
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to cancel the job.")
}

export async function retryJob(id: string): Promise<Job> {
  const res = await apiFetch(`/jobs/${id}/retry`, { method: "POST" })
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to re-run the job.")
  return res.json()
}

export async function deleteJob(id: string): Promise<void> {
  const res = await apiFetch(`/jobs/${id}`, { method: "DELETE" })
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to delete the job.")
}

export async function setPaused(paused: boolean): Promise<void> {
  const res = await apiFetch("/jobs/pause", {
    method: "POST",
    body: JSON.stringify({ paused }),
  })
  if (!res.ok) throw new Error("Failed to update the queue.")
}

// ── presentation ────────────────────────────────────────────────────────────

export const KIND_LABEL: Record<JobKind, string> = {
  train: "Train",
  benchmark: "Benchmark",
  chat: "Chat",
  synth: "Generate SFT data",
}

/**
 * The chip classes for a job's kind — its hue as the text, a wash of the same hue
 * behind it.
 *
 * The kind used to be uppercase grey on every row, which meant six different sorts
 * of work were told apart only by reading them. Colour makes a history of thirty
 * rows scannable: you find last night's training runs by shape, not by word.
 *
 * Written out in full rather than built by interpolation because Tailwind scans
 * source for complete class names — `text-kind-${job.kind}` compiles to nothing.
 */
export const KIND_CHIP: Record<JobKind, string> = {
  train: "text-kind-train bg-kind-train/14",
  benchmark: "text-kind-benchmark bg-kind-benchmark/14",
  chat: "text-kind-chat bg-kind-chat/14",
  synth: "text-kind-synth bg-kind-synth/14",
}

export const STATUS_LABEL: Record<JobStatus, string> = {
  waiting: "Waiting",
  queued: "Queued",
  running: "Running",
  cancelling: "Stopping",
  done: "Done",
  cancelled: "Cancelled",
  failed: "Failed",
}

/**
 * The throbber mode a job should animate in. The mark's whole state vocabulary is
 * motion, so a running job says which kind it is by how it moves — a queued one
 * holds still, which is the honest reading of waiting.
 */
export function throbberMode(
  job: Job
): "idle" | "training" | "benchmarking" | "generating" {
  if (!isRunning(job)) return "idle"
  switch (job.kind) {
    case "train":
      return "training"
    case "benchmark":
      return "benchmarking"
    case "chat":
      return "generating"
    // Auto-labelling is generation with a scorer attached, and generating is what
    // it spends effectively all of its time doing.
    case "synth":
      return "generating"
    // Teaching is the same generation on our own device.
  }
}

/**
 * Seconds a job has been running. Null when it never started.
 *
 * Takes `now` from the caller rather than reading the clock itself: a live row
 * ticks it from an interval, which keeps the reading a pure function of props and
 * out of the render path.
 */
export function elapsedSeconds(job: Job, now: number): number | null {
  if (!job.started_at) return null
  const start = new Date(job.started_at).getTime()
  const end = job.finished_at ? new Date(job.finished_at).getTime() : now
  return Math.max(0, (end - start) / 1000)
}

/**
 * How long a finished job took. Null unless it both started and stopped, so a job
 * that never ran shows nothing rather than a zero it didn't earn. Needs no clock —
 * which is what lets a history row render without one.
 */
export function durationSeconds(job: Job): number | null {
  if (!job.started_at || !job.finished_at) return null
  const start = new Date(job.started_at).getTime()
  const end = new Date(job.finished_at).getTime()
  return Math.max(0, (end - start) / 1000)
}

/** `2h 40m` / `4m 12s` / `18s`, for durations shown beside a row. */
export function formatElapsed(seconds: number): string {
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

/** `14:32` — the wall-clock time a job was queued, for the history ledger. */
export function formatClock(iso: string): string {
  return new Date(iso).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  })
}

/**
 * `Today` / `Yesterday` / `Mar 14` — the heading a history group sits under.
 * Relative for the two days you actually remember, absolute beyond that.
 */
export function formatDayGroup(iso: string): string {
  const date = new Date(iso)
  const midnight = new Date()
  midnight.setHours(0, 0, 0, 0)
  const dayMs = 24 * 60 * 60 * 1000

  if (date.getTime() >= midnight.getTime()) return "Today"
  if (date.getTime() >= midnight.getTime() - dayMs) return "Yesterday"
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" })
}
