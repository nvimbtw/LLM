import { parseResourceId } from "./chats"
import { apiFetch } from "./api"
import type { BenchmarkSuite } from "./benchmark"
import type { TokenizerBuild } from "./tokenizers"
import { decodeArchitecture } from "./architecture/decoder"
import { requireValidationReport } from "./model-validation"
import { selectedArchitectureForEngine } from "./training-selection"
import type {
  ValidationRun,
  ValidationReport,
  ValidationRequest,
} from "./architecture/contracts.generated"
import {
  LEGACY_STRUCTURE_FIELDS,
  RESUMED_ARCH_FIELDS,
  RESUMED_TOKENIZER_FIELDS,
  isFineTune,
  type TrainConfig,
} from "./training"

// The GPU queue. The LLM engine calls these runs and so does the wire; the UI has
// always called them jobs, and `fromRow` is where the two names meet.
//
// Everything that uses the device is a row, executed one at a time in queue
// order. `config` travels with each row, so a finished job can seed a new one
// without a second request.

export type JobKind =
  | "train"
  | "benchmark"
  | "tokenize"
  | "rl"
  | "model_validation"
type QueueKind = Exclude<JobKind, "model_validation">

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
  status: JobStatus
  label: string
  // The job's own config: a TrainConfig, a { id_model }, or a chat request. Typed
  // as unknown because the shape depends on `kind`; the readers below narrow it.
  config: unknown
  progress: number
  stage: string | null
  // What the job produced: a model id, a benchmark run id, or a message id.
  ref_id: string | null
  output: {
    type: "model" | "tokenizer" | "benchmark" | "validation"
    id: string | null
  }
  /** The run's own summary, for whatever has no table to read it back from. */
  result: unknown
  /**
   * The job this one runs after, for a `waiting` row. Null for every ordinary job.
   * The board draws a chain from it: a follow-up is not "queued in position 4", it
   * is "after that one", and a rank would be a lie about when it runs.
   */
  depends_on: string | null
  error: string | null
  created_at: string | null
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

/** The wire shape of a row, and of the board it arrives in. */
interface RunRow extends Omit<Job, "id_job"> {
  run_id: string
}

interface BoardPayload {
  active: RunRow[]
  history: RunRow[]
  history_offset: number
  history_limit: number
  history_total: number
  paused: boolean
}

/** Maps execution identity once; private validation envelopes additionally pass the shared runtime schema before presentation. */
function fromRow(row: RunRow): Job {
  const current =
    row.kind === "model_validation"
      ? decodeArchitecture("ValidationRun", row)
      : row
  if (current.kind === "model_validation" && current.result)
    requireValidationReport(
      decodeArchitecture("ValidationRun", current),
      decodeArchitecture("ValidationReport", current.result)
    )
  const { run_id, ...rest } = current
  return { id_job: run_id, ...rest }
}

/** Adapts an already interpreted validation queue envelope through the one execution identity mapper used by shared job presentation. */
export function jobFromValidation(run: ValidationRun): Job {
  return fromRow(run)
}

/** Returns the exact saved validation request, leaving other job families uninterpreted. */
export function validationConfig(job: Job): ValidationRequest | null {
  return job.kind === "model_validation"
    ? decodeArchitecture("ValidationRequest", job.config)
    : null
}

/** Returns the completed validation verdict/identity, without treating terminal lifecycle alone as a valid model configuration. */
export function validationReport(job: Job): ValidationReport | null {
  return job.kind === "model_validation" && job.result != null
    ? decodeArchitecture("ValidationReport", job.result)
    : null
}

/** Labels completed validation by its actual verdict; ordinary jobs retain their existing lifecycle labels. */
export function jobStatusLabel(job: Job): string {
  const report = job.status === "done" ? validationReport(job) : null
  if (!report) return STATUS_LABEL[job.status]
  return {
    valid: "Validated",
    invalid: "Invalid",
    provisional: "Provisional",
    inspection_only: "Inspected",
  }[report.verdict]
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
  const config = job.config as TrainConfig & {
    base_model?: string
    display_name?: string
  }
  return {
    ...config,
    model_name: config.display_name ?? config.model_name,
    mode: config.mode ?? (config.base_model ? "sft" : "pretrain"),
    base_model_id: config.base_model ?? config.base_model_id,
  }
}

/** The model a benchmark job scores. */
export function benchmarkModelId(job: Job): string | null {
  if (job.kind !== "benchmark") return null
  const config = job.config as { model_name?: string }
  return config?.model_name ?? null
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
  if (config?.suite === "language_v3") return "language_v3"
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

/** Maps a durable execution to its canonical domain page, preserving the execution identity. */
export function jobHref(job: Pick<Job, "kind" | "id_job">): string {
  const prefix: Record<JobKind, string> = {
    train: "/train",
    benchmark: "/bench",
    rl: "/gyms",
    tokenize: "/tokenizers/builds",
    model_validation: "/arch/validations",
  }
  return `${prefix[job.kind]}/${job.id_job}`
}

/**
 * The three thinking budgets, as the SFT corpus tags them.
 *
 * Nothing queues a budget sweep any more — the split lives with the data module
 * that will rebuild that corpus. What is left is reading: a stored benchmark run
 * still says which budget it was scored under.
 */
export type Tier = "low" | "med" | "high"

export const TIER_LABEL: Record<Tier, string> = {
  low: "Short",
  med: "Medium",
  high: "Long",
}

// ── requests ────────────────────────────────────────────────────────────────

/** How many finished jobs a history page holds. Mirrors the server's default. */
export const HISTORY_PAGE_SIZE = 25

export async function getBoard(offset = 0): Promise<JobBoard> {
  const params = new URLSearchParams({
    offset: String(Math.max(0, offset)),
    limit: String(HISTORY_PAGE_SIZE),
  })
  const res = await apiFetch(`/runs?${params}`)
  if (!res.ok) throw new Error("Failed to load the queue.")
  const payload: BoardPayload = await res.json()
  return {
    active: payload.active.map(fromRow),
    history: payload.history.map(fromRow),
    historyOffset: payload.history_offset,
    historyLimit: payload.history_limit,
    historyTotal: payload.history_total,
    paused: payload.paused,
  }
}

/** Reads an execution row under optional captured bearer/abort ownership, returning null only for an unknown UUID; identity mismatches and service failures reject. */
export async function findJob(
  id: string,
  signal?: AbortSignal,
  token?: string
): Promise<Job | null> {
  const identity = parseResourceId(id).toLowerCase()
  const res = await apiFetch(`/runs/${identity}`, {
    signal,
    ...(token ? { headers: { Authorization: `Bearer ${token}` } } : {}),
  })
  if (res.status === 404) return null
  if (!res.ok) throw new Error("Failed to load the job.")
  const row: RunRow = await res.json()
  if (
    row.run_id !== identity ||
    !["train", "tokenize", "benchmark", "rl", "model_validation"].includes(
      row.kind
    )
  )
    throw new Error(
      "The server returned an invalid execution identity or kind."
    )
  return fromRow(row)
}

/** Reads the authoritative execution row under optional captured-session ownership, rejecting missing or mismatched identities. */
export async function getJob(
  id: string,
  request?: { token: string; signal?: AbortSignal }
): Promise<Job> {
  const job = await findJob(id, request?.signal, request?.token)
  if (!job) throw new Error("This execution no longer exists.")
  return job
}

/**
 * A step to run *after* the one before it, against the model that one produces.
 *
 * A flat list, because "then, then, then" is what a person means by a chain. Only
 * `train` and `benchmark` can follow: they are the two kinds that take their model
 * from the run before them. A follow-up's config names **no** model — that is the
 * one question the chain exists to leave open, and the LLM engine answers it when the
 * predecessor lands.
 */
export type FollowUp =
  | { kind: "train"; config: TrainConfig }
  | { kind: "benchmark"; config: unknown }

export interface JobSubmissionRequest {
  requestId: string
  token?: string
  signal?: AbortSignal
}

/** Preserves a definite HTTP rejection separately from an unknown transport/publication outcome. */
export class JobSubmissionError extends Error {
  /** Records a bounded response message and upstream status without assuming server rollback for 5xx errors. */
  constructor(
    public readonly status: number,
    message: string
  ) {
    super(message.slice(0, 2000))
  }
}

/** Posts one queue operation with its captured actor/request identity; ambiguous network failures remain retryable under that identity. */
async function post(
  path: string,
  body: unknown,
  request?: JobSubmissionRequest
): Promise<Response> {
  const query = request
    ? `?request_id=${encodeURIComponent(request.requestId)}`
    : ""
  const res = await apiFetch(path + query, {
    method: "POST",
    body: JSON.stringify(body),
    signal: request?.signal,
    ...(request?.token
      ? { headers: { Authorization: `Bearer ${request.token}` } }
      : {}),
  })
  if (!res.ok) {
    throw new JobSubmissionError(
      res.status,
      (await res.text()) || "Failed to queue the job."
    )
  }
  return res
}

//: Where each kind is queued. `chat` is not here: a turn is answered on the
//: socket that asked for it rather than put on the line — see frontend/CLAUDE.md.
const QUEUE_PATH: Record<QueueKind, string> = {
  train: "/runs/train",
  benchmark: "/runs/benchmark",
  tokenize: "/runs/tokenize",
  rl: "/runs/rl",
}

//: A fine-tune is the same *kind* of run as a pretrain — it produces a model and
//: takes the same place in the queue — but it is a different request, because it
//: takes its architecture from the base model instead of stating one.
const FINE_TUNE_PATH = "/runs/sft"

/** Serializes the effective training contract once for root and chained runs: selected recipes own structure, saved-model consumers inherit it, and browser evidence keys never cross HTTP. */
export function trainingForEngine(
  config: TrainConfig
): Record<string, unknown> {
  const rest: Record<string, unknown> = { ...config }
  delete rest.architecture_validation_key
  if (isFineTune(config.mode) || config.resume_from) {
    for (const key of [
      ...LEGACY_STRUCTURE_FIELDS,
      ...RESUMED_ARCH_FIELDS,
      ...RESUMED_TOKENIZER_FIELDS,
    ])
      delete rest[key]
    if (isFineTune(config.mode)) {
      delete rest.base_model_id
      rest.base_model = config.base_model_id
    }
    return rest
  }
  if (config.architecture) {
    rest.architecture = selectedArchitectureForEngine(config)
    for (const key of LEGACY_STRUCTURE_FIELDS) delete rest[key]
  }
  return rest
}

/** Queues only model/tokenizer/benchmark families, translating follow-up training configs; validation uses its explicit resource-bound API. */
async function queue(
  kind: QueueKind,
  config: unknown,
  steps: FollowUp[],
  path: string = QUEUE_PATH[kind],
  request?: JobSubmissionRequest
): Promise<Job> {
  if (steps.length === 0) {
    return fromRow(await (await post(path, config, request)).json())
  }
  const chain = [
    { kind, config },
    ...steps.map((step) =>
      step.kind === "train"
        ? { ...step, config: trainingForEngine(step.config) }
        : step
    ),
  ]
  const queued: RunRow[] = await (
    await post("/runs/chain", chain, request)
  ).json()
  if (!Array.isArray(queued) || queued.length !== chain.length)
    throw new Error("Queued chain differs from the submitted step count.")
  return fromRow(queued[0])
}

/** Queues a root training/SFT operation or complete chain with one actor-scoped deduplication UUID. */
export function queueTraining(
  config: TrainConfig,
  steps: FollowUp[] = [],
  request: JobSubmissionRequest = { requestId: crypto.randomUUID() }
): Promise<Job> {
  return queue(
    "train",
    trainingForEngine(config),
    steps,
    isFineTune(config.mode) ? FINE_TUNE_PATH : QUEUE_PATH.train,
    request
  )
}

export function queueTokenizer(config: TokenizerBuild): Promise<Job> {
  return queue("tokenize", config, [])
}

/**
 * Queue a reinforcement-learning run against a gym.
 *
 * Typed as the LLM engine's `RlConfig` rather than through a composer shape: there
 * is no composer for one yet, and inventing a second name for every field would
 * be the translation layer `queueTraining` documents the cost of.
 */
export function queueReinforcement(
  config: Record<string, unknown>,
  steps: FollowUp[] = [],
  request: JobSubmissionRequest = { requestId: crypto.randomUUID() }
): Promise<Job> {
  return queue("rl", config, steps, QUEUE_PATH.rl, request)
}

export function queueBenchmark(
  modelId: string,
  suite: BenchmarkSuite = "language",
  steps: FollowUp[] = []
): Promise<Job> {
  return queue("benchmark", { model_name: modelId, suite }, steps)
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
  const query = afterId ? `?after=${afterId}` : ""
  const res = await apiFetch(`/runs/${id}${query}`, { method: "PATCH" })
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to reorder the job.")
}

export async function cancelJob(id: string): Promise<void> {
  const res = await apiFetch(`/runs/${id}/cancel`, { method: "POST" })
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to cancel the job.")
}

/**
 * Runs a finished job again. Queued from its stored config rather than by a
 * server-side clone, so a retry is an ordinary new run and the failed one stays in
 * the history as what happened.
 */
export async function retryJob(
  job: Job,
  request: JobSubmissionRequest = { requestId: crypto.randomUUID() }
): Promise<Job> {
  const res = await post(`/runs/${job.id_job}/retry`, {}, request)
  return fromRow(await res.json())
}

export async function deleteJob(id: string): Promise<void> {
  const res = await apiFetch(`/runs/${id}`, { method: "DELETE" })
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to delete the job.")
}

/** Queues saved recovery under a fresh or explicitly retried actor/request identity. */
export async function recoverJob(
  id: string,
  request: JobSubmissionRequest = { requestId: crypto.randomUUID() }
): Promise<Job> {
  const res = await post(`/runs/${id}/recover`, {}, request)
  return fromRow((await res.json()) as RunRow)
}

export async function setPaused(paused: boolean): Promise<void> {
  const res = await apiFetch(`/runs/pause?paused=${paused}`, { method: "POST" })
  if (!res.ok) throw new Error("Failed to update the queue.")
}

// ── presentation ────────────────────────────────────────────────────────────

export const KIND_LABEL: Record<JobKind, string> = {
  train: "Train",
  benchmark: "Benchmark",
  tokenize: "Tokenizer",
  rl: "Gym",
  model_validation: "Validation",
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
  tokenize: "text-kind-tokenize bg-kind-tokenize/14",
  rl: "text-kind-rl bg-kind-rl/14",
  model_validation: "text-foreground bg-muted",
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
): "idle" | "training" | "benchmarking" | "generating" | "tokenizing" {
  if (!isRunning(job)) return "idle"
  switch (job.kind) {
    case "train":
      return "training"
    case "benchmark":
      return "benchmarking"
    case "tokenize":
      return "tokenizing"
    case "rl":
      return "training"
    case "model_validation":
      return "benchmarking"
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
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null
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
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null
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

/** Formats a known queued time for the history ledger; unavailable or invalid timestamps remain explicitly absent. */
export function formatClock(iso: string | null): string {
  if (!iso || !Number.isFinite(Date.parse(iso))) return "—"
  return new Date(iso).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  })
}

/**
 * `Today` / `Yesterday` / `Mar 14` — the heading a history group sits under.
 * Relative for the two days you actually remember, absolute beyond that.
 */
export function formatDayGroup(iso: string | null): string {
  if (!iso || !Number.isFinite(Date.parse(iso))) return "Unknown date"
  const date = new Date(iso)
  const midnight = new Date()
  midnight.setHours(0, 0, 0, 0)
  const dayMs = 24 * 60 * 60 * 1000

  if (date.getTime() >= midnight.getTime()) return "Today"
  if (date.getTime() >= midnight.getTime() - dayMs) return "Yesterday"
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" })
}
