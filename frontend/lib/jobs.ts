import { apiFetch } from "./api"
import type { BenchmarkSuite } from "./benchmark"
import type { TokenizerBuild } from "./tokenizers"
import { RESUMED_ARCH_FIELDS, isFineTune, type TrainConfig } from "./training"

// The GPU queue. The engine calls these runs and so does the wire; the UI has
// always called them jobs, and `fromRow` is where the two names meet.
//
// Everything that uses the device is a row, executed one at a time in queue
// order. `config` travels with each row, so a finished job can seed a new one
// without a second request.

export type JobKind = "train" | "benchmark" | "chat" | "tokenize"

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
  /** The run's own summary, for whatever has no table to read it back from. */
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

/** The engine's `run_id` is this side's `id_job`; nothing else differs. */
function fromRow({ run_id, ...rest }: RunRow): Job {
  return { id_job: run_id, ...rest }
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

export async function getJob(id: string): Promise<Job> {
  const res = await apiFetch(`/runs/${id}`)
  if (!res.ok) throw new Error("Failed to load the job.")
  return fromRow(await res.json())
}

/**
 * A step to run *after* the one before it, against the model that one produces.
 *
 * A flat list, because "then, then, then" is what a person means by a chain. Only
 * `train` and `benchmark` can follow: they are the two kinds that take their model
 * from the run before them. A follow-up's config names **no** model — that is the
 * one question the chain exists to leave open, and the engine answers it when the
 * predecessor lands.
 */
export type FollowUp =
  | { kind: "train"; config: TrainConfig }
  | { kind: "benchmark"; config: unknown }

async function post(path: string, body: unknown): Promise<Response> {
  const res = await apiFetch(path, {
    method: "POST",
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    // The failure carries the specific validation error — a bad hyperparameter, a
    // missing dataset, a model with no tokenizer — which is what the composer
    // needs to show, so it is surfaced verbatim.
    throw new Error((await res.text()) || "Failed to queue the job.")
  }
  return res
}

//: Where each kind is queued. `chat` is not here: a turn is answered on the
//: socket that asked for it rather than put on the line — see frontend/CLAUDE.md.
const QUEUE_PATH: Record<Exclude<JobKind, "chat">, string> = {
  train: "/runs/train",
  benchmark: "/runs/benchmark",
  tokenize: "/runs/tokenize",
}

//: A fine-tune is the same *kind* of run as a pretrain — it produces a model and
//: takes the same place in the queue — but it is a different request, because it
//: takes its architecture from the base model instead of stating one.
const FINE_TUNE_PATH = "/runs/sft"

/**
 * The engine's shape for a training run.
 *
 * One name is adapted here and nowhere else, in the manner of `id_job` and
 * `id_run`: the composer's **`base_model_id`** is the engine's **`base_model`**.
 * The engine decides what a run *is* by that one field — a config carrying it is
 * a fine-tune, a config without it is a pretrain — so failing to translate it
 * did not produce an error, it produced a **pretrain**: a fine-tune of two chat
 * sets was run as a pretrain that went looking for `smoltalk` among the `.txt`
 * corpora and failed minutes later, having queued perfectly happily.
 *
 * Everything else rides along untouched. The engine's `SftConfig` keeps the
 * fields it knows and drops the pretrain-only ones (`dimensions`, `tokenizer_id`,
 * `curriculum`, …), which is exactly right: a fine-tune that restated the
 * architecture could disagree with the weights it starts from, and the weights
 * would win, silently.
 *
 * A **resuming pretrain** is the same statement in the one place the engine did
 * not get to drop the fields for us. It takes its architecture from the state it
 * continues, and restating it is refused — so the fields are dropped here rather
 * than sent at whatever the form was showing, which is a 400 for every resume
 * whose shape differs from the composer's defaults. See `RESUMED_ARCH_FIELDS`.
 */
function forEngine(config: TrainConfig): Record<string, unknown> {
  if (isFineTune(config.mode)) {
    const { base_model_id, ...rest } = config
    return { ...rest, base_model: base_model_id }
  }
  if (!config.resume_from) return { ...config }
  const rest: Record<string, unknown> = { ...config }
  for (const key of RESUMED_ARCH_FIELDS) delete rest[key]
  return rest
}

async function queue(
  kind: Exclude<JobKind, "chat">,
  config: unknown,
  steps: FollowUp[],
  path: string = QUEUE_PATH[kind]
): Promise<Job> {
  if (steps.length === 0) {
    return fromRow(await (await post(path, config)).json())
  }
  // Chained steps are translated too. A follow-up deliberately names no base —
  // the engine fills it in when its predecessor lands — so this maps an absent
  // `base_model_id` to an absent `base_model`, which is the same statement in
  // the engine's vocabulary rather than a missing field in it.
  const chain = [
    { kind, config },
    ...steps.map((step) =>
      step.kind === "train" ? { ...step, config: forEngine(step.config) } : step
    ),
  ]
  const queued: RunRow[] = await (await post("/runs/chain", chain)).json()
  return fromRow(queued[0])
}

export function queueTraining(
  config: TrainConfig,
  steps: FollowUp[] = []
): Promise<Job> {
  return queue(
    "train",
    forEngine(config),
    steps,
    isFineTune(config.mode) ? FINE_TUNE_PATH : QUEUE_PATH.train
  )
}

export function queueTokenizer(config: TokenizerBuild): Promise<Job> {
  return queue("tokenize", config, [])
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
export function retryJob(job: Job): Promise<Job> {
  if (job.kind === "chat")
    throw new Error("A chat turn is re-asked, not re-run.")
  return queue(job.kind, job.config, [])
}

export async function deleteJob(id: string): Promise<void> {
  const res = await apiFetch(`/runs/${id}`, { method: "DELETE" })
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to delete the job.")
}

export async function setPaused(paused: boolean): Promise<void> {
  const res = await apiFetch(`/runs/pause?paused=${paused}`, { method: "POST" })
  if (!res.ok) throw new Error("Failed to update the queue.")
}

// ── presentation ────────────────────────────────────────────────────────────

export const KIND_LABEL: Record<JobKind, string> = {
  train: "Train",
  benchmark: "Benchmark",
  chat: "Chat",
  tokenize: "Tokenizer",
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
  tokenize: "text-kind-tokenize bg-kind-tokenize/14",
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
    case "chat":
      return "generating"
    // Its own state now. It used to borrow training's, on the grounds that a
    // fourth motion would be one nobody could tell apart — but the mark's
    // vocabulary is colour rather than motion, and cutting each patch in two is
    // a picture of what a vocabulary build actually does.
    case "tokenize":
      return "tokenizing"
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
