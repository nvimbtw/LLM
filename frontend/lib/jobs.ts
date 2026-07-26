import { apiFetch } from "./api"
import type { TrainConfig } from "./training"

// Wire types for the GPU queue (backend routes/jobs.rs).
//
// Everything that uses the device — a training run, a benchmark, a chat turn — is
// a row in `jobs`, executed one at a time in queue order. The board is the whole
// of what the /jobs page renders, and `config` travels with each row so a finished
// job can seed a new one without a second request.

export type JobKind = "train" | "benchmark" | "chat" | "autolabel" | "gather"

export type JobStatus =
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
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface JobBoard {
  // Running and queued, in the order they will execute.
  active: Job[]
  // Finished, newest first.
  history: Job[]
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

/** The chat a chat job replies in, so the row can link back to the conversation. */
export function chatId(job: Job): string | null {
  if (job.kind !== "chat") return null
  const config = job.config as { id_chat?: string }
  return config?.id_chat ?? null
}

// ── requests ────────────────────────────────────────────────────────────────

export async function getBoard(): Promise<JobBoard> {
  const res = await apiFetch("/jobs")
  if (!res.ok) throw new Error("Failed to load the queue.")
  return res.json()
}

export async function getJob(id: string): Promise<Job> {
  const res = await apiFetch(`/jobs/${id}`)
  if (!res.ok) throw new Error("Failed to load the job.")
  return res.json()
}

async function enqueue(kind: JobKind, config: unknown): Promise<Job> {
  const res = await apiFetch("/jobs", {
    method: "POST",
    body: JSON.stringify({ kind, config }),
  })
  if (!res.ok) {
    // A 400 here carries the specific validation failure (a bad hyperparameter, a
    // missing dataset, a model with no tokenizer), which is exactly what the
    // composer needs to show — so it is surfaced verbatim.
    throw new Error((await res.text()) || "Failed to queue the job.")
  }
  return res.json()
}

export function queueTraining(config: TrainConfig): Promise<Job> {
  return enqueue("train", config)
}

export function queueBenchmark(modelId: string): Promise<Job> {
  return enqueue("benchmark", { id_model: modelId })
}

/** Config for an auto-labelling run. Mirrors the backend `AutolabelRequest`. */
export interface AutolabelConfig {
  id_model: string
  // Candidate replies generated per prompt. The best and worst become one
  // thumbs-up and one thumbs-down example, so this must be at least 2.
  candidates: number
  max_prompts: number
  // Above the chat default on purpose: candidates have to differ from each other
  // for a best/worst split to mean anything.
  temperature: number
}

export function queueAutolabel(config: AutolabelConfig): Promise<Job> {
  return enqueue("autolabel", config)
}

/** Config for a signal-gathering run. Mirrors the backend `GatherRequest`. */
export interface GatherConfig {
  id_model: string
  // Replies to generate for rating. Sized to be finishable in one sitting —
  // rating is the scarce resource, not generation.
  batch?: number
  temperature?: number
}

export function queueGather(config: GatherConfig): Promise<Job> {
  return enqueue("gather", config)
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

/**
 * Moves a queued job to sit directly after `afterId`, or to the front when that is
 * null. Expressed as a neighbour rather than an index so a drop lands where it was
 * dropped even if the board shifted underneath the drag.
 */
export async function reorderJob(id: string, afterId: string | null): Promise<void> {
  const res = await apiFetch(`/jobs/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ after: afterId }),
  })
  if (!res.ok) throw new Error((await res.text()) || "Failed to reorder the job.")
}

export async function cancelJob(id: string): Promise<void> {
  const res = await apiFetch(`/jobs/${id}/cancel`, { method: "POST" })
  if (!res.ok) throw new Error((await res.text()) || "Failed to cancel the job.")
}

export async function retryJob(id: string): Promise<Job> {
  const res = await apiFetch(`/jobs/${id}/retry`, { method: "POST" })
  if (!res.ok) throw new Error((await res.text()) || "Failed to re-run the job.")
  return res.json()
}

export async function deleteJob(id: string): Promise<void> {
  const res = await apiFetch(`/jobs/${id}`, { method: "DELETE" })
  if (!res.ok) throw new Error((await res.text()) || "Failed to delete the job.")
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
  autolabel: "Build catalogue",
  gather: "Gather signals",
}

export const STATUS_LABEL: Record<JobStatus, string> = {
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
export function throbberMode(job: Job): "idle" | "training" | "benchmarking" | "generating" {
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
    case "autolabel":
      return "generating"
    // Gathering is generation with a person, rather than a rule, doing the
    // judging — and generating is all the GPU is doing either way.
    case "gather":
      return "generating"
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
