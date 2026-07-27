import { apiFetch } from "./api"
import type { JobKind, JobStatus } from "./jobs"

/**
 * Post-training cycles: the layer above the queue.
 *
 * A cycle repeats one round — build catalogue, self-train, preference-tune,
 * benchmark, decide — until a round stops improving. It is the thing you leave
 * running overnight, and its page is the memory of what it did.
 */

export type CycleStatus = "running" | "paused" | "halted" | "done"

export type CycleStage =
  | "generating"
  | "selftrain"
  | "preference"
  | "benchmarking"
  | "deciding"

export interface CycleConfig {
  max_prompts: number
  candidates: number
  temperature: number
  raft_lr: number
  kto_lr: number
  beta: number
  /** How much the benchmark score must rise for a round to count. Never zero. */
  min_gain: number
}

export interface Cycle {
  id_cycle: string
  label: string
  status: CycleStatus
  stage: CycleStage
  round: number
  max_rounds: number
  id_model_start: string | null
  id_model_current: string | null
  /** The cycle's actual output — not whatever the last round happened to produce. */
  id_model_best: string | null
  best_score: number | null
  id_job_current: string | null
  config: CycleConfig
  halt_reason: string | null
  created_at: string
  updated_at: string
  finished_at: string | null
}

/**
 * One round. The catalogue counts either side of its generation stage are what
 * make yield computable — it is a delta, not a total.
 */
export interface CycleRound {
  id_round: string
  round: number
  approved_before: number | null
  rejected_before: number | null
  approved_after: number | null
  rejected_after: number | null
  id_model_out: string | null
  score: number | null
  verdict: string | null
  note: string | null
  started_at: string
  finished_at: string | null
}

export interface CycleJob {
  id_job: string
  kind: JobKind
  status: JobStatus
  label: string
  cycle_role: string | null
  created_at: string
}

export interface CycleDetail extends Cycle {
  rounds: CycleRound[]
  jobs: CycleJob[]
}

/**
 * The fraction of what the model generated in this round that cleared the frozen
 * bar — the loop's progress metric.
 *
 * A *delta* over the round, not the catalogue's running ratio: the catalogue
 * accumulates across rounds, so a running ratio would be dominated by earlier
 * rounds and would barely move. `null` until the round's generation stage has
 * finished, or when it added nothing.
 */
export function roundYield(round: CycleRound): number | null {
  const { approved_before, rejected_before, approved_after, rejected_after } = round
  if (
    approved_before === null ||
    rejected_before === null ||
    approved_after === null ||
    rejected_after === null
  ) {
    return null
  }
  const approved = approved_after - approved_before
  const total = approved + (rejected_after - rejected_before)
  if (total <= 0) return null
  return approved / total
}

export async function getCycles(): Promise<Cycle[]> {
  const res = await apiFetch("/cycles")
  if (!res.ok) throw new Error("Failed to load cycles.")
  return res.json()
}

export async function getCycle(id: string): Promise<CycleDetail> {
  const res = await apiFetch(`/cycles/${id}`)
  if (!res.ok) throw new Error("Failed to load the cycle.")
  return res.json()
}

export interface StartCycleInput {
  id_model: string
  label?: string
  max_rounds?: number
  config?: Partial<CycleConfig>
}

export async function startCycle(input: StartCycleInput): Promise<Cycle> {
  const res = await apiFetch("/cycles", {
    method: "POST",
    body: JSON.stringify(input),
  })
  if (!res.ok) throw new Error((await res.text()) || "Failed to start the cycle.")
  return res.json()
}

/** Pause, resume, or stop. The only mutation — stages belong to the driver. */
export async function setCycleStatus(
  id: string,
  status: "running" | "paused" | "halted"
): Promise<void> {
  const res = await apiFetch(`/cycles/${id}/status`, {
    method: "POST",
    body: JSON.stringify({ status }),
  })
  if (!res.ok) throw new Error((await res.text()) || "Failed to update the cycle.")
}

export async function deleteCycle(id: string): Promise<void> {
  const res = await apiFetch(`/cycles/${id}`, { method: "DELETE" })
  if (!res.ok) throw new Error((await res.text()) || "Failed to delete the cycle.")
}

// ── presentation ────────────────────────────────────────────────────────────

export const STAGE_LABEL: Record<CycleStage, string> = {
  generating: "Building catalogue",
  selftrain: "Self-training",
  preference: "Preference tuning",
  benchmarking: "Benchmarking",
  deciding: "Deciding",
}

export const CYCLE_STATUS_LABEL: Record<CycleStatus, string> = {
  running: "Running",
  paused: "Paused",
  halted: "Stopped",
  done: "Finished",
}

export function isCycleLive(cycle: Cycle): boolean {
  return cycle.status === "running" || cycle.status === "paused"
}

/** The ordered stages, for rendering a round as a progress track. */
export const STAGES: CycleStage[] = [
  "generating",
  "selftrain",
  "preference",
  "benchmarking",
  "deciding",
]
