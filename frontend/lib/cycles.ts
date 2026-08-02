import { apiFetch } from "./api"
import type { JobKind, JobStatus } from "./jobs"

/**
 * Cycles: the layer above the queue.
 *
 * A cycle is a loop that drives the job queue on your behalf — repeating a round,
 * judging whether it improved, and halting when it stops paying. It is the thing
 * you leave running overnight, and its page is the memory of what it did.
 *
 * Post-training is the first one, not the only shape of one. `CycleKind` is the
 * seam that keeps it that way, and it deliberately mirrors `JobKind`: a
 * discriminant plus an opaque per-kind config, which is what let the queue grow
 * from three kinds to six without the board learning about any of them.
 *
 * The backend has one kind today and does not yet store the column, so this is
 * currently a routing seam rather than a wire field — `startCycle` posts what it
 * always did. Adding the second kind means a migration and a match in
 * `backend/src/cycle.rs`; it does not mean redesigning this page.
 */

export type CycleKind = "post_train"

export interface CycleKindInfo {
  kind: CycleKind
  label: string
  /** One line on what the loop actually repeats. */
  description: string
  /** Where its composer lives. */
  href: string
}

export const CYCLE_KINDS: readonly CycleKindInfo[] = [
  {
    kind: "post_train",
    label: "Post-train",
    description:
      "Build the catalogue, self-train on what cleared the bar, preference-tune away what didn't, benchmark. Repeats until a round stops improving, and keeps the best model rather than the last one.",
    href: "/cycles/new/post-train",
  },
]

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
  /**
   * How long a generated candidate reply may run, in tokens.
   *
   * The cycle's most consequential number. It was hard-coded at 96 and unreachable
   * from here, which capped generated replies at ~265 characters while the assistant
   * benchmark scores held-out replies averaging ~1400 — so every round trained the
   * model to answer shorter and scored it for doing so.
   */
  max_tokens: number
  raft_lr: number
  kto_lr: number
  beta: number
  /** How much the benchmark score must rise for a round to count. Never zero. */
  min_gain: number
  /**
   * Run every round to the limit instead of halting on the first that does not
   * improve. Nothing about the measurement changes — the verdict is still recorded
   * per round, a failing round is written as `explored` rather than `accepted`, and
   * the best model is still only replaced on a genuine gain. For answering "where
   * does this go over 5-10 rounds", which a halt at round 1 cannot show.
   */
  explore: boolean
  /**
   * Run the round as generate → self-train → benchmark, with no preference stage.
   *
   * An experiment knob. Scoring the intermediate model of one round put the two
   * training stages at −3.82 (self-train) and −5.74 (preference) points of assistant
   * score — so preference tuning is the larger single loss, and the only stage with
   * no prose replay anchoring it. Self-train's −3.82 is inside noise, so whether it
   * is salvageable alone is what one round with this set answers.
   */
  skip_preference: boolean
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
  /**
   * The best round's **assistant** score, not a blend of the three suites.
   *
   * Every round is benchmarked on language, markup and assistant, and the verdict
   * weighs all three — but a single stored number has to be one of them, and this is
   * the one the cycle is trying to move. Averaging them would be a composite by the
   * back door, and letting a gain on one hide a collapse on another is precisely what
   * the three-way split exists to prevent.
   */
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
  /** The round's assistant score. See `Cycle.best_score` for why it is that one. */
  score: number | null
  verdict: string | null
  /**
   * All three suites' numbers and what the gate made of them, as one sentence. The
   * only place the language and markup movements are visible, so it is rendered in
   * full rather than truncated.
   */
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
  const { approved_before, rejected_before, approved_after, rejected_after } =
    round
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
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to start the cycle.")
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
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to update the cycle.")
}

export async function deleteCycle(id: string): Promise<void> {
  const res = await apiFetch(`/cycles/${id}`, { method: "DELETE" })
  if (!res.ok)
    throw new Error((await res.text()) || "Failed to delete the cycle.")
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
