import { apiFetch } from "@/lib/api"

/**
 * The gyms surface: what a gym is, what a run of one records, and the calls
 * that reach it.
 *
 * Nothing here is a hard-coded catalogue any more. Which gyms exist and which
 * task types each one generates is `GET /gym`, answered by the registry in
 * `gyms/gyms/registry.py` — adding a task type is a row there and nothing in
 * this file. Grading is likewise absent: the answer is graded inside the gyms
 * service, which is the only place that holds the ground truth while an episode
 * is still open.
 */

export interface GymTaskType {
  id: string
  label: string
  description: string
  /**
   * Which rung of its family this is: `easy`, `medium` or `hard`. A level is a
   * claim about how many shell skills the task needs, not about how big it is —
   * each rung requires exactly one more than the one below it, so a run across
   * a family says *where* a model stops rather than only that it stopped.
   */
  level: string
  /** `ls`, `read` or `find` — the skill the family's ladder is climbing. */
  family: string
}

/** The rungs, in the order they should be attempted. Mirrors `LEVELS` in
 * `gyms/gyms/registry.py`, which is the source of the strings above. */
export const LEVELS = ["easy", "medium", "hard"] as const

export interface Gym {
  id: string
  name: string
  description: string
  task_types: GymTaskType[]
}

export type RunStatus = "pending" | "running" | "done" | "abandoned" | "failed"
export type EpisodeStatus = "pending" | "running" | "graded" | "closed"

export interface GymRun {
  id: string
  gym_id: string
  /**
   * The model's **id**, despite the name. This is the engine's own field and it
   * has always held the id — weights live at `artifacts/models/<id>/model.pt`,
   * so a display name would resolve to no path at all. `modelLabel` turns it
   * back into something readable.
   */
  model_name: string
  task_types: string[]
  seed_base: number
  episode_count: number
  status: RunStatus
  /** Mean score over the graded episodes, or null before any is graded. */
  score: number | null
  graded: number
  correct: number
  created_at: string
  finished_at: string | null
}

export interface GymEpisode {
  id: string
  run_id: string
  index: number
  gym_id: string
  task_type: string
  seed: number
  prompt: string
  system: string
  status: EpisodeStatus
  answer: string | null
  score: number | null
  reward_total: number
  created_at: string
  closed_at: string | null
  /** Both present only once the episode is graded — see `_episode_json`. */
  ground_truth?: string
  reply?: string
  steps?: GymStep[]
}

export interface GymStep {
  step_num: number
  /** The reply of the round that made this call. */
  reply: string | null
  action: string
  stdout: string | null
  stderr: string | null
  exit_code: number | null
  timed_out: boolean
  reward: number | null
}

export interface GymRunDetail extends GymRun {
  episodes: GymEpisode[]
}

export interface NewRun {
  gymId: string
  /** The model's id — see `GymRun.model_name`. */
  modelName: string
  taskTypes: string[]
  count: number
  /** Given, the run is reproducible; omitted, one is drawn. */
  seedBase?: number
}

export async function listGyms(): Promise<Gym[]> {
  return read<Gym[]>("/gym", "list the gyms")
}

export async function listGymRuns(): Promise<GymRun[]> {
  return read<GymRun[]>("/gym/runs", "list the runs")
}

export async function getGymRun(id: string): Promise<GymRunDetail> {
  return read<GymRunDetail>(`/gym/runs/${id}`, "load the run")
}

export async function getGymEpisode(id: string): Promise<GymEpisode> {
  return read<GymEpisode>(`/gym/episodes/${id}`, "load the episode")
}

export async function createGymRun(run: NewRun): Promise<GymRunDetail> {
  const res = await apiFetch("/gym/runs", {
    method: "POST",
    body: JSON.stringify({
      gym_id: run.gymId,
      model_name: run.modelName,
      task_types: run.taskTypes,
      count: run.count,
      ...(run.seedBase === undefined ? {} : { seed_base: run.seedBase }),
    }),
  })
  if (!res.ok) throw new Error(await detail(res, "create the run"))
  return res.json()
}

/**
 * One command a person typed rather than one a model called.
 *
 * The run's own steps go through the socket; this is the way to look at a task
 * by hand when no model can solve it.
 */
export async function stepGymEpisode(id: string, command: string): Promise<GymStep> {
  const res = await apiFetch(`/gym/episodes/${id}/step`, {
    method: "POST",
    body: JSON.stringify({ command }),
  })
  if (!res.ok) throw new Error(await detail(res, "run the command"))
  return res.json()
}

async function read<T>(path: string, what: string): Promise<T> {
  const res = await apiFetch(path)
  if (!res.ok) throw new Error(await detail(res, what))
  return res.json()
}

/** The service's own reason where it gave one — a 502 from the sandbox says
 * something a generic message cannot. */
async function detail(res: Response, what: string): Promise<string> {
  const body = await res.text().catch(() => "")
  try {
    const parsed = JSON.parse(body)
    if (typeof parsed?.detail === "string") return parsed.detail
    if (typeof parsed?.error === "string") return parsed.error
  } catch {
    // Not JSON. The status is all there is to say.
  }
  return `Couldn't ${what} (${res.status}).`
}

/**
 * A run's model, named for a reader.
 *
 * Falls back to the id rather than to "unknown": a run whose model has since
 * been deleted still happened, and the id is the only true thing left to say
 * about it.
 */
export function modelLabel(models: { id: string; name: string }[], id: string): string {
  return models.find((model) => model.id === id)?.name ?? id
}
