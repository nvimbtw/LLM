import { apiFetch } from "./api"
import { parseResourceId } from "./chats"
import { findJob, type Job, type JobKind } from "./jobs"
import { getRun, type BenchmarkRunDetail } from "./benchmark"

export type RouteQuery = Record<string, string | string[] | undefined>
export type ExecutionTarget =
  | { type: "job"; job: Job }
  | { type: "gym"; id: string }
  | { type: "report"; run: BenchmarkRunDetail }

/** Appends query values to a canonical path, retaining repeated keys and parent/seed parameters through redirects. */
export function withRouteQuery(path: string, query: RouteQuery): string {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined) continue
    for (const entry of Array.isArray(value) ? value : [value])
      params.append(key, entry)
  }
  return `${path}${params.size ? `?${params}` : ""}`
}

/** Resolves shared browser execution URLs; only an actual 404 permits a gym/report fallback. */
export async function resolveExecutionTarget(
  id: string,
  kind: JobKind,
  signal?: AbortSignal
): Promise<ExecutionTarget> {
  parseResourceId(id)
  const job = await findJob(id, signal)
  if (job) return { type: "job", job }
  if (kind === "rl") {
    const response = await apiFetch(`/gym/runs/${id}`, { signal })
    if (!response.ok)
      throw new Error(
        response.status === 404
          ? "This gym run no longer exists."
          : "The gym run could not be loaded."
      )
    return { type: "gym", id }
  }
  if (kind === "benchmark")
    return { type: "report", run: await getRun(id, signal) }
  throw new Error("This execution no longer exists.")
}
