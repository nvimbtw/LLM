import { apiFetch } from "./api"

// The data engine's vocabulary: datasets before they are training data, the
// processors that clean them, and the runs that do the work.
//
// The training rail reads from here too. `training.ts` keeps a narrow
// `DatasetInfo` — id, name, document count — because the picker renders three
// fields and should not depend on the whole row; but there is one catalogue
// now, and a run selects from it by id.

/** Prose for next-token training, or conversations for fine-tuning. */
export type DatasetKind = "pretrain" | "sft"

/** Pulled from the Hub, or produced by cleaning one that was. */
export type DatasetOrigin = "huggingface" | "derived"

export type DatasetStatus = "pulling" | "ready" | "failed"

export type RunStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "done"
  | "cancelled"
  | "failed"

/** One step of a pipeline: a processor by name and the parameters it ran with. */
export interface PipelineStep {
  processor: string
  params?: Record<string, ParamValue>
}

export type ParamValue = string | number | boolean

export interface Dataset {
  id: string
  name: string
  description: string | null
  kind: DatasetKind
  origin: DatasetOrigin
  status: DatasetStatus
  source_repo: string | null
  source_config: string | null
  source_split: string | null
  parent_id: string | null
  steps: PipelineStep[] | null
  document_count: number
  byte_size: number
  created_at: string
  updated_at: string
}

/** One turn of a conversation, as the source wrote it. */
export interface Turn {
  role: string
  content: string
}

export interface DatasetDocument {
  id: string
  ordinal: number
  content: string | null
  messages: Turn[] | null
  meta: Record<string, ParamValue> | null
  byte_size: number
}

export interface HubResult {
  repo: string
  author: string | null
  downloads: number | null
  likes: number | null
  tags: string[]
  updated_at: string | null
}

export interface HubConfig {
  name: string
  columns: string[]
  splits: string[]
}

export interface HubDetail {
  repo: string
  downloads: number | null
  likes: number | null
  tags: string[]
  updated_at: string | null
  gated: boolean
  configs: HubConfig[]
  /** Null when the shards are the repo's own; the Hub's conversion branch when
   *  they are the parquet it made of a repo that ships CSV or JSON. */
  revision: string | null
  shards: string[]
  shard_count: number
  prefixes: { prefix: string; shards: number }[]
}

/** One config/split pair the Hub has rows indexed for. */
export interface HubSplit {
  config: string
  split: string
}

/**
 * A look inside a repo before pulling it: real rows, and what they imply about
 * the two fields the pull form cannot guess on its own.
 *
 * Cells are whatever the source column held — a string of prose, a list of
 * turns, a number — so anything rendering them must narrow before it reads.
 */
export interface HubSample {
  repo: string
  config: string
  split: string
  splits: HubSplit[]
  columns: string[]
  rows: Record<string, unknown>[]
  truncated: boolean
  suggested_kind: DatasetKind | null
  suggested_column: string | null
}

export type ParamType = "int" | "float" | "bool" | "string"

export interface ProcessorParam {
  name: string
  label: string
  type: ParamType
  default: ParamValue
  description: string
  minimum: number | null
  maximum: number | null
}

export interface Processor {
  name: string
  title: string
  description: string
  applies_to: DatasetKind[]
  params: ProcessorParam[]
}

/** One document's journey through a pipeline, for the before/after view. */
export interface PreviewResult {
  ordinal: number
  before: { content: string | null; messages: Turn[] | null }
  after: { content: string | null; messages: Turn[] | null } | null
  dropped: boolean
  dropped_by: string | null
  changed: boolean
}

export interface Preview {
  kind: DatasetKind
  sampled: number
  dropped: number
  results: PreviewResult[]
}

export interface DataRun {
  id: string
  kind: "pull" | "process"
  status: RunStatus
  label: string
  progress: number
  stage: string | null
  ref_id: string | null
  error: string | null
  created_at: string
  finished_at: string | null
}

export interface PullRequest {
  repo: string
  name: string
  kind: DatasetKind
  description?: string
  prefix?: string | null
  column?: string | null
  limit_rows: number
}

/** What a queued pull or process answered with. */
export interface Queued {
  dataset_id: string
  run_id: string
}

// The backend forwards the data engine's own status and body, so a refusal
// arrives with its reason in `detail` — surfacing that is the difference between
// "a dataset called x already exists" and "request failed".
async function read<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      error?: string
      detail?: string
    } | null
    throw new Error(
      body?.detail ?? body?.error ?? `request failed (${response.status})`
    )
  }
  return (await response.json()) as T
}

export async function searchHub(query: string): Promise<HubResult[]> {
  const path = `/data/hub?q=${encodeURIComponent(query)}&limit=20`
  const body = await read<{ results: HubResult[] }>(await apiFetch(path))
  return body.results
}

export async function hubDetail(repo: string): Promise<HubDetail> {
  return read<HubDetail>(
    await apiFetch(`/data/hub/detail?repo=${encodeURIComponent(repo)}`)
  )
}

export async function hubSample(
  repo: string,
  where?: HubSplit | null
): Promise<HubSample> {
  const params = new URLSearchParams({ repo })
  if (where) {
    params.set("config", where.config)
    params.set("split", where.split)
  }
  return read<HubSample>(await apiFetch(`/data/hub/sample?${params}`))
}

export async function listDatasets(): Promise<Dataset[]> {
  const body = await read<{ datasets: Dataset[] }>(
    await apiFetch("/data/datasets")
  )
  return body.datasets
}

export async function getDataset(id: string): Promise<Dataset> {
  return read<Dataset>(await apiFetch(`/data/datasets/${id}`))
}

export async function listDocuments(
  id: string,
  offset: number,
  limit: number
): Promise<{ documents: DatasetDocument[]; total: number }> {
  return read(
    await apiFetch(
      `/data/datasets/${id}/documents?offset=${offset}&limit=${limit}`
    )
  )
}

export async function listProcessors(kind: DatasetKind): Promise<Processor[]> {
  const body = await read<{ processors: Processor[] }>(
    await apiFetch(`/data/processors?kind=${kind}`)
  )
  return body.processors
}

export async function preview(
  datasetId: string,
  steps: PipelineStep[],
  limit: number
): Promise<Preview> {
  return read<Preview>(
    await apiFetch("/data/preview", {
      method: "POST",
      body: JSON.stringify({ dataset_id: datasetId, steps, limit }),
    })
  )
}

export async function pullDataset(request: PullRequest): Promise<Queued> {
  return read<Queued>(
    await apiFetch("/data/datasets/pull", {
      method: "POST",
      body: JSON.stringify(request),
    })
  )
}

export async function processDataset(
  id: string,
  name: string,
  steps: PipelineStep[],
  description?: string
): Promise<Queued> {
  return read<Queued>(
    await apiFetch(`/data/datasets/${id}/process`, {
      method: "POST",
      body: JSON.stringify({ name, steps, description }),
    })
  )
}

export async function deleteDataset(id: string): Promise<void> {
  const response = await apiFetch(`/data/datasets/${id}`, { method: "DELETE" })
  if (!response.ok && response.status !== 204) {
    throw new Error(`could not delete (${response.status})`)
  }
}

export async function listRuns(): Promise<DataRun[]> {
  const body = await read<{ runs: DataRun[] }>(await apiFetch("/data/runs"))
  return body.runs
}

/** Bytes as something a person reads at a glance. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ["KB", "MB", "GB", "TB"]
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`
}

/** The text of a document, whichever shape it is in. */
export function documentText(
  document: { content: string | null; messages: Turn[] | null } | null
): string {
  if (!document) return ""
  if (document.content !== null) return document.content
  return (document.messages ?? [])
    .map((m) => `${m.role}: ${m.content}`)
    .join("\n\n")
}
