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
  file: string
  version_hash: string
}

export interface ProcessorSourceStatus {
  name: string
  version_hash: string
  active_version_hash: string
  reload_error: string | null
}

export interface ProcessorSource extends ProcessorSourceStatus {
  title: string
  file: string
  source: string
  editable: boolean
}

/** Validates saved/active source identity before the editor interprets a response. */
function processorSourceStatus(raw: unknown): ProcessorSourceStatus {
  if (typeof raw !== "object" || raw === null)
    throw new Error("Invalid processor source response")
  const value = raw as Record<string, unknown>
  const { name, version_hash, active_version_hash, reload_error } = value
  if (
    typeof name !== "string" ||
    typeof version_hash !== "string" ||
    !/^[a-f0-9]{64}$/.test(version_hash) ||
    typeof active_version_hash !== "string" ||
    !/^[a-f0-9]{64}$/.test(active_version_hash) ||
    !(reload_error === null || typeof reload_error === "string")
  ) {
    throw new Error("Invalid processor source identity")
  }
  return { name, version_hash, active_version_hash, reload_error }
}

/** Validates source text, editability and metadata at the editor's network boundary. */
export function parseProcessorSource(raw: unknown): ProcessorSource {
  const status = processorSourceStatus(raw)
  const value = raw as Record<string, unknown>
  const { title, file, source, editable } = value
  if (
    typeof title !== "string" ||
    typeof file !== "string" ||
    typeof source !== "string" ||
    typeof editable !== "boolean"
  ) {
    throw new Error("Invalid processor source payload")
  }
  return { ...status, title, file, source, editable }
}

/** Reads saved source for one identity; cancellation aborts the browser request. */
export async function getProcessorSource(
  name: string,
  signal?: AbortSignal
): Promise<ProcessorSource> {
  const result = parseProcessorSource(
    await read<unknown>(
      await apiFetch(`/data/processors/${encodeURIComponent(name)}/source`, {
        signal,
      })
    )
  )
  if (result.name !== name) throw new Error("Processor source identity changed")
  return result
}

/** Saves custom source as supplied and reports disk/active state, including reload errors. */
export async function saveProcessorSource(
  name: string,
  source: string,
  signal?: AbortSignal
): Promise<ProcessorSourceStatus> {
  const raw = await read<unknown>(
    await apiFetch(`/data/processors/${encodeURIComponent(name)}/source`, {
      method: "PUT",
      body: JSON.stringify({ source }),
      signal,
    })
  )
  const result = processorSourceStatus(raw)
  if (
    typeof raw !== "object" ||
    raw === null ||
    !("saved" in raw) ||
    raw.saved !== true ||
    result.name !== name
  ) {
    throw new Error("Processor source was not saved")
  }
  return result
}

/** Creates a valid first custom processor version from its complete Python source. */
export async function createProcessor(
  source: string,
  signal?: AbortSignal
): Promise<ProcessorSource> {
  const result = parseProcessorSource(
    await read<unknown>(
      await apiFetch("/data/processors", {
        method: "POST",
        body: JSON.stringify({ source }),
        signal,
      })
    )
  )
  if (!result.editable) throw new Error("The new processor is not editable")
  return result
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

/** Checks the parameter metadata interpreted by generated forms. */
function isProcessorParam(raw: unknown): raw is ProcessorParam {
  if (typeof raw !== "object" || raw === null) return false
  const p = raw as Record<string, unknown>
  return (
    typeof p.name === "string" &&
    typeof p.label === "string" &&
    typeof p.description === "string" &&
    typeof p.type === "string" &&
    ["int", "float", "bool", "string"].includes(p.type) &&
    (typeof p.default === "string" ||
      typeof p.default === "boolean" ||
      (typeof p.default === "number" && Number.isFinite(p.default))) &&
    [p.minimum, p.maximum].every(
      (bound) =>
        bound === null || (typeof bound === "number" && Number.isFinite(bound))
    )
  )
}

/** Checks catalogue identities and forms before rendering them. */
function isProcessor(raw: unknown): raw is Processor {
  if (typeof raw !== "object" || raw === null) return false
  const p = raw as Record<string, unknown>
  return (
    typeof p.name === "string" &&
    typeof p.title === "string" &&
    typeof p.description === "string" &&
    typeof p.file === "string" &&
    typeof p.version_hash === "string" &&
    /^[a-f0-9]{64}$/.test(p.version_hash) &&
    Array.isArray(p.applies_to) &&
    p.applies_to.every((kind) => kind === "pretrain" || kind === "sft") &&
    Array.isArray(p.params) &&
    p.params.every(isProcessorParam)
  )
}

/** Validates the catalogue wire contract, turning malformed responses into load errors. */
export function parseProcessorCatalogue(raw: unknown): Processor[] {
  if (
    typeof raw !== "object" ||
    raw === null ||
    !("processors" in raw) ||
    !Array.isArray(raw.processors) ||
    !raw.processors.every(isProcessor)
  ) {
    throw new Error("Invalid processor catalogue response")
  }
  return raw.processors
}

/** Lists every active processor when the kind filter is null. */
export async function listProcessors(
  kind: DatasetKind | null = null
): Promise<Processor[]> {
  return parseProcessorCatalogue(
    await read<unknown>(
      await apiFetch(
        kind ? `/data/processors?kind=${kind}` : "/data/processors"
      )
    )
  )
}

export async function preview(
  datasetId: string,
  steps: PipelineStep[],
  limit: number,
  signal?: AbortSignal
): Promise<Preview> {
  return read<Preview>(
    await apiFetch("/data/preview", {
      method: "POST",
      body: JSON.stringify({ dataset_id: datasetId, steps, limit }),
      signal,
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
