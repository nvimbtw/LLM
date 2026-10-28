"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import type { CurriculumStage, DatasetSelection } from "@/lib/training"
import { estimateTrainingParameters } from "@/lib/training-architecture"
import { useSessionToken } from "./use-auth"

export type ModelTier = "flagship" | "balanced" | "fast"

export interface Model {
  id: string
  name: string
  family: string
  tier: ModelTier | null
  /** Measured count or legacy shape estimate; null means unavailable and zero remains a measured zero. */
  params: number | null
  /** Distinguishes a legacy shape estimate from a saved measurement. */
  parameterCountEstimated?: boolean
  /** Context window in tokens. */
  contextWindow: number
  description: string
  /** Whether the model is published (visible to non-admins). */
  published: boolean
  /** Immutable saved modular semantics; absence retains legacy architecture presentation. */
  definitionId?: string | null
  /** True only when public runtime metadata reports implemented context extension. */
  supportsContextExtension?: boolean
}

export interface RawModelConfig {
  mode?: "pretrain" | "sft" | "kto"
  dimensions?: number
  ctx_window?: number
  heads?: number
  layers?: number
  ff_multiplier?: number
  max_vocab?: number
  lr?: number
  epochs?: number
  base_model?: string
  resume_from?: string
  datasets?: DatasetSelection[]
  curriculum?: CurriculumStage[]
}

export interface RawArchitecture {
  dimensions: number
  heads: number
  layers: number
  ff_multiplier: number
  ctx_window: number
}

export interface RawCheckpointResult {
  index?: number
  step?: number
  total_steps?: number
  epoch?: number
  stage?: number
  loss?: number | null
}

export interface RawModelResult {
  iterations?: number
  parameters?: number
  vocab?: number
  dataset?: {
    vocab_size?: number
    documents?: number
    conversations?: number
    chars?: number
    corpus_len?: number
  }
  files?: number
  chars?: number
  vocab_size?: number
  merges?: number
  corpus_len?: number
  checkpoint?: RawCheckpointResult
  architecture?: RawArchitecture
  base_model?: string
}

export interface RawModel {
  definition_id?: string | null
  context_extension_supported?: boolean
  architecture?: RawArchitecture | null
  vocab_size?: number | null
  parameter_count?: number | null
  size?: ModelTier | null
  recovery_kept?: boolean
  model_id: string
  name: string
  description: string | null
  parameters: {
    config?: RawModelConfig
    result?: RawModelResult
  } | null
  context: number | null
  family: string | null
  published: boolean
  has_weights: boolean
  created_at: string | null
  run_id: string | null
  parent_id: string | null
  checkpoint_step: number | null
  checkpoint_index: number | null
}

/** Returns authoritative resolved shape, including explicit unavailability; older responses alone fall back to saved results/config. */
export function architectureOf(raw: RawModel): RawArchitecture | null {
  if ("architecture" in raw)
    return validArchitecture(raw.architecture) ? { ...raw.architecture } : null
  const c = raw.parameters?.config
  for (const shape of [
    raw.architecture,
    raw.parameters?.result?.architecture,
    c,
  ]) {
    if (validArchitecture(shape)) {
      return {
        dimensions: shape.dimensions,
        heads: shape.heads,
        layers: shape.layers,
        ff_multiplier: shape.ff_multiplier,
        ctx_window: shape.ctx_window,
      }
    }
  }
  return null
}

/** Narrows interpreted shape metadata to a complete set of positive integral dimensions. */
function validArchitecture(
  shape: RawModelConfig | RawArchitecture | null | undefined
): shape is RawArchitecture {
  return Boolean(
    shape &&
    validCount(shape.dimensions) &&
    validCount(shape.heads) &&
    validCount(shape.layers) &&
    validCount(shape.ff_multiplier) &&
    validCount(shape.ctx_window)
  )
}

/** Accepts positive safe integral metadata counts; absent and malformed values are unavailable. */
function validCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
}

/** Returns measured vocabulary size across current and legacy result shapes; the requested learning ceiling is not a measurement. */
export function vocabularySizeOf(raw: RawModel): number | null {
  const result = raw.parameters?.result
  return (
    [
      raw.vocab_size,
      result?.vocab,
      result?.vocab_size,
      result?.dataset?.vocab_size,
    ].find(validCount) ?? null
  )
}

/** Returns the base model identity from a fine-tune request or its saved provenance. */
export function baseModelOf(raw: RawModel): string | null {
  return (
    raw.parameters?.config?.base_model ??
    raw.parameters?.result?.base_model ??
    null
  )
}

/** Returns the model or checkpoint a continued pretraining run started from. */
export function resumedFromOf(raw: RawModel): string | null {
  return raw.parameters?.config?.resume_from ?? null
}

export interface LineageDatasetRef {
  id: string
  /** The name stored with the selection at train time, or "" if none was. */
  name: string
  /** Capped document/conversation count, or null for "all of it". */
  count: number | null
}

/** Flattens this model’s selected training datasets across curriculum stages, deduplicating by identity. */
export function datasetsOf(raw: RawModel): LineageDatasetRef[] {
  const config = raw.parameters?.config
  if (!config) return []
  const selections: DatasetSelection[] =
    config.curriculum && config.curriculum.length > 0
      ? config.curriculum.flatMap((stage) => stage.datasets)
      : (config.datasets ?? [])
  const seen = new Set<string>()
  const out: LineageDatasetRef[] = []
  for (const d of selections) {
    if (seen.has(d.id)) continue
    seen.add(d.id)
    out.push({ id: d.id, name: d.name ?? "", count: d.count ?? null })
  }
  return out
}

/** Reads a local nonnegative parameter measurement without confusing an actual zero with absence. */
function measuredParameters(raw: RawModel): number | null {
  return (
    [raw.parameter_count, raw.parameters?.result?.parameters].find(
      (value) =>
        typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ) ?? null
  )
}

/** Returns recorded nonnegative parameters, or the standard legacy shape estimate; unavailable counts remain null. */
export function estimateParams(raw: RawModel): number | null {
  const measured = measuredParameters(raw)
  if (measured !== null) return measured
  const arch = architectureOf(raw)
  if (!arch) return null
  return estimateTrainingParameters(arch, vocabularySizeOf(raw))
}

/** Returns only an explicitly assigned size category; unknown values never acquire an automatic tier. */
export function tierFor(raw: RawModel): ModelTier | null {
  return raw.size === "fast" ||
    raw.size === "balanced" ||
    raw.size === "flagship"
    ? raw.size
    : null
}

/** Maps a registry response into shared card and picker facts without inventing unavailable measurements. */
export function mapModel(raw: RawModel): Model {
  if (raw.definition_id != null && !/^[0-9a-f]{64}$/.test(raw.definition_id))
    throw new Error("Model definition identity is invalid.")
  const arch = architectureOf(raw)
  const description =
    raw.description ??
    (arch
      ? `${arch.layers}-layer transformer, ${arch.dimensions}d`
      : "Custom trained model")
  return {
    id: raw.model_id,
    name: raw.name,
    family: raw.family ?? "Custom",
    tier: tierFor(raw),
    params: estimateParams(raw),
    parameterCountEstimated:
      measuredParameters(raw) === null && estimateParams(raw) !== null,
    contextWindow: raw.context ?? arch?.ctx_window ?? 0,
    description,
    published: raw.published,
    definitionId: raw.definition_id ?? null,
    supportsContextExtension: raw.context_extension_supported === true,
  }
}

/**
 * Removes one model: its row, its weights on disk, and the `id_model` on any
 * message that referenced it (detached, not deleted).
 *
 * Admin-only on the backend. Exported as a bare function rather than living on
 * `useModel` because the registry deletes models it is not holding the detail of —
 * see `ModelsGrid`'s batch flow, which needs to delete by id and nothing else.
 */
export async function deleteModel(id: string): Promise<void> {
  const res = await apiFetch(`/models/${id}`, { method: "DELETE" })
  if (!res.ok) {
    throw new Error((await res.text()) || "Failed to delete the model.")
  }
}

/** Maps registry models while retaining usable public rows whose private training provenance has been deliberately removed. */
export function modelsFromRegistry(rows: readonly RawModel[]): Model[] {
  return rows
    .filter((model) => model.has_weights || model.parameters)
    .map(mapModel)
}

/** Loads registry facts for one actual session/refresh identity, aborting replaced reads and masking old private models immediately. */
export function useModels(refreshKey?: unknown) {
  const token = useSessionToken()
  const [state, setState] = React.useState<{
    token: string
    refreshKey: unknown
    models: Model[]
    error: Error | null
  } | null>(null)
  React.useEffect(() => {
    if (!token) return
    const controller = new AbortController()
    let active = true
    void apiFetch("/models", {
      signal: controller.signal,
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(async (response) => {
        if (!response.ok) throw new Error("Unable to load models.")
        const data: { models: RawModel[] } = await response.json()
        if (!Array.isArray(data.models))
          throw new Error("Model registry response is invalid.")
        return modelsFromRegistry(data.models)
      })
      .then((models) => {
        if (active) setState({ token, refreshKey, models, error: null })
      })
      .catch((cause: unknown) => {
        if (active)
          setState({
            token,
            refreshKey,
            models: [],
            error:
              cause instanceof Error
                ? cause
                : new Error("Unable to load models."),
          })
      })
    return () => {
      active = false
      controller.abort()
    }
  }, [token, refreshKey])
  const current =
    state?.token === token && Object.is(state.refreshKey, refreshKey)
      ? state
      : null
  return {
    models: current?.models ?? [],
    isLoading: Boolean(token && !current),
    error: current?.error ?? null,
  }
}
