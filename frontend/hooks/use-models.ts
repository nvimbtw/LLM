"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import type { CurriculumStage, DatasetSelection } from "@/lib/training"
import { estimateTrainingParameters } from "@/lib/training-architecture"

export type ModelTier = "flagship" | "balanced" | "fast"

export interface Model {
  id: string
  name: string
  family: string
  tier: ModelTier | null
  /** Raw parameter count. 0 means unknown. */
  params: number
  /** Context window in tokens. */
  contextWindow: number
  description: string
  /** Whether the model is published (visible to non-admins). */
  published: boolean
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

/** Returns the resolved shape, preferring actual results over requested dimensions. Invalid or incomplete metadata stays unavailable. */
export function architectureOf(raw: RawModel): RawArchitecture | null {
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

/** Returns recorded parameters, or the shared training estimate from shape and measured vocabulary; zero means unavailable. */
export function estimateParams(raw: RawModel): number {
  const measured = [
    raw.parameter_count,
    raw.parameters?.result?.parameters,
  ].find(validCount)
  if (measured !== undefined) return measured
  const arch = architectureOf(raw)
  if (!arch) return 0
  return estimateTrainingParameters(arch, vocabularySizeOf(raw)) ?? 0
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
    contextWindow: raw.context ?? arch?.ctx_window ?? 0,
    description,
    published: raw.published,
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

/** Loads finished registry models for the supplied refresh key, ignoring obsolete reads. Checkpoints remain on their parent’s page. */
export function useModels(refreshKey?: unknown) {
  const [models, setModels] = React.useState<Model[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<Error | null>(null)

  const [loadedFor, setLoadedFor] = React.useState(refreshKey)
  if (loadedFor !== refreshKey) {
    setLoadedFor(refreshKey)
    setIsLoading(true)
  }

  React.useEffect(() => {
    let active = true
    apiFetch(`/models`)
      .then((res) => {
        if (!res.ok) throw new Error("Unable to load models.")
        return res.json() as Promise<{ models: RawModel[] }>
      })
      .then((data) => {
        if (active)
          setModels(data.models.filter((m) => m.parameters).map(mapModel))
      })
      .catch((err) => {
        if (active)
          setError(
            err instanceof Error ? err : new Error("Unable to load models.")
          )
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [refreshKey])

  return { models, isLoading, error }
}
