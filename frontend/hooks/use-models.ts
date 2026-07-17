"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"

export type ModelTier = "flagship" | "balanced" | "fast"

export interface Model {
  id: string
  name: string
  family: string
  tier: ModelTier
  /** Raw parameter count. 0 means unknown. */
  params: number
  /** Context window in tokens. */
  contextWindow: number
  description: string
  /** Whether the model is published (visible to non-admins). */
  published: boolean
}

// Shape returned by GET /model (see backend routes/model.rs ModelResponse).
// `parameters` is the training run's stored config + dataset stats.
export interface RawModelConfig {
  dimensions?: number
  ctx_window?: number
  heads?: number
  layers?: number
  ff_multiplier?: number
  max_vocab?: number
  lr?: number
  epochs?: number
}

export interface RawModelDataset {
  files?: number
  chars?: number
  vocab_size?: number
  merges?: number
  corpus_len?: number
}

export interface RawModel {
  id_model: string
  name: string
  description: string | null
  parameters: {
    config?: RawModelConfig
    dataset?: RawModelDataset
  } | null
  context: number | null
  family: string | null
  published: boolean
  created_at: string
}

// Rough transformer parameter count from the training config — enough for a
// meaningful "size" stat on the card. Returns 0 when the config isn't available.
export function estimateParams(raw: RawModel): number {
  const c = raw.parameters?.config
  if (!c?.dimensions || !c.layers) return 0
  const d = c.dimensions
  const ctx = c.ctx_window ?? 0
  const ff = c.ff_multiplier ?? 4
  const vocab = raw.parameters?.dataset?.vocab_size ?? c.max_vocab ?? 0
  // embeddings (token in + out) + positions + per-layer attention & FFN
  const perLayer = d * d * (4 + 2 * ff)
  return Math.round(2 * vocab * d + ctx * d + c.layers * perLayer)
}

// Bucket by hidden size so the card glyph/badge varies with model scale.
export function tierFor(raw: RawModel): ModelTier {
  const d = raw.parameters?.config?.dimensions ?? raw.context ?? 0
  if (d >= 256) return "flagship"
  if (d >= 96) return "balanced"
  return "fast"
}

export function mapModel(raw: RawModel): Model {
  const c = raw.parameters?.config
  const description =
    raw.description ??
    (c?.layers && c.dimensions
      ? `${c.layers}-layer transformer, ${c.dimensions}d`
      : "Custom trained model")
  return {
    id: raw.id_model,
    name: raw.name,
    family: raw.family ?? "Custom",
    tier: tierFor(raw),
    params: estimateParams(raw),
    contextWindow: raw.context ?? 0,
    description,
    published: raw.published,
  }
}

// `refreshKey` re-fetches the model list whenever it changes — pass a value that
// advances when a new model is registered (e.g. after a training run finishes) so
// the just-trained model appears without a page reload.
export function useModels(refreshKey?: unknown) {
  const [models, setModels] = React.useState<Model[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<Error | null>(null)

  React.useEffect(() => {
    let active = true
    setIsLoading(true)
    apiFetch(`/model`)
      .then((res) => {
        if (!res.ok) throw new Error("Unable to load models.")
        return res.json() as Promise<RawModel[]>
      })
      .then((data) => {
        if (active) setModels(data.map(mapModel))
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err : new Error("Unable to load models."))
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
