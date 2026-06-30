"use client"

import * as React from "react"

import { API_URL, authHeaders } from "@/lib/api"

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
}

// Shape returned by GET /model (see backend routes/model.rs ModelResponse).
interface RawModel {
  id_model: string
  name: string
  description: string | null
  parameters: {
    config?: {
      dimensions?: number
      ctx_window?: number
      layers?: number
      ff_multiplier?: number
      max_vocab?: number
    }
    dataset?: { vocab_size?: number }
  } | null
  context: number | null
  family: string | null
  created_at: string
}

// Rough transformer parameter count from the training config — enough for a
// meaningful "size" stat on the card. Returns 0 when the config isn't available.
function estimateParams(raw: RawModel): number {
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
function tierFor(raw: RawModel): ModelTier {
  const d = raw.parameters?.config?.dimensions ?? raw.context ?? 0
  if (d >= 256) return "flagship"
  if (d >= 96) return "balanced"
  return "fast"
}

function mapModel(raw: RawModel): Model {
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
  }
}

export function useModels() {
  const [models, setModels] = React.useState<Model[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<Error | null>(null)

  React.useEffect(() => {
    let active = true
    setIsLoading(true)
    fetch(`${API_URL}/model`, { headers: authHeaders() })
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
  }, [])

  return { models, isLoading, error }
}
