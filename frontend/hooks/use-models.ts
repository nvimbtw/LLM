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

// Shape returned by GET /models. `parameters` is the training run's stored
// config and the summary it finished with.
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

// What one intermediate save recorded about itself, as the engine's
// `CheckpointWriter` writes it — the run position it was taken at, and the loss
// there. Every field is optional because it is only present on a checkpoint, and
// only for a checkpoint whose row carries its `parameters` at all.
export interface RawCheckpointResult {
  index?: number
  step?: number
  total_steps?: number
  epoch?: number
  stage?: number
  loss?: number | null
}

export interface RawModelResult {
  files?: number
  chars?: number
  vocab_size?: number
  merges?: number
  corpus_len?: number
  checkpoint?: RawCheckpointResult
}

export interface RawModel {
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
  // The run that produced this model, and — on an intermediate save — the
  // finished model it is an intermediate of and where in the run it was taken.
  // `checkpoint_index` being non-null is the whole definition of "this is a
  // checkpoint"; all four are null on an ordinary model.
  run_id: string | null
  parent_id: string | null
  checkpoint_step: number | null
  checkpoint_index: number | null
}

// Rough transformer parameter count from the training config — enough for a
// meaningful "size" stat on the card. Returns 0 when the config isn't available.
export function estimateParams(raw: RawModel): number {
  const c = raw.parameters?.config
  if (!c?.dimensions || !c.layers) return 0
  const d = c.dimensions
  const ctx = c.ctx_window ?? 0
  const ff = c.ff_multiplier ?? 4
  const vocab = raw.parameters?.result?.vocab_size ?? c.max_vocab ?? 0
  // embeddings (token in + out) + positions + per-layer attention & FFN
  const perLayer = d * d * (4 + 2 * ff)
  return Math.round(2 * vocab * d + ctx * d + c.layers * perLayer)
}

// Bucket by parameter count: flagship > 100M, balanced > 50M, fast below.
export function tierFor(raw: RawModel): ModelTier {
  const params = estimateParams(raw)
  if (params >= 100_000_000) return "flagship"
  if (params >= 50_000_000) return "balanced"
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
    id: raw.model_id,
    name: raw.name,
    family: raw.family ?? "Custom",
    tier: tierFor(raw),
    params: estimateParams(raw),
    contextWindow: raw.context ?? 0,
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

// `refreshKey` re-fetches the model list whenever it changes — pass a value that
// advances when a new model is registered (e.g. after a training run finishes) so
// the just-trained model appears without a page reload.
//
// This is the only place the model list is fetched, and it asks for `/models`
// bare. That matters: the engine leaves a run's ~10 intermediate checkpoints out
// unless a request sets `checkpoints=true`, and nothing here ever should — one
// run would otherwise outnumber every finished model in the grid, the chat
// picker and the benchmark composer at once. A checkpoint is offered from the
// one place it belongs to, its parent's page (`useCheckpoints`).
export function useModels(refreshKey?: unknown) {
  const [models, setModels] = React.useState<Model[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<Error | null>(null)

  // See `use-model.ts` — the same reset-during-render, keyed on `refreshKey`, so
  // an advancing key puts the list back into its loading state without a
  // cascading render out of the effect.
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
        // A directory with no row has no id to address, so it is not offered.
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
