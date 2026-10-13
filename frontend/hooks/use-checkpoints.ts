"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import type { RawModel } from "./use-models"

/**
 * One intermediate save of a training run, as its parent's page reads it.
 *
 * A checkpoint is a model — its own id, its own weights, its own detail page —
 * so this is a narrowed view of the same row, not a second kind of thing. What
 * distinguishes it is the two positions: `index` is which of the run's saves it
 * was, `step` is where in the run it was taken.
 */
export interface Checkpoint {
  id: string
  name: string
  index: number
  step: number
  /**
   * The loss at the step it was taken, or null when nothing recorded one.
   *
   * Null is the ordinary case today: the LLM engine writes this into the
   * checkpoint's `meta.json` but does not carry it onto the row, so the wire
   * field is absent. It must render as "—" and never as 0 — a zero loss reads
   * as a perfect model.
   */
  loss: number | null
  createdAt: string
  hasWeights: boolean
  kept: boolean
}

function mapCheckpoint(raw: RawModel): Checkpoint {
  const recorded = raw.parameters?.result?.checkpoint
  return {
    id: raw.model_id,
    name: raw.name,
    index: raw.checkpoint_index ?? recorded?.index ?? 0,
    step: raw.checkpoint_step ?? recorded?.step ?? 0,
    loss: typeof recorded?.loss === "number" ? recorded.loss : null,
    createdAt: raw.created_at ?? "",
    hasWeights: raw.has_weights,
    kept: raw.recovery_kept ?? false,
  }
}

/**
 * A model's checkpoints, ordered by step — the LLM engine orders them, and the
 * order is not re-derived here.
 *
 * `null` for `modelId` fetches nothing and reports an empty list, so a caller
 * that has not picked a model yet needs no branch of its own.
 */
export function useCheckpoints(modelId: string | null) {
  const [checkpoints, setCheckpoints] = React.useState<Checkpoint[]>([])
  const [isLoading, setIsLoading] = React.useState(Boolean(modelId))
  const [error, setError] = React.useState<Error | null>(null)

  // Reset during render rather than in the effect, the same way `use-model` does:
  // switching models has to clear the previous one's list before anything paints,
  // or a checkpoint of the model you just left renders under the one you opened.
  const [loadedFor, setLoadedFor] = React.useState(modelId)
  if (loadedFor !== modelId) {
    setLoadedFor(modelId)
    setCheckpoints([])
    setIsLoading(Boolean(modelId))
    setError(null)
  }

  React.useEffect(() => {
    if (!modelId) return
    let active = true
    apiFetch(`/models/${modelId}/checkpoints`)
      .then((res) => {
        if (!res.ok) throw new Error("Unable to load this model's checkpoints.")
        return res.json() as Promise<{ checkpoints: RawModel[] }>
      })
      .then((data) => {
        if (active) setCheckpoints((data.checkpoints ?? []).map(mapCheckpoint))
      })
      .catch((err) => {
        if (active)
          setError(
            err instanceof Error
              ? err
              : new Error("Unable to load this model's checkpoints.")
          )
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [modelId])

  return { checkpoints, isLoading, error }
}
