"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import {
  estimateParams,
  mapModel,
  type Model,
  type RawModel,
  type RawModelConfig,
  type RawModelResult,
} from "./use-models"

// The card-level Model plus the full training config, run summary and timestamps
// the detail page surfaces.
export interface ModelDetail extends Model {
  createdAt: string
  estimatedParams: number
  config: RawModelConfig | null
  dataset: RawModelResult | null
  /**
   * Which of its run's saves this is, and where in the run it was taken — null
   * on an ordinary model. A non-null `checkpointIndex` is the whole definition
   * of "this row is an intermediate", and the page says so rather than
   * presenting a checkpoint as a peer of the model it is an intermediate of.
   */
  checkpointIndex: number | null
  checkpointStep: number | null
  /** The finished model this is an intermediate of, when it is one. */
  parentId: string | null
}

function mapDetail(raw: RawModel): ModelDetail {
  return {
    ...mapModel(raw),
    createdAt: raw.created_at ?? "",
    estimatedParams: estimateParams(raw),
    config: raw.parameters?.config ?? null,
    dataset: raw.parameters?.result ?? null,
    checkpointIndex: raw.checkpoint_index,
    checkpointStep: raw.checkpoint_step,
    parentId: raw.parent_id,
  }
}

export function useModel(id: string) {
  const [model, setModel] = React.useState<ModelDetail | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [notFound, setNotFound] = React.useState(false)
  const [error, setError] = React.useState<Error | null>(null)

  // Reset during render rather than in the effect. Switching model ids has to
  // clear the previous model's outcome *before* anything paints, or the page
  // shows the last model's 404 (or its data) for one frame while the new fetch
  // is in flight. React re-runs this render immediately and discards the
  // in-progress output, so nothing downstream ever sees the mismatched pair.
  const [loadedFor, setLoadedFor] = React.useState(id)
  if (loadedFor !== id) {
    setLoadedFor(id)
    setIsLoading(true)
    setNotFound(false)
    setError(null)
  }

  React.useEffect(() => {
    let active = true
    apiFetch(`/models/${id}`)
      .then(async (res) => {
        if (res.status === 404) {
          if (active) setNotFound(true)
          return null
        }
        if (!res.ok) throw new Error("Unable to load this model.")
        return (await res.json()) as RawModel
      })
      .then((raw) => {
        if (active && raw) setModel(mapDetail(raw))
      })
      .catch((err) => {
        if (active)
          setError(
            err instanceof Error ? err : new Error("Unable to load this model.")
          )
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [id])

  // Admin only. Sends the fields; the backend COALESCEs unchanged ones.
  const update = React.useCallback(
    async (name: string, description: string, family: string) => {
      const res = await apiFetch(`/models/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ display_name: name, description, family }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || "Unable to update the model.")
      }
      setModel(mapDetail((await res.json()) as RawModel))
    },
    [id]
  )

  // Admin only. Publish/unpublish toggle.
  const setPublished = React.useCallback(
    async (published: boolean) => {
      const res = await apiFetch(`/models/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ published }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || "Unable to update visibility.")
      }
      setModel(mapDetail((await res.json()) as RawModel))
    },
    [id]
  )

  // Admin only. Removes the registry row and the on-disk weights.
  const remove = React.useCallback(async () => {
    const res = await apiFetch(`/models/${id}`, { method: "DELETE" })
    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || "Unable to delete the model.")
    }
  }, [id])

  return { model, isLoading, notFound, error, update, setPublished, remove }
}
