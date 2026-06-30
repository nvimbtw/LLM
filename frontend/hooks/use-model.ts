"use client"

import * as React from "react"

import { API_URL, authHeaders } from "@/lib/api"
import {
  estimateParams,
  mapModel,
  type Model,
  type RawModel,
  type RawModelConfig,
  type RawModelDataset,
} from "./use-models"

// The card-level Model plus the full training config/dataset and timestamps the
// detail page surfaces.
export interface ModelDetail extends Model {
  createdAt: string
  estimatedParams: number
  config: RawModelConfig | null
  dataset: RawModelDataset | null
}

function mapDetail(raw: RawModel): ModelDetail {
  return {
    ...mapModel(raw),
    createdAt: raw.created_at,
    estimatedParams: estimateParams(raw),
    config: raw.parameters?.config ?? null,
    dataset: raw.parameters?.dataset ?? null,
  }
}

export function useModel(id: string) {
  const [model, setModel] = React.useState<ModelDetail | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [notFound, setNotFound] = React.useState(false)
  const [error, setError] = React.useState<Error | null>(null)

  React.useEffect(() => {
    let active = true
    setIsLoading(true)
    setNotFound(false)
    setError(null)
    fetch(`${API_URL}/model/${id}`, { headers: authHeaders() })
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
          setError(err instanceof Error ? err : new Error("Unable to load this model."))
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
      const res = await fetch(`${API_URL}/model/${id}`, {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({ name, description, family }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || "Unable to update the model.")
      }
      setModel(mapDetail((await res.json()) as RawModel))
    },
    [id]
  )

  // Admin only. Removes the registry row and the on-disk weights.
  const remove = React.useCallback(async () => {
    const res = await fetch(`${API_URL}/model/${id}`, {
      method: "DELETE",
      headers: authHeaders(),
    })
    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || "Unable to delete the model.")
    }
  }, [id])

  return { model, isLoading, notFound, error, update, remove }
}
