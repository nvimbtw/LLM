"use client"

import * as React from "react"

import {
  fetchAttention,
  type AttentionMatrix,
  type AttentionQuery,
} from "@/lib/attention"

export function useAttention(modelId: string) {
  const [matrix, setMatrix] = React.useState<AttentionMatrix | null>(null)
  const [isLoading, setIsLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  // Nothing is fetched on mount, and that is the feature rather than laziness:
  // materializing a matrix costs the engine a forward pass with the fused kernel
  // switched off, so it happens when someone asks and at no other time.
  const load = React.useCallback(
    async (query: AttentionQuery) => {
      setIsLoading(true)
      setError(null)
      try {
        setMatrix(await fetchAttention(modelId, query))
      } catch (err) {
        // The previous matrix is dropped rather than left on screen under new
        // layer/head labels that no longer describe it.
        setMatrix(null)
        setError(
          err instanceof Error
            ? err.message
            : "Unable to read this model's attention."
        )
      } finally {
        setIsLoading(false)
      }
    },
    [modelId]
  )

  return { matrix, isLoading, error, load }
}
