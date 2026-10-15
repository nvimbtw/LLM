"use client"

import * as React from "react"

import { listTokenizers, type TokenizerInfo } from "@/lib/tokenizers"

/**
 * Every built vocabulary. Shaped like `use-models`, and refetched the same way:
 * bump `refreshKey` after a build lands or one is deleted.
 */
export function useTokenizers(refreshKey?: unknown) {
  const [tokenizers, setTokenizers] = React.useState<TokenizerInfo[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<Error | null>(null)

  const [loadedFor, setLoadedFor] = React.useState(refreshKey)
  if (loadedFor !== refreshKey) {
    setLoadedFor(refreshKey)
    setIsLoading(true)
  }

  React.useEffect(() => {
    let active = true
    listTokenizers()
      .then((next) => {
        if (active) {
          setTokenizers(next)
          setError(null)
        }
      })
      .catch((err: unknown) => {
        if (active) {
          setError(
            err instanceof Error ? err : new Error("Unable to load tokenizers.")
          )
        }
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [refreshKey])

  return { tokenizers, isLoading, error }
}
