"use client"

import * as React from "react"

import { getCorpus, type CorpusStats } from "@/lib/synth"

/**
 * The SFT corpus as it stands, measured against one model's context window.
 *
 * Refetched rather than streamed: the corpus only changes when a generation run
 * writes to it, which is an event the caller already knows about (a finished job),
 * so a socket here would be a second description of something the jobs socket
 * already reports.
 */
export function useCorpus(modelId?: string) {
  const [corpus, setCorpus] = React.useState<CorpusStats | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const refresh = React.useCallback(async () => {
    try {
      setCorpus(await getCorpus(modelId))
      setError(null)
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Failed to read the SFT corpus."
      )
    } finally {
      setIsLoading(false)
    }
  }, [modelId])

  React.useEffect(() => {
    void refresh()
  }, [refresh])

  return { corpus, isLoading, error, refresh }
}
