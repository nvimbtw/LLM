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

  // The fetch is inlined rather than calling `refresh` here so no state is written
  // synchronously from the effect body; `refresh` stays for callers that want to
  // re-read on demand. Same shape as `use-signals.ts`. The `active` flag is not
  // just lint appeasement — switching model mid-flight would otherwise let the
  // previous model's corpus land last and win.
  React.useEffect(() => {
    let active = true
    getCorpus(modelId)
      .then((next) => {
        if (!active) return
        setCorpus(next)
        setError(null)
      })
      .catch((e: unknown) => {
        if (!active) return
        setError(
          e instanceof Error ? e.message : "Failed to read the SFT corpus."
        )
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [modelId])

  return { corpus, isLoading, error, refresh }
}
