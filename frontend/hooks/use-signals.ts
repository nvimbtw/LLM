"use client"

import * as React from "react"

import { getSignalStatus, type SignalStatus } from "@/lib/signals"

/**
 * How much preference signal exists, and whether a KTO run can start.
 *
 * `refreshKey` re-reads the count when something upstream changed it — a rating
 * saved in the panel, a gather job finishing. The count is small and read rarely,
 * so it refetches rather than trying to keep a local tally in sync with two
 * sources that both write to the server.
 */
export function useSignals(refreshKey?: unknown) {
  const [status, setStatus] = React.useState<SignalStatus | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const refresh = React.useCallback(async () => {
    try {
      setStatus(await getSignalStatus())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load the signal count.")
    } finally {
      setIsLoading(false)
    }
  }, [])

  // The fetch is inlined rather than calling `refresh` here so no state is written
  // synchronously from the effect body; `refresh` stays for callers that want to
  // re-read on demand.
  React.useEffect(() => {
    let active = true
    getSignalStatus()
      .then((next) => {
        if (!active) return
        setStatus(next)
        setError(null)
      })
      .catch((e: unknown) => {
        if (!active) return
        setError(e instanceof Error ? e.message : "Failed to load the signal count.")
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [refreshKey])

  return { status, isLoading, error, refresh }
}
