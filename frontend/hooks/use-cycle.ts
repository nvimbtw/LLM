"use client"

import * as React from "react"

import {
  getCycle,
  setCycleStatus,
  type CycleDetail,
  type CycleStatus,
} from "@/lib/cycles"

/**
 * One cycle, with its rounds and the jobs it has spawned.
 *
 * Polls while the cycle is live and stops once it has finished — a halted cycle is
 * a record, and re-reading it forever would be a request every few seconds for a
 * page that can no longer change.
 */
export function useCycle(idCycle: string, intervalMs = 8000) {
  const [cycle, setCycle] = React.useState<CycleDetail | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const live = cycle ? cycle.status === "running" || cycle.status === "paused" : true

  React.useEffect(() => {
    let active = true
    const load = () => {
      getCycle(idCycle)
        .then((next) => {
          if (active) {
            setCycle(next)
            setError(null)
          }
        })
        .catch((e: unknown) => {
          if (active) {
            setError(e instanceof Error ? e.message : "Failed to load the cycle.")
          }
        })
        .finally(() => {
          if (active) setIsLoading(false)
        })
    }
    load()
    if (!live) {
      return () => {
        active = false
      }
    }
    const timer = setInterval(load, intervalMs)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [idCycle, intervalMs, live])

  const refresh = React.useCallback(async () => {
    try {
      setCycle(await getCycle(idCycle))
    } catch {
      // A failed refresh leaves the last good view up; the poll will retry.
    }
  }, [idCycle])

  const setStatus = React.useCallback(
    async (status: Extract<CycleStatus, "running" | "paused" | "halted">) => {
      try {
        await setCycleStatus(idCycle, status)
        await refresh()
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to update the cycle.")
      }
    },
    [idCycle, refresh]
  )

  return { cycle, isLoading, error, refresh, setStatus }
}
