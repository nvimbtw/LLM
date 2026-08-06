"use client"

import * as React from "react"

import { toast } from "@/components/ui/toast"
import {
  getCycle,
  setCycleStatus,
  type CycleDetail,
  type CycleStatus,
} from "@/lib/cycles"

// What each transition actually did, which is not what the verb implies: neither
// pausing nor halting interrupts the stage already on the GPU.
const STATUS_DONE: Record<"running" | "paused" | "halted", string> = {
  running: "Cycle resumed.",
  paused: "Cycle paused after the current stage.",
  halted: "Cycle stopping. The best model so far is kept.",
}

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

  const live = cycle
    ? cycle.status === "running" || cycle.status === "paused"
    : true

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
            setError(
              e instanceof Error ? e.message : "Failed to load the cycle."
            )
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
        // The page's own `error` stays for "the cycle could not be loaded". This
        // is an outcome, and one whose effect is delayed — pausing lets the
        // current stage finish — so it has to say so rather than look inert.
        toast.success(STATUS_DONE[status])
        await refresh()
      } catch (e) {
        toast.error(
          "Couldn't update the cycle",
          e instanceof Error ? e.message : "Something went wrong."
        )
      }
    },
    [idCycle, refresh]
  )

  return { cycle, isLoading, error, refresh, setStatus }
}
