"use client"

import * as React from "react"

import { getCycles, type Cycle } from "@/lib/cycles"

/**
 * Every cycle, newest first.
 *
 * Polled rather than pushed: a cycle changes stage every few minutes at most, so a
 * socket would be plumbing for an event that fires rarely. The jobs board already
 * pushes; this sits above it and moves at a different speed.
 */
export function useCycles(intervalMs = 15000) {
  const [cycles, setCycles] = React.useState<Cycle[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const refresh = React.useCallback(async () => {
    try {
      setCycles(await getCycles())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load cycles.")
    } finally {
      setIsLoading(false)
    }
  }, [])

  React.useEffect(() => {
    let active = true
    const load = () => {
      getCycles()
        .then((next) => {
          if (active) {
            setCycles(next)
            setError(null)
          }
        })
        .catch((e: unknown) => {
          if (active) {
            setError(e instanceof Error ? e.message : "Failed to load cycles.")
          }
        })
        .finally(() => {
          if (active) setIsLoading(false)
        })
    }
    load()
    const timer = setInterval(load, intervalMs)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [intervalMs])

  return { cycles, isLoading, error, refresh }
}
