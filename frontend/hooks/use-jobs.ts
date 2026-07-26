"use client"

import * as React from "react"

import {
  cancelJob,
  deleteJob,
  getBoard,
  reorderJob,
  retryJob,
  setPaused,
  type Job,
  type JobBoard,
} from "@/lib/jobs"
import { watchBoard } from "@/lib/jobs-socket"

const EMPTY: JobBoard = { active: [], history: [], paused: false }

/**
 * Owns the queue board and keeps it live.
 *
 * The socket carries nudges rather than rows, so every change funnels through one
 * refetch of `GET /jobs`. That costs a round trip per change — which is rare, the
 * board only moves when a job is queued, starts, or ends — and buys a single
 * description of a job instead of a REST shape and a socket shape drifting apart.
 */
export function useJobs(enabled = true) {
  const [board, setBoard] = React.useState<JobBoard>(EMPTY)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  // Guards against a slow refetch landing after a newer one and reviving stale
  // rows — the socket can fire several nudges in quick succession.
  const generation = React.useRef(0)

  const refresh = React.useCallback(async () => {
    const mine = ++generation.current
    try {
      const next = await getBoard()
      if (mine === generation.current) {
        setBoard(next)
        setError(null)
      }
    } catch (e) {
      if (mine === generation.current) {
        setError(e instanceof Error ? e.message : "Failed to load the queue.")
      }
    } finally {
      if (mine === generation.current) setIsLoading(false)
    }
  }, [])

  // Advances the running job's progress in place. Only the two fields the event
  // carries are touched, so a refetch racing this can't be undone by it.
  const patchProgress = React.useCallback(
    (idJob: string, progress: number, stage: string | null) => {
      setBoard((prev) => {
        const index = prev.active.findIndex((job) => job.id_job === idJob)
        if (index === -1) return prev
        const active = [...prev.active]
        active[index] = { ...active[index], progress, stage }
        return { ...prev, active }
      })
    },
    []
  )

  React.useEffect(() => {
    // The queue is admin-only, so a non-admin session must not fetch it or open
    // a socket for it — both would just collect 403s for the whole session.
    if (!enabled) {
      setIsLoading(false)
      return
    }
    void refresh()
    const stop = watchBoard({ onChanged: refresh, onProgress: patchProgress })
    return stop
  }, [enabled, refresh, patchProgress])

  // Every mutation refetches rather than patching local state: the server decides
  // ordering and status, and a queued job the runner claimed mid-request would
  // make an optimistic patch wrong in exactly the moment it matters.
  const act = React.useCallback(
    async (run: () => Promise<unknown>) => {
      try {
        await run()
        setError(null)
      } catch (e) {
        setError(e instanceof Error ? e.message : "Something went wrong.")
      } finally {
        await refresh()
      }
    },
    [refresh]
  )

  const cancel = React.useCallback((id: string) => act(() => cancelJob(id)), [act])
  const remove = React.useCallback((id: string) => act(() => deleteJob(id)), [act])
  const retry = React.useCallback((id: string) => act(() => retryJob(id)), [act])

  const reorder = React.useCallback(
    (id: string, afterId: string | null) => act(() => reorderJob(id, afterId)),
    [act]
  )

  const pause = React.useCallback(
    (paused: boolean) => act(() => setPaused(paused)),
    [act]
  )

  const running: Job | null =
    board.active.find((j) => j.status === "running" || j.status === "cancelling") ?? null
  const queued = board.active.filter((j) => j.status === "queued")

  return {
    board,
    running,
    queued,
    history: board.history,
    paused: board.paused,
    isLoading,
    error,
    refresh,
    cancel,
    remove,
    retry,
    reorder,
    pause,
  }
}
