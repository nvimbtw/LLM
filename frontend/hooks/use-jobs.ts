"use client"

import * as React from "react"

import { toast } from "@/components/ui/toast"
import { useJobSubmission } from "./use-job-submission"
import {
  HISTORY_PAGE_SIZE,
  cancelJob,
  deleteJob,
  getBoard,
  reorderJob,
  retryJob,
  setPaused,
  type Job,
  type JobBoard,
} from "@/lib/jobs"

const EMPTY: JobBoard = {
  active: [],
  history: [],
  historyOffset: 0,
  historyLimit: HISTORY_PAGE_SIZE,
  historyTotal: 0,
  paused: false,
}

const POLL_MS = 3000

/**
 * Owns the queue board and keeps it live.
 *
 * Every change funnels through one refetch of `GET /runs`, so there is a single
 * description of a job rather than a REST shape and a socket shape that drift
 * apart.
 */
export function useJobs(enabled = true) {
  const submission = useJobSubmission("queue-retry")
  const [board, setBoard] = React.useState<JobBoard>(EMPTY)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const [historyPage, setHistoryPage] = React.useState(0)
  const pageRef = React.useRef(0)

  const generation = React.useRef(0)
  const lifetime = React.useRef({ active: false })
  React.useEffect(() => {
    const owner = { active: true }
    lifetime.current = owner
    return () => {
      owner.active = false
    }
  }, [])

  // The poll runs every few seconds for the life of a page. Most answers are
  // byte-identical to the one already on screen, and adopting an identical
  // board still re-renders every row, chart and control under it.
  const lastBoard = React.useRef<string | null>(null)

  /** Refreshes only this mounted queue owner and ignores obsolete pages/responses. */
  const refresh = React.useCallback(async () => {
    const owner = lifetime.current
    if (!enabled || !owner.active) return
    const mine = ++generation.current
    try {
      let next = await getBoard(pageRef.current * HISTORY_PAGE_SIZE)
      if (!owner.active || mine !== generation.current) return
      if (next.history.length === 0 && pageRef.current > 0) {
        pageRef.current = lastPage(next.historyTotal)
        setHistoryPage(pageRef.current)
        next = await getBoard(pageRef.current * HISTORY_PAGE_SIZE)
      }
      if (mine === generation.current && owner.active) {
        const serialized = JSON.stringify(next)
        if (serialized !== lastBoard.current) {
          lastBoard.current = serialized
          setBoard(next)
        }
        setError(null)
      }
    } catch (e) {
      if (owner.active && mine === generation.current) {
        setError(e instanceof Error ? e.message : "Failed to load the queue.")
      }
    } finally {
      if (owner.active && mine === generation.current) setIsLoading(false)
    }
  }, [enabled])

  React.useEffect(() => {
    if (!enabled) return
    void Promise.resolve().then(refresh)

    let timer: ReturnType<typeof setInterval> | undefined

    /** Starts a single visible-page polling timer. */
    const start = () => {
      if (timer === undefined) timer = setInterval(refresh, POLL_MS)
    }
    /** Stops this effect owner's polling timer. */
    const stop = () => {
      if (timer !== undefined) clearInterval(timer)
      timer = undefined
    }
    /** Pauses hidden-page polling and reconciles the current board when visible again. */
    const onVisibility = () => {
      if (document.hidden) {
        stop()
        return
      }
      void refresh()
      start()
    }

    if (!document.hidden) start()
    document.addEventListener("visibilitychange", onVisibility)
    return () => {
      document.removeEventListener("visibilitychange", onVisibility)
      stop()
    }
  }, [enabled, refresh])

  /** Performs an explicit mutation and reports/reconciles it only for its still-mounted session. */
  const act = React.useCallback(
    async (run: () => Promise<unknown>, done: string) => {
      const owner = lifetime.current
      if (!owner.active || !enabled) return
      try {
        await run()
        if (owner.active) toast.success(done)
      } catch (e) {
        if (owner.active)
          toast.error(
            "That didn't work",
            e instanceof Error ? e.message : "Something went wrong."
          )
      } finally {
        if (owner.active) await refresh()
      }
    },
    [refresh, enabled]
  )

  /** Requests cooperative cancellation through the existing engine queue owner. */
  const cancel = React.useCallback(
    (id: string) => act(() => cancelJob(id), "Cancelling the job."),
    [act]
  )
  /** Deletes a finished row and refreshes the authoritative page. */
  const remove = React.useCallback(
    (id: string) => act(() => deleteJob(id), "Job deleted."),
    [act]
  )
  /** Requests a new attempt over the saved job inputs. */
  const retry = React.useCallback(
    async (job: Job) => {
      if (!enabled) return
      const next = await submission.owner?.submit(
        `retry:${job.id_job}`,
        (request) => retryJob(job, request)
      )
      if (next) {
        toast.success("Re-queued.")
        submission.owner?.acknowledge()
        await refresh()
      }
    },
    [submission.owner, enabled, refresh]
  )

  /** Moves a queued row relative to its selected neighbour. */
  const reorder = React.useCallback(
    (id: string, afterId: string | null) =>
      act(() => reorderJob(id, afterId), "Queue reordered."),
    [act]
  )

  /** Changes admission pause without cancelling the active execution. */
  const pause = React.useCallback(
    (paused: boolean) =>
      act(
        () => setPaused(paused),
        paused ? "Queue paused. The running job carries on." : "Queue resumed."
      ),
    [act]
  )

  /** Loads a bounded server page rather than retaining unbounded history. */
  const goToHistoryPage = React.useCallback(
    (page: number) => {
      const next = Math.max(0, page)
      pageRef.current = next
      setHistoryPage(next)
      void refresh()
    },
    [refresh]
  )

  const visibleBoard = enabled ? board : EMPTY
  const historyPages = lastPage(visibleBoard.historyTotal) + 1

  /** Identifies execution still holding or releasing its active lease. */
  const isLive = (job: Job) =>
    job.status === "running" || job.status === "cancelling"
  const running: Job | null = visibleBoard.active.find(isLive) ?? null
  const queued = visibleBoard.active.filter((j) => j.status === "queued")
  const waiting = visibleBoard.active.filter((j) => j.status === "waiting")

  return {
    board: visibleBoard,
    running,
    queued,
    waiting,
    history: visibleBoard.history,
    historyTotal: visibleBoard.historyTotal,
    historyPage,
    historyPages,
    goToHistoryPage,
    paused: visibleBoard.paused,
    isLoading: enabled && isLoading,
    error: enabled ? (submission.state?.error ?? error) : null,
    refresh,
    cancel,
    remove,
    retry,
    reorder,
    pause,
  }
}

/** Zero-based index of the last page of `total` finished jobs. */
function lastPage(total: number): number {
  return Math.max(0, Math.ceil(total / HISTORY_PAGE_SIZE) - 1)
}
