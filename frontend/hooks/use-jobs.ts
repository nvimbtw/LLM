"use client"

import * as React from "react"

import { toast } from "@/components/ui/toast"
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
import { watchBoard } from "@/lib/jobs-socket"

const EMPTY: JobBoard = {
  active: [],
  history: [],
  historyOffset: 0,
  historyLimit: HISTORY_PAGE_SIZE,
  historyTotal: 0,
  paused: false,
}

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

  // Which page of history is on screen. Held here rather than in the view because
  // every queue nudge refetches the board, and the refetch has to ask for the page
  // the reader is on — otherwise finishing a job would silently snap them back to
  // the first page.
  const [historyPage, setHistoryPage] = React.useState(0)
  // Read by `refresh`, which is a stable callback the socket holds for the life of
  // the subscription: a page in its dependency list would tear down and rebuild the
  // socket on every page turn.
  const pageRef = React.useRef(0)

  // Guards against a slow refetch landing after a newer one and reviving stale
  // rows — the socket can fire several nudges in quick succession.
  const generation = React.useRef(0)

  const refresh = React.useCallback(async () => {
    const mine = ++generation.current
    try {
      let next = await getBoard(pageRef.current * HISTORY_PAGE_SIZE)
      // A page can empty out under the reader: deleting its last row, or
      // re-queueing a job, moves the whole tail up by one. Land on the last page
      // that does exist rather than on a blank one under a pager insisting there is
      // something there. The response says how many there are, so this corrects in
      // one hop rather than walking back a page at a time.
      if (next.history.length === 0 && pageRef.current > 0) {
        pageRef.current = lastPage(next.historyTotal)
        setHistoryPage(pageRef.current)
        next = await getBoard(pageRef.current * HISTORY_PAGE_SIZE)
      }
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
  //
  // Outcomes are toasts, not the `error` banner above — that one is reserved for
  // "the board could not be loaded", which is a state of the page. Whether a
  // cancel took is an event, and an event that silently repaints a row four
  // seconds later is indistinguishable from nothing happening.
  const act = React.useCallback(
    async (run: () => Promise<unknown>, done: string) => {
      try {
        await run()
        toast.success(done)
      } catch (e) {
        toast.error(
          "That didn't work",
          e instanceof Error ? e.message : "Something went wrong."
        )
      } finally {
        await refresh()
      }
    },
    [refresh]
  )

  const cancel = React.useCallback(
    (id: string) => act(() => cancelJob(id), "Cancelling the job."),
    [act]
  )
  const remove = React.useCallback(
    (id: string) => act(() => deleteJob(id), "Job deleted."),
    [act]
  )
  const retry = React.useCallback(
    (id: string) => act(() => retryJob(id), "Re-queued."),
    [act]
  )

  const reorder = React.useCallback(
    (id: string, afterId: string | null) =>
      act(() => reorderJob(id, afterId), "Queue reordered."),
    [act]
  )

  const pause = React.useCallback(
    (paused: boolean) =>
      act(
        () => setPaused(paused),
        paused ? "Queue paused. The running job carries on." : "Queue resumed."
      ),
    [act]
  )

  // Turning a page is a refetch, not a slice of something already held: the client
  // only ever has one page, which is the point of paging it server-side.
  const goToHistoryPage = React.useCallback(
    (page: number) => {
      const next = Math.max(0, page)
      pageRef.current = next
      setHistoryPage(next)
      void refresh()
    },
    [refresh]
  )

  const historyPages = lastPage(board.historyTotal) + 1

  // One running job *per lane*, not one overall: the two runners proceed
  // independently, so a generation run and a training run are both "now", and
  // picking the first of them would leave the other showing nowhere on the page.
  const isLive = (job: Job) =>
    job.status === "running" || job.status === "cancelling"
  const running: Job | null =
    board.active.find((j) => isLive(j) && j.lane === "device") ?? null
  const runningNetwork: Job | null =
    board.active.find((j) => isLive(j) && j.lane === "network") ?? null
  const queued = board.active.filter((j) => j.status === "queued")

  return {
    board,
    running,
    runningNetwork,
    queued,
    history: board.history,
    historyTotal: board.historyTotal,
    historyPage,
    historyPages,
    goToHistoryPage,
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

/** Zero-based index of the last page of `total` finished jobs. */
function lastPage(total: number): number {
  return Math.max(0, Math.ceil(total / HISTORY_PAGE_SIZE) - 1)
}
