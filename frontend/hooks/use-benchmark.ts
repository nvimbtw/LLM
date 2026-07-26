"use client"

import * as React from "react"

import {
  type BenchmarkRun,
  type BenchmarkRunDetail,
  type SuiteInfo,
  deleteRun as apiDeleteRun,
  getRun,
  getSuite,
  listRuns,
} from "@/lib/benchmark"

// Poll interval while a run is in flight. This page is history: it only needs to
// notice a run *finishing* so it can select the fresh report. Live progress is on
// /jobs, which is pushed rather than polled — so a couple of seconds is plenty,
// and polling stops entirely once nothing is running.
const POLL_MS = 2000

export function useBenchmark() {
  const [runs, setRuns] = React.useState<BenchmarkRun[]>([])
  const [suite, setSuite] = React.useState<SuiteInfo | null>(null)
  const [selectedId, setSelectedId] = React.useState<string | null>(null)
  // The last report fetched, whichever run it belongs to. `detail` below narrows
  // it to the current selection, so switching runs shows nothing rather than the
  // previous run's report — without having to clear state from an effect.
  const [fetched, setFetched] = React.useState<BenchmarkRunDetail | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const active = runs.find((run) => run.status === "running") ?? null

  const refresh = React.useCallback(async () => {
    const list = await listRuns()
    setRuns(list)
    return list
  }, [])

  // Initial load: history plus whether the reference corpus is installed.
  React.useEffect(() => {
    let alive = true
    Promise.all([listRuns(), getSuite()])
      .then(([list, info]) => {
        if (!alive) return
        setRuns(list)
        setSuite(info)
        // Land on the newest finished run so the page has something to show
        // instead of an empty frame.
        setSelectedId((current) => current ?? list.find((r) => r.status === "done")?.id_run ?? null)
      })
      .catch((err) => {
        if (alive) setError(err instanceof Error ? err.message : "Failed to load benchmarks.")
      })
      .finally(() => {
        if (alive) setIsLoading(false)
      })
    return () => {
      alive = false
    }
  }, [])

  // Poll only while something is running.
  React.useEffect(() => {
    if (!active) return
    let alive = true
    const timer = setInterval(() => {
      listRuns()
        .then((list) => {
          if (!alive) return
          setRuns(list)
          // The moment the run we were watching finishes, select it — the whole
          // point of starting a run is to see its report.
          const finished = list.find((r) => r.id_run === active.id_run && r.status !== "running")
          if (finished) setSelectedId(finished.id_run)
        })
        .catch(() => {
          // A transient poll failure isn't worth surfacing; the next tick retries
          // and a genuine outage shows up on the next user action.
        })
    }, POLL_MS)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [active])

  // Fetch the full report whenever the selection changes. Summaries in the list
  // deliberately omit it (a report embeds generated text), so it's loaded on demand.
  React.useEffect(() => {
    if (!selectedId) return
    let alive = true
    getRun(selectedId)
      .then((run) => {
        if (alive) setFetched(run)
      })
      .catch((err) => {
        if (alive) setError(err instanceof Error ? err.message : "Failed to load run.")
      })
    return () => {
      alive = false
    }
  }, [selectedId, runs])

  const remove = React.useCallback(
    async (id: string) => {
      setError(null)
      try {
        await apiDeleteRun(id)
        const list = await refresh()
        setSelectedId((current) =>
          current === id ? (list.find((r) => r.status === "done")?.id_run ?? null) : current
        )
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to delete run.")
      }
    },
    [refresh]
  )

  return {
    runs,
    suite,
    active,
    detail: fetched?.id_run === selectedId ? fetched : null,
    selectedId,
    select: setSelectedId,
    remove,
    isLoading,
    error,
    dismissError: () => setError(null),
  }
}
