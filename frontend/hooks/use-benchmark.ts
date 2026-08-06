"use client"

import * as React from "react"

import { toast } from "@/components/ui/toast"
import {
  type BenchmarkRun,
  type BenchmarkRunDetail,
  type BenchmarkSuite,
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
  // Which of the two benchmarks the page is showing. Everything below is scoped to
  // it: the history, the selection, and — importantly — `active`. Without the
  // scoping, queueing both suites for one model would show the markup run's spinner
  // over the language page and then select the wrong report when it finished.
  const [selected, setSelected] = React.useState<BenchmarkSuite>("language")
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
    const list = await listRuns(selected)
    setRuns(list)
    return list
  }, [selected])

  // Switching suite clears the selection and the fetched report here, in the
  // handler, rather than from the effect below: a language run id means nothing in
  // the markup history, and clearing it in an effect would cascade a render.
  const selectSuite = React.useCallback(
    (next: BenchmarkSuite) => {
      if (next === selected) return
      setSelected(next)
      setRuns([])
      setSelectedId(null)
      setFetched(null)
      setIsLoading(true)
    },
    [selected]
  )

  // History for the chosen suite, plus what the suite has installed on disk.
  React.useEffect(() => {
    let alive = true
    Promise.all([listRuns(selected), getSuite()])
      .then(([list, info]) => {
        if (!alive) return
        setRuns(list)
        setSuite(info)
        // Land on the newest finished run so the page has something to show
        // instead of an empty frame.
        setSelectedId(list.find((r) => r.status === "done")?.id_run ?? null)
      })
      .catch((err) => {
        if (alive)
          setError(
            err instanceof Error ? err.message : "Failed to load benchmarks."
          )
      })
      .finally(() => {
        if (alive) setIsLoading(false)
      })
    return () => {
      alive = false
    }
  }, [selected])

  // Poll only while something is running.
  React.useEffect(() => {
    if (!active) return
    let alive = true
    const timer = setInterval(() => {
      listRuns(selected)
        .then((list) => {
          if (!alive) return
          setRuns(list)
          // The moment the run we were watching finishes, select it — the whole
          // point of starting a run is to see its report.
          const finished = list.find(
            (r) => r.id_run === active.id_run && r.status !== "running"
          )
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
  }, [active, selected])

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
        if (alive)
          setError(err instanceof Error ? err.message : "Failed to load run.")
      })
    return () => {
      alive = false
    }
  }, [selectedId, runs])

  const remove = React.useCallback(
    async (id: string) => {
      try {
        await apiDeleteRun(id)
        toast.success("Run deleted.")
        const list = await refresh()
        setSelectedId((current) =>
          current === id
            ? (list.find((r) => r.status === "done")?.id_run ?? null)
            : current
        )
      } catch (err) {
        toast.error(
          "Couldn't delete the run",
          err instanceof Error ? err.message : "Something went wrong."
        )
      }
    },
    [refresh]
  )

  return {
    runs,
    suite,
    selectedSuite: selected,
    selectSuite,
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
