"use client"

import * as React from "react"

import { getModelRuns, type ModelBenchmarks } from "@/lib/benchmark"

// A settled fetch, tagged with the model it answers for. Tagged rather than reset
// on the way in, because clearing state as the effect starts is a synchronous
// setState inside an effect — a cascading render, and the thing the state-in-effect
// rule exists to stop. Comparing the tag to the current id says "still loading" for
// free, and a late response for a model the page has navigated away from is
// discarded by the same comparison.
interface Settled {
  id: string
  data: ModelBenchmarks | null
  error: string | null
}

/**
 * One model's benchmark record, for the results section of its detail page.
 *
 * `enabled` exists because the benchmark endpoints are admin-only: a normal
 * session asking would collect a 403 for every model page it opens, and the
 * section it feeds is not rendered for them anyway.
 */
export function useModelBenchmarks(id: string, enabled = true) {
  const [settled, setSettled] = React.useState<Settled | null>(null)

  React.useEffect(() => {
    if (!enabled) return
    let active = true
    getModelRuns(id)
      .then((data) => {
        if (active) setSettled({ id, data, error: null })
      })
      .catch((e: unknown) => {
        if (!active) return
        setSettled({
          id,
          data: null,
          error:
            e instanceof Error ? e.message : "Failed to load the benchmarks.",
        })
      })
    return () => {
      active = false
    }
  }, [id, enabled])

  const current = settled?.id === id ? settled : null
  return {
    data: current?.data ?? null,
    isLoading: enabled && current === null,
    error: current?.error ?? null,
  }
}
