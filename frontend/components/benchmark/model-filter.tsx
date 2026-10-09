"use client"

import * as React from "react"

import { Checkbox } from "@/components/ui/checkbox"
import type { BenchmarkRun } from "@/lib/benchmark"
import { cn } from "@/lib/utils"

/**
 * Which models the two figures above the history are plotting.
 *
 * **A scatter's readability is bounded by how many names fit beside its dots**, and
 * the history only grows: a dozen fine-tunes of one base model produce a cloud where
 * every label has been nudged three rows off its own mark, and the comparison you
 * actually came to make — this checkpoint against the one before it — is buried in
 * it. Hiding models is the cheapest fix, and it is non-destructive: nothing is
 * deleted, the history table below still lists every run.
 *
 * The state is **hidden**, not visible. A model absent from the set is shown, so a
 * benchmark that finishes while you are looking at the page appears rather than
 * silently missing the filter, and switching suite carries no stale selection.
 *
 * One row, above the charts it governs, per the filter convention — not inside
 * either panel, since it controls both.
 */
export interface ModelFilterProps {
  runs: BenchmarkRun[]
  hidden: Set<string>
  onToggle: (model: string) => void
  onShowAll: () => void
  onOnly: (model: string) => void
}

export function ModelFilter({
  runs,
  hidden,
  onToggle,
  onShowAll,
  onOnly,
}: ModelFilterProps) {
  // Newest first: the model you just trained is the reason you opened the page, and
  // it should be the first checkbox rather than wherever the alphabet puts it.
  const models = React.useMemo(() => {
    const seen = new Map<string, { runs: number; last: number }>()
    for (const run of runs) {
      const at = new Date(run.created_at).getTime()
      const entry = seen.get(run.model_name)
      if (entry) {
        entry.runs += 1
        entry.last = Math.max(entry.last, at)
      } else {
        seen.set(run.model_name, { runs: 1, last: at })
      }
    }
    return [...seen.entries()]
      .map(([name, stats]) => ({ name, ...stats }))
      .sort((a, b) => b.last - a.last)
  }, [runs])

  // Nothing to choose between. One model is not a filter, it is a checkbox that can
  // only empty the plot.
  if (models.length < 2) return null

  const shown = models.length - models.filter((m) => hidden.has(m.name)).length

  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-3 border-y border-border/60 py-3">
      <span className="eyebrow text-muted-foreground">Models</span>

      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-5 gap-y-2.5">
        {models.map((model) => {
          const on = !hidden.has(model.name)
          return (
            <label
              key={model.name}
              className="group flex max-w-full min-w-0 cursor-pointer items-center gap-2 text-sm"
            >
              <Checkbox
                checked={on}
                onCheckedChange={() => onToggle(model.name)}
              />
              <span
                className={cn(
                  "truncate transition-colors",
                  !on && "text-muted-foreground"
                )}
              >
                {model.name}
              </span>
              <span className="text-xs text-muted-foreground tabular-nums">
                {model.runs}
              </span>
              {/* The shortcut the checkboxes are clumsy at: comparing one model
                  against nothing takes as many clicks as there are others. Kept
                  quiet until hover so the row reads as a list of models rather
                  than a list of buttons. */}
              <button
                type="button"
                onClick={(event) => {
                  event.preventDefault()
                  onOnly(model.name)
                }}
                className="text-xs text-muted-foreground underline underline-offset-4 opacity-100 transition-opacity group-hover:opacity-100 hover:text-foreground focus-visible:opacity-100 sm:opacity-0"
              >
                only
              </button>
            </label>
          )
        })}
      </div>

      <button
        type="button"
        onClick={onShowAll}
        disabled={shown === models.length}
        className="text-xs text-muted-foreground underline underline-offset-4 transition-colors hover:text-foreground disabled:pointer-events-none disabled:opacity-0"
      >
        Show all {models.length}
      </button>
    </div>
  )
}
