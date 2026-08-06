"use client"

import Link from "next/link"

import { RunningDot } from "@/components/ui/running-dot"
import { useCycles } from "@/hooks/use-cycles"
import { CYCLE_STATUS_LABEL, STAGE_LABEL, isCycleLive } from "@/lib/cycles"

/**
 * Every cycle, newest first. A ledger, not a dashboard: one line each, saying
 * where it got to and why it stopped.
 *
 * The row is a grid rather than a `justify-between` pair, because at board width
 * the latter throws the label and its numbers to opposite edges of a very wide
 * page and nothing lines up down the column. Fixed tracks mean round, stage,
 * score and status each read as a column you can scan vertically.
 */
export function CyclesView() {
  const { cycles, isLoading, error } = useCycles()

  if (isLoading) {
    return (
      <p className="py-16 text-center text-sm text-muted-foreground">
        Loading…
      </p>
    )
  }
  if (error) {
    return <p className="py-16 text-center text-sm text-destructive">{error}</p>
  }
  if (cycles.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border px-6 py-12 text-center">
        <p className="text-sm text-muted-foreground">
          No cycles yet. A cycle repeats one round — build catalogue,
          self-train, preference-tune, benchmark — until a round stops
          improving, and it keeps the best model rather than the last one.
        </p>
        <Link
          href="/cycles/new"
          className="mt-4 inline-block text-sm underline-offset-4 hover:underline"
        >
          Start one →
        </Link>
      </div>
    )
  }

  return (
    <ul className="divide-y divide-border/30 overflow-hidden rounded-lg border border-border">
      {cycles.map((cycle) => (
        <li key={cycle.id_cycle}>
          <Link
            href={`/cycles/${cycle.id_cycle}`}
            className="grid grid-cols-[1fr_auto] items-baseline gap-x-6 gap-y-1 px-5 py-4 transition-colors hover:bg-muted/40 sm:grid-cols-[minmax(0,1fr)_7rem_9rem_8rem_6rem]"
          >
            <span className="flex min-w-0 items-center gap-2">
              {cycle.status === "running" && <RunningDot />}
              <span className="truncate font-medium">{cycle.label}</span>
            </span>
            {/* Below `sm` the five tracks collapse to one inline run — the same
                values, wrapped, rather than five columns squeezed to nothing. */}
            <span className="col-start-2 flex flex-wrap items-center gap-x-3 text-xs text-muted-foreground tabular-nums sm:contents">
              <span className="sm:text-right">
                round {cycle.round}/{cycle.max_rounds}
              </span>
              <span className="sm:truncate">
                {isCycleLive(cycle) ? STAGE_LABEL[cycle.stage] : null}
              </span>
              <span className="sm:text-right">
                {cycle.best_score !== null
                  ? `best ${cycle.best_score.toFixed(4)}`
                  : null}
              </span>
              <span className="sm:text-right">
                {CYCLE_STATUS_LABEL[cycle.status]}
              </span>
            </span>
          </Link>
          {/* Why it stopped belongs on the list too — otherwise every finished
              cycle needs a click to answer the only question about it. */}
          {cycle.halt_reason && !isCycleLive(cycle) && (
            <p className="px-5 pb-3 text-xs text-muted-foreground">
              {cycle.halt_reason}
            </p>
          )}
        </li>
      ))}
    </ul>
  )
}
