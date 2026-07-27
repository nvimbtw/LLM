"use client"

import Link from "next/link"

import { Throbber } from "@/components/ui/throbber"
import { useCycles } from "@/hooks/use-cycles"
import { CYCLE_STATUS_LABEL, STAGE_LABEL, isCycleLive } from "@/lib/cycles"

/**
 * Every cycle, newest first. A ledger, not a dashboard: one line each, saying
 * where it got to and why it stopped.
 */
export function CyclesView() {
  const { cycles, isLoading, error } = useCycles()

  if (isLoading) {
    return <p className="py-16 text-center text-sm text-muted-foreground">Loading…</p>
  }
  if (error) {
    return <p className="py-16 text-center text-sm text-destructive">{error}</p>
  }
  if (cycles.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border/60 px-6 py-12 text-center">
        <p className="text-sm text-muted-foreground">
          No cycles yet. A cycle repeats one round — build catalogue, self-train,
          preference-tune, benchmark — until a round stops improving, and it keeps
          the best model rather than the last one.
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
    <ul className="divide-y divide-border/30 rounded-xl border border-border/60">
      {cycles.map((cycle) => (
        <li key={cycle.id_cycle}>
          <Link
            href={`/cycles/${cycle.id_cycle}`}
            className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 px-5 py-4 hover:bg-muted/40"
          >
            <span className="flex min-w-0 items-center gap-2">
              {cycle.status === "running" && <Throbber mode="training" />}
              <span className="truncate font-medium">{cycle.label}</span>
            </span>
            <span className="flex flex-wrap items-center gap-x-3 text-xs text-muted-foreground tabular-nums">
              <span>
                round {cycle.round}/{cycle.max_rounds}
              </span>
              {isCycleLive(cycle) && <span>{STAGE_LABEL[cycle.stage]}</span>}
              {cycle.best_score !== null && <span>best {cycle.best_score.toFixed(4)}</span>}
              <span>{CYCLE_STATUS_LABEL[cycle.status]}</span>
            </span>
          </Link>
          {/* Why it stopped belongs on the list too — otherwise every finished
              cycle needs a click to answer the only question about it. */}
          {cycle.halt_reason && !isCycleLive(cycle) && (
            <p className="px-5 pb-3 text-xs text-muted-foreground">{cycle.halt_reason}</p>
          )}
        </li>
      ))}
    </ul>
  )
}
