"use client"

import Link from "next/link"

import { cn } from "@/lib/utils"
import { useModelNames } from "@/hooks/use-model-names"
import { modelLabel, type GymRun } from "@/lib/gyms"

/**
 * Every run of one gym, newest first.
 *
 * A ledger rather than a grid of cards: these rows are read as a column of
 * scores — did this model get better than that one — and a card per run would
 * put a border between every comparison.
 */
export function RunList({ runs }: { runs: GymRun[] }) {
  const { names } = useModelNames()

  if (runs.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No runs yet. Create one above and it will appear here.
      </p>
    )
  }

  return (
    <ul className="divide-y divide-border">
      {runs.map((run) => (
        <li key={run.id}>
          <Link
            href={`/gyms/${run.gym_id}/runs/${run.id}`}
            className="flex items-baseline gap-4 py-3 transition-colors hover:text-foreground"
          >
            <span className="min-w-0 flex-1 truncate font-medium">
              {modelLabel(names, run.model_name)}
            </span>
            <span className="hidden shrink-0 text-xs text-muted-foreground sm:inline">
              {run.task_types.join(" · ")}
            </span>
            <span className="shrink-0 text-xs text-muted-foreground">
              {run.graded}/{run.episode_count}
            </span>
            <Score run={run} />
          </Link>
        </li>
      ))}
    </ul>
  )
}

/**
 * The headline number, or why there isn't one.
 *
 * A run with nothing graded shows its status instead of `0%` — an unstarted run
 * and a model that got everything wrong are not the same fact, and one number
 * standing for both is the kind of thing a benchmark gets quietly wrong.
 */
function Score({ run }: { run: GymRun }) {
  if (run.score === null) {
    return (
      <span className="w-16 shrink-0 text-right text-xs text-muted-foreground">
        {run.status}
      </span>
    )
  }
  return (
    <span
      className={cn(
        "w-16 shrink-0 text-right font-mono text-sm tabular-nums",
        run.score >= 1 && "text-foreground"
      )}
    >
      {Math.round(run.score * 100)}%
    </span>
  )
}
