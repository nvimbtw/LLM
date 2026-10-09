"use client"

import { Trash2 } from "lucide-react"

import { Panel } from "@/components/ui/panel"
import {
  type BenchmarkRun,
  NO_VALUE,
  formatCompact,
  formatDate,
  formatNumber,
} from "@/lib/benchmark"
import { TIER_LABEL } from "@/lib/jobs"
import { cn } from "@/lib/utils"

// Every run as a league table, best score first.
//
// Ranked by score rather than by date because the question asked of this table is
// "which model is the best one so far", and a ranking answers it by position — the
// top row *is* the answer, with no scanning. Recency is still on every row for the
// times the question is "what did I just run"; the panels above cover what a score
// cost — in tokens and in time — which an ordering by date used only to hint at.
export interface HistoryPanelProps {
  runs: BenchmarkRun[]
  selectedId: string | null
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

export function HistoryPanel({
  runs,
  selectedId,
  onSelect,
  onDelete,
}: HistoryPanelProps) {
  // Scored runs rank by score, descending. A run with no score can't take part in the
  // ranking, so the unscored ones sit below the table's body in date order — still
  // reachable (a failed run carries its error, a running one its progress) without
  // interleaving blanks into the ranking.
  const ranked = [...runs].sort((a, b) => {
    if (a.score === null || b.score === null) {
      if (a.score !== null) return -1
      if (b.score !== null) return 1
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    }
    return b.score - a.score
  })

  return (
    <Panel
      variant="open"
      title="Models by score"
      description={`${runs.length} ${runs.length === 1 ? "run" : "runs"}`}
      bodyClassName="p-0"
    >
      {runs.length === 0 ? (
        <p className="border-y border-border/60 py-6 text-sm text-muted-foreground">
          No benchmarks yet.
        </p>
      ) : (
        // A real table: four aligned columns of the same kind of thing per row, which
        // is what makes two models comparable at a glance down a column.
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border eyebrow text-muted-foreground">
                <th
                  scope="col"
                  className="w-10 py-2.5 pr-2 text-right font-medium"
                >
                  #
                </th>
                <th scope="col" className="py-2.5 pr-3 text-left font-medium">
                  Model
                </th>
                <th
                  scope="col"
                  className="hidden py-2.5 pr-3 text-right font-medium sm:table-cell"
                >
                  Tokens
                </th>
                <th
                  scope="col"
                  className="hidden py-2.5 pr-3 text-right font-medium sm:table-cell"
                >
                  Run
                </th>
                <th scope="col" className="py-2.5 pr-3 text-right font-medium">
                  Score
                </th>
                {/* The delete column's header is its own row's action, named for
                    screen readers rather than shown. */}
                <th scope="col" className="w-10 py-2.5 pr-4">
                  <span className="sr-only">Delete</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((run, index) => (
                <HistoryRow
                  key={run.id_run}
                  run={run}
                  // Only scored runs are ranked, so the unscored ones below them get
                  // no number rather than a misleading one.
                  rank={run.score !== null ? index + 1 : null}
                  selected={run.id_run === selectedId}
                  onSelect={onSelect}
                  onDelete={onDelete}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}

function HistoryRow({
  run,
  rank,
  selected,
  onSelect,
  onDelete,
}: {
  run: BenchmarkRun
  rank: number | null
  selected: boolean
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}) {
  const score = run.score

  return (
    <tr
      onClick={() => onSelect(run.id_run)}
      className={cn(
        "group cursor-pointer border-b border-border/40 transition-colors last:border-0 hover:bg-secondary/40",
        selected && "bg-secondary/60"
      )}
    >
      <td className="py-3 pr-2 text-right text-xs text-muted-foreground tabular-nums">
        {rank ?? NO_VALUE}
      </td>

      <td className="min-w-0 py-3 pr-3">
        {/* The button carries the accessible name and the keyboard path; the row's
            own onClick is the mouse convenience on top of it. */}
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation()
            onSelect(run.id_run)
          }}
          className="block max-w-full truncate text-left font-medium hover:underline"
        >
          {run.model_name}
          {/* The budget belongs on the name, not in a column of its own: without it
              a sweep is three rows of one model with three different scores and
              nothing saying why they differ. */}
          {run.think_budget && (
            <span className="ml-1.5 text-xs font-normal text-muted-foreground">
              {TIER_LABEL[run.think_budget].toLowerCase()}
            </span>
          )}
        </button>
        {run.status === "failed" && run.error && (
          <span className="mt-0.5 block truncate text-xs text-destructive/70">
            {run.error}
          </span>
        )}
      </td>

      <td className="hidden py-3 pr-3 text-right text-xs text-muted-foreground tabular-nums sm:table-cell">
        {/* What the run cost, not what the model was trained on — the same
            measure the tokens chart plots, so the table and the figure agree. */}
        {run.eval_tokens === null ? NO_VALUE : formatCompact(run.eval_tokens)}
      </td>

      <td className="hidden py-3 pr-3 text-right text-xs text-muted-foreground tabular-nums sm:table-cell">
        {formatDate(run.created_at)}
      </td>

      <td className="py-3 pr-3 text-right">
        {run.status === "running" ? (
          // Live progress belongs to the job producing this run, and /admin is
          // where it is shown; repeating it here would mean polling for a number
          // that is already being pushed somewhere better.
          <span className="text-xs text-muted-foreground">…</span>
        ) : (
          <span
            className={cn(
              "figure text-lg",
              run.status === "failed" && "text-destructive/70",
              score === null && "text-muted-foreground/30"
            )}
          >
            {run.status === "failed" ? "failed" : formatNumber(score, 1)}
          </span>
        )}
      </td>

      <td className="py-3 pr-4 text-right">
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation()
            onDelete(run.id_run)
          }}
          disabled={run.status === "running"}
          aria-label={`Delete benchmark of ${run.model_name}`}
          className="rounded-(--radius-marker) p-1.5 text-muted-foreground transition-colors group-hover:text-muted-foreground hover:!text-destructive focus-visible:text-muted-foreground disabled:pointer-events-none sm:text-muted-foreground/0"
        >
          <Trash2 className="size-3.5" />
        </button>
      </td>
    </tr>
  )
}
