"use client"

import { Trash2 } from "lucide-react"
import { CartesianGrid, Scatter, ScatterChart, XAxis, YAxis, ZAxis } from "recharts"

import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart"
import { Panel } from "@/components/ui/panel"
import { axisProps, gridProps, yAxisProps } from "@/components/training/chart-theme"
import { type BenchmarkRun, NO_VALUE, formatDate, formatNumber } from "@/lib/benchmark"
import { cn } from "@/lib/utils"

const chartConfig = {
  score: { label: "Eval score", color: "var(--primary)" },
} satisfies ChartConfig

export interface HistoryPanelProps {
  runs: BenchmarkRun[]
  selectedId: string | null
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

interface Point {
  time: number
  score: number
  name: string
  id: string
}

export function HistoryPanel({ runs, selectedId, onSelect, onDelete }: HistoryPanelProps) {
  // Only finished runs with a score belong on the chart; a failed or in-flight run
  // has no y value and would either vanish or plot at zero.
  const points: Point[] = runs
    .filter((run) => run.status === "done" && run.score !== null)
    .map((run) => ({
      time: new Date(run.created_at).getTime(),
      score: run.score as number,
      name: run.model_name,
      id: run.id_run,
    }))
    .sort((a, b) => a.time - b.time)

  return (
    <Panel title="History" description={`${runs.length} ${runs.length === 1 ? "run" : "runs"}`}>
      {points.length > 1 && (
        // Scatter, not a line: consecutive runs are usually *different models*, so
        // joining them would draw a trajectory through things that aren't a series.
        // Position over time still shows whether the project is getting better.
        <ChartContainer config={chartConfig} className="mb-6 aspect-auto h-48 w-full">
          <ScatterChart margin={{ left: 4, right: 12, top: 8, bottom: 4 }}>
            <CartesianGrid {...gridProps} stroke="var(--border)" strokeOpacity={0.6} />
            <XAxis
              {...axisProps}
              dataKey="time"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(value: number) =>
                new Date(value).toLocaleDateString("en-US", { month: "short", day: "numeric" })
              }
              className="text-[10px]"
            />
            <YAxis
              {...yAxisProps}
              dataKey="score"
              type="number"
              domain={[0, 100]}
              className="text-[10px]"
            />
            <ZAxis range={[48, 48]} />
            <ChartTooltip cursor={{ stroke: "var(--border)" }} content={<PointTooltip />} />
            <Scatter data={points} fill="var(--primary)" fillOpacity={0.75} />
          </ScatterChart>
        </ChartContainer>
      )}

      {runs.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted-foreground">
          No benchmarks yet. Pick a model above and run one.
        </p>
      ) : (
        <ul className="flex flex-col gap-px overflow-hidden rounded-lg border border-border/60 bg-border/60">
          {runs.map((run) => (
            <HistoryRow
              key={run.id_run}
              run={run}
              selected={run.id_run === selectedId}
              onSelect={onSelect}
              onDelete={onDelete}
            />
          ))}
        </ul>
      )}
    </Panel>
  )
}

function HistoryRow({
  run,
  selected,
  onSelect,
  onDelete,
}: {
  run: BenchmarkRun
  selected: boolean
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}) {
  const score = run.score
  return (
    <li className={cn("group flex items-center gap-3 bg-background px-4", selected && "bg-secondary/60")}>
      <button
        type="button"
        onClick={() => onSelect(run.id_run)}
        className="flex min-w-0 flex-1 items-center gap-4 py-3 text-left"
      >
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium">{run.model_name}</span>
          <span className="block text-[11px] text-muted-foreground/60 tabular-nums">
            {formatDate(run.created_at)}
            {run.status === "failed" && run.error && ` · ${run.error}`}
          </span>
        </span>

        {/* Length carries the score; the number is right there for the exact value. */}
        <span className="hidden w-28 shrink-0 sm:block">
          {score !== null && (
            <span className="block h-1 w-full overflow-hidden rounded-full bg-secondary">
              <span
                className="block h-full rounded-full bg-primary"
                style={{ width: `${Math.max(Math.min(score, 100), 2)}%` }}
              />
            </span>
          )}
        </span>

        <span className="w-20 shrink-0 text-right">
          {run.status === "running" ? (
            <span className="text-xs text-muted-foreground">
              {run.progress ? `${Math.round(run.progress.fraction * 100)}%` : "…"}
            </span>
          ) : (
            <span
              className={cn(
                "font-display text-lg font-semibold tabular-nums",
                run.status === "failed" && "text-destructive/70",
                score === null && "text-muted-foreground/30"
              )}
            >
              {run.status === "failed" ? "failed" : formatNumber(score, 1)}
            </span>
          )}
        </span>
      </button>

      <button
        type="button"
        onClick={() => onDelete(run.id_run)}
        disabled={run.status === "running"}
        aria-label={`Delete benchmark of ${run.model_name}`}
        className="shrink-0 rounded p-1.5 text-muted-foreground/0 transition-colors group-hover:text-muted-foreground/50 hover:!text-destructive disabled:pointer-events-none"
      >
        <Trash2 className="size-3.5" />
      </button>
    </li>
  )
}

function PointTooltip({ active, payload }: { active?: boolean; payload?: { payload: Point }[] }) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  return (
    <div className="min-w-36 rounded-lg border border-border/60 bg-popover px-3 py-2 text-xs text-popover-foreground shadow-md">
      <div className="mb-1 text-[10px] font-medium uppercase tracking-[0.12em] text-muted-foreground/70">
        {formatDate(new Date(point.time).toISOString())}
      </div>
      <div className="flex items-baseline justify-between gap-6">
        <span className="truncate">{point.name}</span>
        <span className="font-medium tabular-nums">{formatNumber(point.score, 1) || NO_VALUE}</span>
      </div>
    </div>
  )
}
