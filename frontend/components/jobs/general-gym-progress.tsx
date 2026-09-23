"use client"

import * as React from "react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"

import { axisProps, gridProps, yAxisProps } from "@/components/training/chart-theme"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import type { GymSourceScore } from "@/lib/gyms"

export interface SourceHistoryPoint {
  step: number
  source_scores: Record<string, GymSourceScore>
  task_sequence?: string[]
  pass_rate?: number
  signal?: number
}

export interface GeneralGymProgressProps {
  history: SourceHistoryPoint[]
  sources: string[]
}

const RECENT_TASKS = 14
const preferredOrder = ["bash", "math", "state_tracking", "constraint_logic"]
const chartConfig = {
  correct: { label: "Correct", color: "var(--chart-1)" },
} satisfies ChartConfig

function sourceLabel(source: string): string {
  return source === "bash" ? "just-bash" : source.replaceAll("_", " ")
}

export function GeneralGymProgress({ history, sources }: GeneralGymProgressProps) {
  const orderedSources = React.useMemo(
    () => [...new Set([...preferredOrder, ...sources])],
    [sources]
  )
  const progress = React.useMemo(() => {
    const totals = new Map<string, { correct: number; graded: number; episodes: number }>()
    const series = new Map<string, { step: number; correct: number }[]>()
    for (const entry of history) {
      for (const [source, score] of Object.entries(entry.source_scores ?? {})) {
        const previous = totals.get(source) ?? { correct: 0, graded: 0, episodes: 0 }
        const next = {
          correct: previous.correct + score.correct,
          graded: previous.graded + score.graded,
          episodes: previous.episodes + score.episode_count,
        }
        totals.set(source, next)
        if (!series.has(source)) series.set(source, [])
        series.get(source)!.push({ step: entry.step, correct: next.correct })
      }
    }
    return { totals, series }
  }, [history])
  const recent = history.flatMap((entry) =>
    (entry.task_sequence?.length ? entry.task_sequence : Object.keys(entry.source_scores ?? {}))
      .map((source) => ({ source, step: entry.step }))
  ).slice(-RECENT_TASKS)
  const hasGroupOrder = history.some((entry) => (entry.task_sequence?.length ?? 0) > 0)
  const latestStep = recent.at(-1)?.step
  const recentRef = React.useRef<HTMLDivElement>(null)
  React.useEffect(() => {
    if (recentRef.current) recentRef.current.scrollLeft = recentRef.current.scrollWidth
  }, [recent.length, latestStep])

  return (
    <div className="space-y-5 border-b border-border pb-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">General gym · source progress</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Cumulative correct answers by gym; each chart advances when that gym is sampled.
          </p>
        </div>
        <span className="eyebrow text-muted-foreground">{history.length} batches</span>
      </div>

      <div className="rounded-(--radius-control) border border-border/70 bg-background/50 px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <span className="eyebrow text-muted-foreground">
            {hasGroupOrder ? "Recent task groups" : "Recent gym mix by batch"}
          </span>
          <span className="text-xs text-muted-foreground">Latest →</span>
        </div>
        {recent.length ? (
          <div ref={recentRef} className="mt-3 flex gap-1.5 overflow-x-auto pb-1" aria-label="Recent task sources, oldest to newest">
            {recent.map(({ source, step }, index) => (
              <span
                key={`${step}-${index}`}
                title={`Iteration ${step} · ${sourceLabel(source)}`}
                className={`shrink-0 rounded-(--radius-control-sm) border px-2.5 py-1 text-xs ${
                  index === recent.length - 1
                    ? "border-primary bg-primary/10 font-semibold text-foreground"
                    : "border-border text-muted-foreground"
                }`}
              >
                {sourceLabel(source)}
              </span>
            ))}
          </div>
        ) : (
          <p className="mt-3 text-sm text-muted-foreground">Waiting for the first batch…</p>
        )}
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        {orderedSources.map((source) => {
          const values = progress.series.get(source) ?? []
          const total = progress.totals.get(source)
          return (
            <div key={source} className="min-w-0 rounded-(--radius-control) border border-border/70 bg-background/30 p-4">
              <div className="flex items-baseline justify-between gap-3">
                <h4 className="text-sm font-semibold">{sourceLabel(source)}</h4>
                <span className="figure text-xl tabular-nums">{total?.correct ?? 0}</span>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {total
                  ? `${total.correct} correct / ${total.graded} graded${total.graded ? ` · ${Math.round(total.correct / total.graded * 100)}%` : " · none graded"}`
                  : "Waiting for tasks"}
              </p>
              {values.length ? (
                <ChartContainer config={chartConfig} className="mt-3 aspect-auto h-36 w-full">
                  <LineChart data={values} margin={{ left: 0, right: 10, top: 8, bottom: 0 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} {...axisProps} />
                    <YAxis allowDecimals={false} domain={[0, "dataMax"]} {...yAxisProps} />
                    <ChartTooltip cursor={false} content={<ChartTooltipContent labelFormatter={(v) => `Iteration ${v}`} />} />
                    <Line dataKey="correct" type="linear" stroke="var(--chart-1)" strokeWidth={1.5}
                      dot={values.length === 1} isAnimationActive={false} />
                  </LineChart>
                </ChartContainer>
              ) : (
                <div className="mt-3 flex h-36 items-center justify-center text-xs text-muted-foreground">No batches yet</div>
              )}
              <p className="mt-1 text-right text-xs text-muted-foreground">
                {total?.episodes ?? 0} attempts · {values.length} batches
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
