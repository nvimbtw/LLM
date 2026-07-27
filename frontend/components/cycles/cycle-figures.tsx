"use client"

import * as React from "react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"

import {
  axisProps,
  gridProps,
  verdictColors,
  yAxisProps,
} from "@/components/training/chart-theme"
import { Figure, FigurePlaceholder } from "@/components/training/stat-run"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import { roundYield, type CycleRound } from "@/lib/cycles"

/**
 * What a cycle has actually done, per round.
 *
 * **Two charts, never one with two axes.** Yield is a fraction and the benchmark
 * score is a composite on its own scale; putting them on a shared plot with two
 * y-axes would let the crossing point imply a relationship that is an artifact of
 * where the axes were pinned. They share an x-axis instead, stacked, so a round is
 * read straight down.
 *
 * **Yield is the one that matters.** It is the fraction of what the model generated
 * that cleared the frozen bar — and it only means anything *because* the bar is
 * frozen. Against a moving quantile it would read 30% every round forever.
 */
export interface CycleFiguresProps {
  rounds: CycleRound[]
}

const yieldConfig = {
  yield: { label: "Cleared the bar", color: verdictColors.approved },
} satisfies ChartConfig

const scoreConfig = {
  score: { label: "Benchmark", color: "var(--chart-3)" },
} satisfies ChartConfig

export function CycleFigures({ rounds }: CycleFiguresProps) {
  const yieldPoints = React.useMemo(
    () =>
      rounds
        .map((r) => ({ round: r.round, yield: roundYield(r) }))
        .filter((p): p is { round: number; yield: number } => p.yield !== null),
    [rounds]
  )

  const scorePoints = React.useMemo(
    () =>
      rounds
        .filter((r) => r.score !== null)
        .map((r) => ({ round: r.round, score: r.score as number })),
    [rounds]
  )

  return (
    <div className="space-y-6">
      <Figure
        caption="Cleared the bar, per round"
        aside={
          yieldPoints.length > 0 ? (
            <span className="tabular-nums">
              {(yieldPoints[yieldPoints.length - 1].yield * 100).toFixed(0)}% latest
            </span>
          ) : undefined
        }
        minHeight="13rem"
      >
        {yieldPoints.length > 0 ? (
          <ChartContainer config={yieldConfig} className="aspect-auto h-52 w-full">
            <LineChart data={yieldPoints} margin={{ left: 4, right: 12, top: 8 }}>
              <CartesianGrid {...gridProps} />
              <XAxis
                dataKey="round"
                type="number"
                domain={["dataMin", "dataMax"]}
                allowDecimals={false}
                {...axisProps}
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                {...yAxisProps}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(v) => `Round ${v}`}
                    formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
                  />
                }
              />
              {/* One series, so no legend box — the caption names it. Dots are
                  visible because a cycle has a handful of rounds, and at n=4 the
                  points are the data. */}
              <Line
                dataKey="yield"
                name="Cleared the bar"
                type="monotone"
                stroke={verdictColors.approved}
                strokeWidth={2}
                dot={{ r: 4 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ChartContainer>
        ) : (
          <FigurePlaceholder className="h-52">
            No round has finished building its catalogue yet.
          </FigurePlaceholder>
        )}
      </Figure>

      <Figure
        caption="Benchmark score, per round"
        aside={
          scorePoints.length > 0 ? (
            <span className="tabular-nums">
              {scorePoints[scorePoints.length - 1].score.toFixed(4)} latest
            </span>
          ) : undefined
        }
        minHeight="13rem"
      >
        {scorePoints.length > 0 ? (
          <ChartContainer config={scoreConfig} className="aspect-auto h-52 w-full">
            <LineChart data={scorePoints} margin={{ left: 4, right: 12, top: 8 }}>
              <CartesianGrid {...gridProps} />
              <XAxis
                dataKey="round"
                type="number"
                domain={["dataMin", "dataMax"]}
                allowDecimals={false}
                {...axisProps}
              />
              <YAxis
                domain={["auto", "auto"]}
                tickFormatter={(v: number) => v.toFixed(2)}
                {...yAxisProps}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(v) => `Round ${v}`}
                    formatter={(value) => Number(value).toFixed(4)}
                  />
                }
              />
              <Line
                dataKey="score"
                name="Benchmark"
                type="monotone"
                stroke="var(--chart-3)"
                strokeWidth={2}
                dot={{ r: 4 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ChartContainer>
        ) : (
          <FigurePlaceholder className="h-52">
            No round has been benchmarked yet.
          </FigurePlaceholder>
        )}
      </Figure>
    </div>
  )
}
