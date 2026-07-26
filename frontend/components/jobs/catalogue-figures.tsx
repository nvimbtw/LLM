"use client"

import * as React from "react"
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts"

import { Figure, FigurePlaceholder } from "@/components/training/stat-run"
import {
  axisProps,
  gridProps,
  verdictColors,
  yAxisProps,
} from "@/components/training/chart-theme"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"

/**
 * The two figures a catalogue run produces while it runs.
 *
 * Both answer questions the progress bar cannot. The first is *is this balanced,
 * and filling* — a one-sided catalogue is the failure that otherwise surfaces only
 * when the KTO run refuses it, minutes of GPU time later. The second is *why are
 * things landing where they are*: the scores against the calibrated boundary, which
 * is the difference between "the judge is separating good from bad" and "everything
 * piled up on one side of a threshold fit to too few ratings".
 *
 * Approved/rejected is a polarity pair, and the colours are blue/amber rather than
 * the obvious green/red because that pair is the classic colourblind collision —
 * see `chart-theme.ts`, where both modes' steps are validated.
 */

const chartConfig = {
  approved: { label: "Good", color: verdictColors.approved },
  rejected: { label: "Bad", color: verdictColors.rejected },
} satisfies ChartConfig

export interface CatalogueFiguresProps {
  scores: { score: number; good: boolean }[]
  threshold?: number
}

export function CatalogueFigures({ scores, threshold }: CatalogueFiguresProps) {
  // Cumulative totals, one point per verdict. Cumulative rather than a rate,
  // because the question is "how much of each do I have" — the same question the
  // gate asks — and a per-bucket rate would answer a question nobody has.
  const cumulative = React.useMemo(() => {
    // A plain loop rather than a closure over running totals: the compiler cannot
    // prove a callback that mutates captured counters stays inside the render, and
    // it is right to object — this shape has nothing to prove.
    const points: { n: number; approved: number; rejected: number }[] = []
    let approved = 0
    let rejected = 0
    for (let i = 0; i < scores.length; i++) {
      if (scores[i].good) approved += 1
      else rejected += 1
      points.push({ n: i + 1, approved, rejected })
    }
    return points
  }, [scores])

  const histogram = React.useMemo(() => bin(scores), [scores])
  const latest = cumulative.at(-1)

  return (
    <div className="space-y-6">
      <Figure
        caption="Catalogue building"
        aside={
          latest ? (
            <Legend approved={latest.approved} rejected={latest.rejected} />
          ) : undefined
        }
      >
        {cumulative.length > 1 ? (
          <ChartContainer config={chartConfig} className="aspect-auto h-56 w-full">
            <AreaChart data={cumulative} margin={{ left: 4, right: 4, top: 4 }}>
              <CartesianGrid {...gridProps} />
              <XAxis
                dataKey="n"
                type="number"
                domain={["dataMin", "dataMax"]}
                {...axisProps}
              />
              <YAxis allowDecimals={false} {...yAxisProps} />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(v) => `After ${v} labelled`}
                  />
                }
              />
              {/* Not stacked: the two are read against each other, and stacking
                  would make the upper series' height mean the total instead. */}
              <Area
                dataKey="approved"
                name="Good"
                type="monotone"
                stroke={verdictColors.approved}
                fill={verdictColors.approved}
                fillOpacity={0.12}
                strokeWidth={2}
                isAnimationActive={false}
              />
              <Area
                dataKey="rejected"
                name="Bad"
                type="monotone"
                stroke={verdictColors.rejected}
                fill={verdictColors.rejected}
                fillOpacity={0.12}
                strokeWidth={2}
                isAnimationActive={false}
              />
            </AreaChart>
          </ChartContainer>
        ) : (
          <FigurePlaceholder className="h-56">
            Waiting for the first verdicts…
          </FigurePlaceholder>
        )}
      </Figure>

      <Figure
        caption="Scores against the boundary"
        aside={
          threshold !== undefined ? (
            <span className="tabular-nums">
              threshold {threshold.toFixed(3)}
            </span>
          ) : undefined
        }
      >
        {histogram.length > 0 ? (
          <ChartContainer config={chartConfig} className="aspect-auto h-56 w-full">
            {/* Extra top margin so the boundary label has somewhere to sit. */}
            <BarChart data={histogram} margin={{ left: 4, right: 4, top: 16 }}>
              <CartesianGrid {...gridProps} />
              <XAxis
                dataKey="mid"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(v: number) => v.toFixed(2)}
                {...axisProps}
              />
              <YAxis allowDecimals={false} {...yAxisProps} />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(v) => `Score ≈ ${Number(v).toFixed(3)}`}
                  />
                }
              />
              {threshold !== undefined && (
                <ReferenceLine
                  x={threshold}
                  stroke="var(--muted-foreground)"
                  strokeDasharray="3 3"
                  label={{
                    value: "boundary",
                    position: "top",
                    fontSize: 10,
                    fill: "var(--muted-foreground)",
                  }}
                />
              )}
              {/* One bar per bin, coloured by the side it fell on — the split is
                  the threshold, so a stacked pair would be two colours where the
                  data only ever has one. */}
              <Bar dataKey="count" name="Replies" isAnimationActive={false}>
                {histogram.map((b) => (
                  <Cell
                    key={b.mid}
                    fill={
                      b.good ? verdictColors.approved : verdictColors.rejected
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ChartContainer>
        ) : (
          <FigurePlaceholder className="h-56">
            Waiting for the first verdicts…
          </FigurePlaceholder>
        )}
      </Figure>
    </div>
  )
}

/**
 * Direct labels for the two series, so identity is never carried by colour alone.
 * A swatch beside text in the ordinary ink, rather than coloured text — the mark
 * carries the identity, the label stays legible.
 */
function Legend({
  approved,
  rejected,
}: {
  approved: number
  rejected: number
}) {
  return (
    <span className="flex items-center gap-3 tabular-nums">
      <Swatch color={verdictColors.approved} /> {approved} good
      <Swatch color={verdictColors.rejected} /> {rejected} bad
    </span>
  )
}

function Swatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden
      className="inline-block size-2 rounded-[2px] align-middle"
      style={{ background: color }}
    />
  )
}

const BINS = 24

/** Buckets the scores for the histogram, tagging each bin with the side it fell on. */
function bin(scores: { score: number; good: boolean }[]) {
  if (scores.length === 0) return []

  const values = scores.map((s) => s.score)
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  // Every candidate scoring identically is possible early in a run; one bin is the
  // honest rendering of that, and a zero width would divide by zero below.
  const width = hi > lo ? (hi - lo) / BINS : 1

  const buckets = Array.from({ length: BINS }, (_, i) => ({
    mid: lo + width * (i + 0.5),
    count: 0,
    goods: 0,
  }))

  for (const s of scores) {
    const idx = Math.min(BINS - 1, Math.floor((s.score - lo) / width))
    buckets[idx].count += 1
    if (s.good) buckets[idx].goods += 1
  }

  // Coloured by what the bin's replies were actually labelled, not by where its
  // midpoint sits relative to the threshold — so a bin straddling the boundary
  // takes its majority rather than implying a split that the bar cannot show.
  return buckets
    .filter((b) => b.count > 0)
    .map((b) => ({ mid: b.mid, count: b.count, good: b.goods * 2 >= b.count }))
}
