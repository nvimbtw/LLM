"use client"

import * as React from "react"
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  XAxis,
  YAxis,
} from "recharts"

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
import type { SynthCall } from "@/lib/jobs"

/**
 * The two figures a generation run produces while it runs.
 *
 * The first is *is this working* — conversations kept against conversations thrown
 * away, cumulative, one point per model call. A run that is producing nothing looks
 * exactly like a run that is producing plenty until you separate those two lines.
 *
 * The second is *why* — the rejection tally, ranked. This is the figure that tells a
 * broken harness from a weak model: "over the token window" every time means the
 * length rule needs tightening, "bad markup" means the prompt is not landing, and a
 * provider error means nothing is reaching the model at all.
 *
 * Kept/rejected is a polarity pair and reuses the catalogue's CVD-validated
 * blue/amber (see `chart-theme.ts`). The tally is a single ranked series, so it
 * takes one colour and no legend — the axis names each bar.
 */

const yieldConfig = {
  kept: { label: "Kept", color: verdictColors.approved },
  lost: { label: "Discarded", color: verdictColors.rejected },
} satisfies ChartConfig

const tallyConfig = {
  count: { label: "Discarded", color: verdictColors.rejected },
} satisfies ChartConfig

export interface SynthFiguresProps {
  calls: SynthCall[]
}

export function SynthFigures({ calls }: SynthFiguresProps) {
  // Cumulative rather than per-call: the question is "how much do I have", which
  // is the same question the run's target answers, and a per-call rate would answer
  // one nobody is asking.
  const cumulative = React.useMemo(() => {
    const points: { call: number; kept: number; lost: number }[] = []
    let kept = 0
    let lost = 0
    for (const call of calls) {
      kept += call.kept
      lost += call.duplicates + call.rejected.reduce((n, [, c]) => n + c, 0)
      points.push({ call: call.call, kept, lost })
    }
    return points
  }, [calls])

  const tally = React.useMemo(() => {
    const totals = new Map<string, number>()
    for (const call of calls) {
      for (const [reason, count] of call.rejected) {
        totals.set(reason, (totals.get(reason) ?? 0) + count)
      }
      if (call.duplicates > 0) {
        totals.set(
          "duplicate opener",
          (totals.get("duplicate opener") ?? 0) + call.duplicates
        )
      }
    }
    return [...totals.entries()]
      .map(([reason, count]) => ({ reason, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8)
  }, [calls])

  const latest = cumulative.at(-1)

  return (
    <div className="space-y-6">
      <Figure
        caption="Yield"
        aside={
          latest ? (
            <span className="flex items-center gap-3 tabular-nums">
              <Swatch color={verdictColors.approved} /> {latest.kept} kept
              <Swatch color={verdictColors.rejected} /> {latest.lost} discarded
            </span>
          ) : undefined
        }
      >
        {cumulative.length > 1 ? (
          <ChartContainer
            config={yieldConfig}
            className="aspect-auto h-56 w-full"
          >
            <AreaChart data={cumulative} margin={{ left: 4, right: 4, top: 4 }}>
              <CartesianGrid {...gridProps} />
              <XAxis
                dataKey="call"
                type="number"
                domain={["dataMin", "dataMax"]}
                allowDecimals={false}
                {...axisProps}
              />
              <YAxis allowDecimals={false} {...yAxisProps} />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(v) => `After call ${v}`}
                  />
                }
              />
              {/* Not stacked: the two are read against each other, and stacking
                  would make the upper series' height mean the total instead. */}
              <Area
                dataKey="kept"
                name="Kept"
                type="monotone"
                stroke={verdictColors.approved}
                fill={verdictColors.approved}
                fillOpacity={0.12}
                strokeWidth={2}
                isAnimationActive={false}
              />
              <Area
                dataKey="lost"
                name="Discarded"
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
            Waiting for the first call to come back…
          </FigurePlaceholder>
        )}
      </Figure>

      <Figure caption="Where the discarded output went">
        {tally.length > 0 ? (
          <ChartContainer
            config={tallyConfig}
            className="aspect-auto h-56 w-full"
          >
            {/* Horizontal: the categories are sentences, and sentences do not fit
                under a vertical bar without being turned or truncated. */}
            <BarChart
              data={tally}
              layout="vertical"
              margin={{ left: 4, right: 16, top: 4 }}
            >
              <CartesianGrid {...gridProps} horizontal={false} />
              <XAxis type="number" allowDecimals={false} {...axisProps} />
              {/* `width` after the spread on purpose: `yAxisProps` carries the
                  numeric-axis width, and a sentence-length category label needs
                  more room than a four-digit number. */}
              <YAxis
                type="category"
                dataKey="reason"
                tick={{ fontSize: 11 }}
                {...yAxisProps}
                width={190}
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar
                dataKey="count"
                name="Discarded"
                fill={verdictColors.rejected}
                radius={[0, 4, 4, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ChartContainer>
        ) : (
          <FigurePlaceholder className="h-56">
            Nothing discarded yet.
          </FigurePlaceholder>
        )}
      </Figure>
    </div>
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
