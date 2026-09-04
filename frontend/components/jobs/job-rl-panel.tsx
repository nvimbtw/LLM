"use client"

import * as React from "react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"

import { Appendix } from "@/components/training/appendix"
import {
  ACCENT_COLOR,
  MUTED_SERIES,
  axisProps,
  gridProps,
  yAxisProps,
} from "@/components/training/chart-theme"
import { FigurePlaceholder } from "@/components/training/stat-run"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import { Panel } from "@/components/ui/panel"
import type { useJob } from "@/hooks/use-job"

/**
 * A reinforcement-learning run's body: a pass rate, and the number that says
 * whether the pass rate means anything.
 *
 * There is no loss curve here on purpose. The policy-gradient loss is an
 * advantage-weighted sum whose sign and magnitude say nothing on their own — it
 * is not a quantity that goes down — so plotting it would be a chart that
 * invites exactly the wrong reading.
 *
 * **Signal is the diagnostic.** A group is G rollouts of one task and its
 * advantage is how much better a rollout did than its siblings, so a group whose
 * rollouts all passed or all failed contributes an exactly zero gradient.
 * `signal` is the fraction of groups that had any spread at all, and a run whose
 * signal has collapsed is spending without learning however good its pass rate
 * looks — which is why the two are plotted on one axis rather than in two
 * panels a reader has to hold side by side.
 */
export interface JobRlPanelProps {
  run: ReturnType<typeof useJob>
}

const chartConfig = {
  pass_rate: { label: "Pass rate", color: ACCENT_COLOR },
  signal: { label: "Groups with signal", color: MUTED_SERIES },
} satisfies ChartConfig

export function JobRlPanel({ run }: JobRlPanelProps) {
  const points = React.useMemo(
    () =>
      run.diag
        .filter((d) => d.pass_rate != null)
        .map((d) => ({
          step: d.step,
          pass_rate: d.pass_rate ?? 0,
          signal: d.signal ?? 0,
        })),
    [run.diag]
  )
  const last = run.diag.at(-1)

  return (
    <div className="space-y-6">
      <Panel
        title="Practising"
        description="One iteration is one gym run: T tasks, G rollouts of each"
      >
        <div className="space-y-5">
          <dl className="grid grid-cols-2 gap-x-8 gap-y-5 border-y border-border py-4 sm:grid-cols-4">
            <Stat label="Pass rate" value={percent(last?.pass_rate)} />
            <Stat label="Signal" value={percent(last?.signal)} />
            <Stat
              label="Groups"
              value={
                last?.groups == null
                  ? "—"
                  : `${last.groups - (last.skipped_groups ?? 0)}/${last.groups}`
              }
            />
            <Stat label="KL" value={fixed(last?.kl, 4)} />
            <Stat label="Episodes" value={integer(last?.episodes)} />
            <Stat label="Abandoned" value={integer(last?.abandoned)} />
            <Stat label="Rounds / episode" value={fixed(last?.episode_rounds, 1)} />
            <Stat label="Mean return" value={fixed(last?.mean_return, 3)} />
          </dl>

          {points.length > 1 ? (
            <ChartContainer config={chartConfig} className="aspect-auto h-80 w-full">
              <LineChart data={points} margin={{ left: 4, right: 4, top: 4 }}>
                <CartesianGrid {...gridProps} />
                <XAxis
                  dataKey="step"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  {...axisProps}
                />
                {/* Both series are fractions, so the axis is pinned to [0, 1]
                    rather than fitted: a pass rate climbing from 0.02 to 0.04
                    should not fill the panel the way an autoscaled one would. */}
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                  {...yAxisProps}
                />
                <ChartTooltip
                  cursor={false}
                  content={
                    <ChartTooltipContent
                      labelFormatter={(v) => `Iteration ${v}`}
                    />
                  }
                />
                <Line
                  dataKey="pass_rate"
                  name="Pass rate"
                  type="linear"
                  stroke={ACCENT_COLOR}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  dataKey="signal"
                  name="Groups with signal"
                  type="linear"
                  stroke={MUTED_SERIES}
                  strokeWidth={1.5}
                  strokeDasharray="3 3"
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ChartContainer>
          ) : (
            <FigurePlaceholder>
              Waiting for the first batch to be rolled out…
            </FigurePlaceholder>
          )}
        </div>
      </Panel>

      <Appendix log={run.log} checkpoints={run.checkpoints} />
    </div>
  )
}

function percent(value: number | undefined): string {
  return value == null ? "—" : `${(value * 100).toFixed(0)}%`
}

function fixed(value: number | undefined, places: number): string {
  return value == null ? "—" : value.toFixed(places)
}

function integer(value: number | undefined): string {
  return value == null ? "—" : String(value)
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="pr-6">
      <dt className="eyebrow text-muted-foreground">{label}</dt>
      <dd className="figure mt-1.5 text-2xl">{value}</dd>
    </div>
  )
}
