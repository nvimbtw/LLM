"use client"

import * as React from "react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import { Progress } from "@/components/ui/progress"
import type { BatchPoint, LossPoint } from "@/hooks/use-training"
import type { TrainStats } from "@/lib/training-socket"

// Fixed colours for the intra-batch spread chart's three lines.
const batchChartConfig = {
  max: { label: "Max", color: "hsl(0 72% 51%)" },
  avg: { label: "Avg", color: "hsl(217 91% 60%)" },
  min: { label: "Min", color: "hsl(142 71% 45%)" },
} satisfies ChartConfig

// Per-epoch line colour: a warm→cool gradient across the run so early epochs
// read warm (red/orange) and later ones cool (green), making the downward drift
// of loss from epoch to epoch easy to see at a glance.
function epochColor(epoch: number, totalEpochs: number): string {
  const t = totalEpochs > 1 ? (epoch - 1) / (totalEpochs - 1) : 0
  const hue = 12 + Math.min(1, Math.max(0, t)) * 148 // 12° (red) → 160° (green)
  return `hsl(${hue.toFixed(0)} 70% 45%)`
}

const epochKey = (epoch: number) => `e${epoch}`

function formatEta(secs: number | undefined): string {
  if (secs == null || !Number.isFinite(secs)) return "—"
  const s = Math.max(0, Math.round(secs))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-card p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 font-heading text-xl font-semibold tabular-nums">
        {value}
      </div>
    </div>
  )
}

export function TrainPanel({
  train,
  loss,
  batchSeries,
  checkpoints,
}: {
  train: Partial<TrainStats>
  loss: LossPoint[]
  batchSeries: BatchPoint[]
  checkpoints: string[]
}) {
  const totalSteps =
    (train.total_epochs ?? 0) * (train.total_chunks ?? 0) || undefined
  const pct = totalSteps
    ? Math.min(100, Math.round(((train.step ?? 0) / totalSteps) * 100))
    : 0

  // Distinct epochs present in the data, in order — one chart line each.
  const epochs = React.useMemo(() => {
    const seen = new Set<number>()
    for (const p of loss) seen.add(p.epoch)
    return [...seen].sort((a, b) => a - b)
  }, [loss])

  // Pivot the flat points into wide rows keyed by chunk: each row holds every
  // epoch's loss at that chunk position, so recharts can draw the epochs as
  // separate lines over a shared (bounded) x-axis. Missing cells stay undefined
  // and are bridged with connectNulls.
  const rows = React.useMemo(() => {
    const byChunk = new Map<number, Record<string, number>>()
    for (const p of loss) {
      let row = byChunk.get(p.chunk)
      if (!row) {
        row = { chunk: p.chunk }
        byChunk.set(p.chunk, row)
      }
      row[epochKey(p.epoch)] = p.loss
    }
    return [...byChunk.values()].sort((a, b) => a.chunk - b.chunk)
  }, [loss])

  // Cover every epoch we know about, even before its first point lands, so
  // colours stay stable as the run progresses.
  const totalEpochs = Math.max(train.total_epochs ?? 0, ...epochs, 1)

  const chartConfig = React.useMemo(
    () =>
      Object.fromEntries(
        epochs.map((e) => [
          epochKey(e),
          { label: `Epoch ${e}`, color: epochColor(e, totalEpochs) },
        ])
      ) satisfies ChartConfig,
    [epochs, totalEpochs]
  )

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat
          label="Epoch"
          value={`${train.epoch ?? 0} / ${train.total_epochs ?? 0}`}
        />
        <Stat
          label="Loss"
          value={
            train.loss != null && Number.isFinite(train.loss)
              ? train.loss.toFixed(4)
              : "—"
          }
        />
        <Stat
          label="Avg loss"
          value={
            train.avg_loss != null && Number.isFinite(train.avg_loss)
              ? train.avg_loss.toFixed(4)
              : "—"
          }
        />
        <Stat label="ETA" value={formatEta(train.eta_secs)} />
      </div>

      <div className="space-y-1.5">
        <Progress value={pct} />
        <div className="text-right text-xs text-muted-foreground tabular-nums">
          {pct}%
        </div>
      </div>

      {loss.length > 1 ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Loss per epoch (x = chunk within epoch)</span>
            {epochs.length > 1 && (
              <span className="flex items-center gap-1.5">
                <span style={{ color: epochColor(1, totalEpochs) }}>early</span>
                <span
                  aria-hidden
                  className="h-1.5 w-16 rounded-full"
                  style={{
                    background: `linear-gradient(to right, ${epochColor(1, totalEpochs)}, ${epochColor(totalEpochs, totalEpochs)})`,
                  }}
                />
                <span style={{ color: epochColor(totalEpochs, totalEpochs) }}>
                  late
                </span>
              </span>
            )}
          </div>
          <ChartContainer
            config={chartConfig}
            className="aspect-auto h-64 w-full"
          >
            <LineChart data={rows} margin={{ left: 4, right: 4, top: 4 }}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="chunk"
                type="number"
                domain={[1, "dataMax"]}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={32}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                width={44}
                tickMargin={4}
                domain={["auto", "auto"]}
              />
              <ChartTooltip
                cursor={false}
                content={
                  <ChartTooltipContent labelFormatter={(v) => `Chunk ${v}`} />
                }
              />
              {epochs.map((e) => (
                <Line
                  key={e}
                  dataKey={epochKey(e)}
                  name={`Epoch ${e}`}
                  type="monotone"
                  stroke={epochColor(e, totalEpochs)}
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ChartContainer>
        </div>
      ) : (
        <div className="flex h-64 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
          Waiting for the first loss values…
        </div>
      )}

      {batchSeries.length > 1 && (
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground">
            Intra-batch spread (last {batchSeries.length} reports) — highest,
            average and lowest chunk loss within each window
          </div>
          <ChartContainer
            config={batchChartConfig}
            className="aspect-auto h-48 w-full"
          >
            <LineChart
              data={batchSeries}
              margin={{ left: 4, right: 4, top: 4 }}
            >
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="step"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={32}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                width={44}
                tickMargin={4}
                domain={["auto", "auto"]}
              />
              <ChartTooltip
                cursor={false}
                content={
                  <ChartTooltipContent labelFormatter={(v) => `Step ${v}`} />
                }
              />
              {(["max", "avg", "min"] as const).map((key) => (
                <Line
                  key={key}
                  dataKey={key}
                  name={batchChartConfig[key].label}
                  type="monotone"
                  stroke={batchChartConfig[key].color}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ChartContainer>
        </div>
      )}

      {checkpoints.length > 0 && (
        <div className="space-y-1">
          <div className="text-sm font-medium">Checkpoints</div>
          <ul className="space-y-0.5 font-mono text-xs text-muted-foreground">
            {checkpoints.map((cp) => (
              <li key={cp} className="truncate">
                {cp}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
