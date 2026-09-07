"use client"

import * as React from "react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import {
  axisProps,
  batchRamp,
  gridProps,
  lossEpochColor,
  lossStageColor,
  yAxisProps,
} from "@/components/training/chart-theme"
import { BuildMetrics } from "@/components/training/build-metrics"
import { BuildProgress } from "@/components/training/build-progress"
import { Figure, FigurePlaceholder } from "@/components/training/stat-run"
import type { BatchPoint, BuildPoint, LossPoint } from "@/hooks/use-training"
import type {
  CurriculumStage,
  RunPhase,
  TokenizeStats,
  TrainStats,
} from "@/lib/training"

const epochKey = (epoch: number) => `e${epoch}`

const batchChartConfig = {
  max: { label: "Max", color: batchRamp.max },
  avg: { label: "Avg", color: batchRamp.avg },
  min: { label: "Min", color: batchRamp.min },
} satisfies ChartConfig

export interface TelemetryProps {
  status: RunPhase | null
  isSft: boolean
  tokenize: Partial<TokenizeStats>
  train: Partial<TrainStats>
  loss: LossPoint[]
  batchSeries: BatchPoint[]
  // Phase 1's own curve, accumulated from its frames while the page is attached.
  buildSeries: BuildPoint[]
  // The run's curriculum stages (empty for a plain single-pass run), for the
  // per-stage colouring of the loss overlay and its legend.
  curriculum: CurriculumStage[]
  // The window the run trains at, which the build's document-length histogram is
  // measured against. It is a property of the run rather than of the build, so it
  // comes from the config and not from the phase-1 frame.
  ctxWindow: number
}

/**
 * The run's figures: the loss overlay, and the intra-batch spread beneath it.
 *
 * The numbers, the phase and the progress live in `TrainingHero` above these —
 * this is only the plotting. Each figure reserves its height and shows a
 * placeholder before its data exists, so a figure filling in never moves the one
 * below it.
 */
export function Telemetry({
  status,
  isSft,
  tokenize,
  train,
  loss,
  batchSeries,
  buildSeries,
  curriculum,
  ctxWindow,
}: TelemetryProps) {
  // The training figure is the hero from the moment training starts through done.
  const trainPhase = status != null && status !== "tokenizing"
  const showCurriculum = status != null && (curriculum?.length ?? 0) > 1

  // ── loss overlay: pivot flat points into per-chunk rows, one column/epoch ──
  const epochs = React.useMemo(() => {
    const seen = new Set<number>()
    for (const p of loss) seen.add(p.epoch)
    return [...seen].sort((a, b) => a - b)
  }, [loss])

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

  const yDomain = React.useMemo<[number, number]>(() => {
    let lo = Infinity
    let hi = -Infinity
    for (const p of loss) {
      if (!Number.isFinite(p.loss)) continue
      if (p.loss < lo) lo = p.loss
      if (p.loss > hi) hi = p.loss
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1]
    const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.08 || 1
    return [Math.max(0, lo - pad), hi + pad]
  }, [loss])

  const totalEpochs = Math.max(train.total_epochs ?? 0, ...epochs, 1)

  // Colour a (1-based, global) epoch line. For a curriculum run each stage gets
  // its own hue, swept light→dark across that stage's epochs; otherwise it's the
  // single-hue epoch ramp. The stage of an epoch comes from the cumulative stage
  // epoch counts, so no per-point stage tag is needed.
  const epochColor = React.useMemo(() => {
    if (!showCurriculum) {
      return (e: number) => lossEpochColor(e, totalEpochs)
    }
    const counts = curriculum.map((s) => Math.max(1, Math.floor(s.epochs)))
    const starts: number[] = []
    let acc = 0
    for (const c of counts) {
      starts.push(acc)
      acc += c
    }
    return (e: number) => {
      // Stage k owns global epochs (starts[k], starts[k] + counts[k]].
      let k = 0
      for (let i = 0; i < counts.length; i++) if (e > starts[i]) k = i
      const within = e - starts[k] - 1 // 0-based epoch within the stage
      const t = counts[k] > 1 ? within / (counts[k] - 1) : 0
      return lossStageColor(k, Math.min(1, Math.max(0, t)))
    }
  }, [showCurriculum, curriculum, totalEpochs])

  const lossChartConfig = React.useMemo(
    () =>
      Object.fromEntries(
        epochs.map((e) => [
          epochKey(e),
          { label: `Epoch ${e}`, color: epochColor(e) },
        ])
      ) satisfies ChartConfig,
    [epochs, epochColor]
  )

  return (
    <div className="space-y-6">
      {/* Hero figure — the loss overlay once training, else the phase-1 build. */}
      {trainPhase ? (
        <Figure
          caption={
            showCurriculum
              ? "Loss per epoch (x = chunk within epoch) — coloured by stage"
              : "Loss per epoch (x = chunk within epoch)"
          }
          aside={
            showCurriculum ? (
              <span className="flex flex-wrap items-center justify-end gap-x-2.5 gap-y-1">
                {curriculum.map((s, k) => (
                  <span
                    key={k}
                    className="flex items-center gap-1 text-xs text-muted-foreground"
                  >
                    <span
                      aria-hidden
                      className="size-2 shrink-0 rounded-full"
                      style={{ background: lossStageColor(k, 0.7) }}
                    />
                    <span className="max-w-[8rem] truncate">
                      {s.name?.trim() || `Stage ${k + 1}`}
                    </span>
                  </span>
                ))}
              </span>
            ) : epochs.length > 1 ? (
              <span className="flex items-center gap-1.5">
                <span style={{ color: lossEpochColor(1, totalEpochs) }}>
                  early
                </span>
                <span
                  aria-hidden
                  className="h-1.5 w-16 rounded-full"
                  style={{
                    background: `linear-gradient(to right, ${lossEpochColor(1, totalEpochs)}, ${lossEpochColor(totalEpochs, totalEpochs)})`,
                  }}
                />
                <span
                  style={{ color: lossEpochColor(totalEpochs, totalEpochs) }}
                >
                  late
                </span>
              </span>
            ) : undefined
          }
        >
          {loss.length > 1 ? (
            <ChartContainer
              config={lossChartConfig}
              className="aspect-auto h-80 w-full"
            >
              <LineChart data={rows} margin={{ left: 4, right: 4, top: 4 }}>
                <CartesianGrid {...gridProps} />
                <XAxis
                  dataKey="chunk"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  {...axisProps}
                />
                <YAxis
                  domain={yDomain}
                  allowDataOverflow
                  tickFormatter={(v: number) => v.toFixed(1)}
                  {...yAxisProps}
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
                    type="linear"
                    stroke={epochColor(e)}
                    strokeWidth={1.5}
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ChartContainer>
          ) : (
            <FigurePlaceholder>
              Waiting for the first loss values…
            </FigurePlaceholder>
          )}
        </Figure>
      ) : (
        <BuildProgress
          isSft={isSft}
          tokenize={tokenize}
          series={buildSeries}
        />
      )}

      {/* What the build measured. Not part of the phase-1 slot above: these are
          facts about the corpus this run trains on, and they stay readable once
          the loss overlay has taken the hero position — a finished run's page is
          where they are actually read. Renders nothing for a fine-tune, whose
          phase 1 measures none of them. */}
      <BuildMetrics tokenize={tokenize} ctxWindow={ctxWindow} />

      {/* Secondary figure — intra-batch spread (training only). Reserves its slot. */}
      {trainPhase && (
        <Figure
          caption={`Intra-batch spread (last ${Math.max(batchSeries.length, 0)} reports) — highest, average and lowest chunk loss per window`}
          minHeight="14rem"
        >
          {batchSeries.length > 1 ? (
            <ChartContainer
              config={batchChartConfig}
              className="aspect-auto h-56 w-full"
            >
              <LineChart
                data={batchSeries}
                margin={{ left: 4, right: 4, top: 4 }}
              >
                <CartesianGrid {...gridProps} />
                <XAxis
                  dataKey="step"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  {...axisProps}
                />
                <YAxis domain={["auto", "auto"]} {...yAxisProps} />
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
                    type="linear"
                    stroke={batchChartConfig[key].color}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ChartContainer>
          ) : (
            <FigurePlaceholder className="h-56">
              Waiting for batch reports…
            </FigurePlaceholder>
          )}
        </Figure>
      )}
    </div>
  )
}
