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
import {
  axisProps,
  batchRamp,
  gridProps,
  lossEpochColor,
  VOCAB_COLOR,
  yAxisProps,
} from "@/components/training/chart-theme"
import { DiagnosticsPanel } from "@/components/training/diagnostics-panel"
import { Figure, FigurePlaceholder, StatRun, type Stat } from "@/components/training/stat-run"
import type { BatchPoint, LossPoint, VocabPoint } from "@/hooks/use-training"
import type {
  DiagPoint,
  JobStatus,
  TokenizeStats,
  TrainStats,
} from "@/lib/training-socket"

function compact(n: number | undefined): string {
  if (n == null) return "—"
  return Intl.NumberFormat("en-US", { notation: "compact" }).format(n)
}

function formatEta(secs: number | undefined): string {
  if (secs == null || !Number.isFinite(secs)) return "—"
  const s = Math.max(0, Math.round(secs))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

function loss4(v: number | undefined): string {
  return v != null && Number.isFinite(v) ? v.toFixed(4) : "—"
}

const epochKey = (epoch: number) => `e${epoch}`

const batchChartConfig = {
  max: { label: "Max", color: batchRamp.max },
  avg: { label: "Avg", color: batchRamp.avg },
  min: { label: "Min", color: batchRamp.min },
} satisfies ChartConfig

const vocabChartConfig = {
  vocab: { label: "Vocab size", color: VOCAB_COLOR },
} satisfies ChartConfig

export interface TelemetryProps {
  status: JobStatus | null
  isSft: boolean
  tokenize: Partial<TokenizeStats>
  train: Partial<TrainStats>
  loss: LossPoint[]
  vocabSeries: VocabPoint[]
  batchSeries: BatchPoint[]
  diag: DiagPoint[]
}

// The main column's live telemetry: a stat run pinned at the top, a thin progress
// rule, the hero figure (loss once training, else the phase-1 figure), and the
// secondary spread figure. Every region keeps its slot and fills in — figures
// reserve their height and show a placeholder before data, so nothing pops.
export function Telemetry({
  status,
  isSft,
  tokenize,
  train,
  loss,
  vocabSeries,
  batchSeries,
  diag,
}: TelemetryProps) {
  const tokenizing = status === "tokenizing"
  // The training figure is the hero from the moment training starts through done.
  const trainPhase = status != null && status !== "tokenizing"

  // ── stat run (phase- and mode-aware) ──────────────────────────────────────
  const stats: Stat[] = trainPhase
    ? [
        {
          label: "Epoch",
          value: `${train.epoch ?? 0} / ${train.total_epochs ?? 0}`,
        },
        { label: "Loss", value: loss4(train.loss) },
        { label: "Avg loss", value: loss4(train.avg_loss) },
        { label: "ETA", value: formatEta(train.eta_secs) },
      ]
    : isSft
      ? [
          { label: "Conversations", value: compact(tokenize.conversations) },
          { label: "Total tokens", value: compact(tokenize.total_tokens) },
          { label: "Trainable", value: compact(tokenize.trainable_tokens) },
        ]
      : [
          { label: "Files", value: compact(tokenize.files) },
          { label: "Characters", value: compact(tokenize.chars) },
          { label: "Vocab size", value: compact(tokenize.vocab) },
          { label: "Merges", value: compact(tokenize.merges) },
        ]

  // ── progress ───────────────────────────────────────────────────────────────
  const totalSteps =
    (train.total_epochs ?? 0) * (train.total_chunks ?? 0) || undefined
  const trainPct = totalSteps
    ? Math.min(100, Math.round(((train.step ?? 0) / totalSteps) * 100))
    : 0
  const pct = trainPhase ? trainPct : Math.round((tokenize.progress ?? 0) * 100)

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

  const lossChartConfig = React.useMemo(
    () =>
      Object.fromEntries(
        epochs.map((e) => [
          epochKey(e),
          { label: `Epoch ${e}`, color: lossEpochColor(e, totalEpochs) },
        ])
      ) satisfies ChartConfig,
    [epochs, totalEpochs]
  )

  return (
    <div className="space-y-6">
      <StatRun stats={stats} />

      <div className="space-y-1.5">
        <Progress value={pct} />
        <div className="text-right text-xs text-muted-foreground tabular-nums">
          {pct}%
        </div>
      </div>

      {/* Hero figure — the loss overlay once training, else the phase-1 figure. */}
      {trainPhase ? (
        <Figure
          caption="Loss per epoch (x = chunk within epoch)"
          aside={
            epochs.length > 1 ? (
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
                <span style={{ color: lossEpochColor(totalEpochs, totalEpochs) }}>
                  late
                </span>
              </span>
            ) : undefined
          }
        >
          {loss.length > 1 ? (
            <ChartContainer config={lossChartConfig} className="aspect-auto h-64 w-full">
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
                  content={<ChartTooltipContent labelFormatter={(v) => `Chunk ${v}`} />}
                />
                {epochs.map((e) => (
                  <Line
                    key={e}
                    dataKey={epochKey(e)}
                    name={`Epoch ${e}`}
                    type="monotone"
                    stroke={lossEpochColor(e, totalEpochs)}
                    strokeWidth={1.5}
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ChartContainer>
          ) : (
            <FigurePlaceholder>Waiting for the first loss values…</FigurePlaceholder>
          )}
        </Figure>
      ) : (
        <Figure
          caption={
            isSft
              ? "Dataset build — rendering chat conversations into the masked SFT dataset"
              : "Vocabulary growth (x = merges learned)"
          }
        >
          {!isSft && vocabSeries.length > 1 ? (
            <ChartContainer config={vocabChartConfig} className="aspect-auto h-64 w-full">
              <LineChart data={vocabSeries} margin={{ left: 4, right: 4, top: 4 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="merges" {...axisProps} />
                <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                <Line
                  dataKey="vocab"
                  type="monotone"
                  stroke="var(--color-vocab)"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ChartContainer>
          ) : (
            <FigurePlaceholder>
              {isSft
                ? tokenizing
                  ? "Building the masked SFT dataset…"
                  : "Dataset built."
                : "Building the BPE tokenizer…"}
            </FigurePlaceholder>
          )}
        </Figure>
      )}

      {/* Secondary figure — intra-batch spread (training only). Reserves its slot. */}
      {trainPhase && (
        <Figure
          caption={`Intra-batch spread (last ${Math.max(batchSeries.length, 0)} reports) — highest, average and lowest chunk loss per window`}
          minHeight="12rem"
        >
          {batchSeries.length > 1 ? (
            <ChartContainer config={batchChartConfig} className="aspect-auto h-48 w-full">
              <LineChart data={batchSeries} margin={{ left: 4, right: 4, top: 4 }}>
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
                  content={<ChartTooltipContent labelFormatter={(v) => `Step ${v}`} />}
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
          ) : (
            <FigurePlaceholder className="h-48">
              Waiting for batch reports…
            </FigurePlaceholder>
          )}
        </Figure>
      )}

      {/* Deep-dive diagnostics — tabbed, below the loss/spread heroes. */}
      {trainPhase && <DiagnosticsPanel diag={diag} />}
    </div>
  )
}
