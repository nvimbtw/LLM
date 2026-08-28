"use client"

import * as React from "react"
import { Check } from "lucide-react"
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
import {
  compact,
  Figure,
  FigurePlaceholder,
} from "@/components/training/stat-run"
import { cn } from "@/lib/utils"
import type { BatchPoint, LossPoint } from "@/hooks/use-training"
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

// One line of the phase-1 figure: what the build step produced, once it has.
function BuildStep({
  label,
  value,
  state,
}: {
  label: string
  value: string
  state: "done" | "active" | "todo"
}) {
  return (
    <li className="flex items-baseline justify-between gap-4 border-b border-border/40 py-2.5 last:border-b-0">
      <span className="flex min-w-0 items-center gap-2">
        <span
          aria-hidden
          className={cn(
            "flex size-4 shrink-0 items-center justify-center rounded-full",
            state === "done" && "bg-chart-2/20 text-chart-2",
            state === "active" &&
              "animate-pulse bg-primary/30 ring-1 ring-primary/50",
            state === "todo" && "bg-muted"
          )}
        >
          {state === "done" && <Check className="size-2.5" />}
        </span>
        <span
          className={cn(
            "truncate text-sm",
            state === "todo" ? "text-muted-foreground" : "text-foreground"
          )}
        >
          {label}
        </span>
      </span>
      <span className="shrink-0 text-sm text-muted-foreground tabular-nums">
        {value}
      </span>
    </li>
  )
}

/**
 * Phase 1, as the steps it actually consists of.
 *
 * This slot used to plot vocabulary against merges, which is a straight line by
 * construction — the vocabulary grows by exactly one entry per merge learned —
 * drawn from the one sample the tokenizer can report, since the library it wraps
 * has no callback to sample during training. A checklist says what that chart
 * was pretending to: which step the build is on, and what each one produced.
 */
function PhaseOneFigure({
  isSft,
  tokenize,
  done,
}: {
  isSft: boolean
  tokenize: Partial<TokenizeStats>
  done: boolean
}) {
  // A step is done once its own count has arrived, and active once the step
  // before it is done — the build reports each measurement as it lands, so
  // "what has not reported yet" is what is happening now.
  const state = (
    measured: boolean,
    previous: boolean
  ): "done" | "active" | "todo" =>
    measured ? "done" : previous ? "active" : "todo"

  const collected = tokenize.documents ?? tokenize.files
  const steps = isSft
    ? [
        {
          label: "Conversations rendered",
          value: compact(tokenize.conversations ?? tokenize.examples),
          state: state(tokenize.conversations != null, true),
        },
        {
          label: "Tokens written",
          value: compact(tokenize.total_tokens),
          state: state(
            tokenize.total_tokens != null,
            tokenize.conversations != null
          ),
        },
        {
          label: "Trainable (assistant replies)",
          value: compact(tokenize.trainable_tokens),
          state: state(tokenize.trainable_tokens != null, done),
        },
      ]
    : [
        {
          // `documents`, not `files`: the text is read from the data engine a
          // page at a time and there are no files. `files` is what a run queued
          // before that move reported, and its frames still render here.
          label: "Documents collected",
          value: compact(tokenize.documents ?? tokenize.files),
          state: state(collected != null, true),
        },
        {
          label: "Tokenizer trained",
          value:
            tokenize.vocab != null
              ? `${compact(tokenize.vocab)} vocab · ${compact(tokenize.merges)} merges`
              : "—",
          state: state(tokenize.vocab != null, collected != null),
        },
        {
          label: "Corpus packed",
          value:
            tokenize.total_tokens != null
              ? `${compact(tokenize.total_tokens)} tokens`
              : "—",
          state: state(tokenize.total_tokens != null, tokenize.vocab != null),
        },
      ]

  return (
    <ul className="rounded-(--radius-control-sm) border border-dashed px-4 py-1">
      {steps.map((s) => (
        <BuildStep key={s.label} {...s} />
      ))}
    </ul>
  )
}

export interface TelemetryProps {
  status: RunPhase | null
  isSft: boolean
  tokenize: Partial<TokenizeStats>
  train: Partial<TrainStats>
  loss: LossPoint[]
  batchSeries: BatchPoint[]
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
        <Figure
          caption={
            isSft
              ? "Dataset build — rendering chat conversations into the masked SFT dataset"
              : "Dataset build — tokenizer, then the packed corpus"
          }
        >
          <PhaseOneFigure
            isSft={isSft}
            tokenize={tokenize}
            done={tokenize.done === true}
          />
        </Figure>
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
