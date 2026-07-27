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
  lossStageColor,
  VOCAB_COLOR,
  yAxisProps,
} from "@/components/training/chart-theme"
import { EmbeddingGlobe } from "@/components/training/embedding-globe"
import {
  Figure,
  FigurePlaceholder,
  StatRun,
  type Stat,
} from "@/components/training/stat-run"
import { cn } from "@/lib/utils"
import type { BatchPoint, LossPoint, VocabPoint } from "@/hooks/use-training"
import type {
  CurriculumStage,
  RunPhase,
  TokenizeStats,
  TrainStats,
} from "@/lib/training"

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

// The curriculum figure for the hero: which stage we're on plus a segmented
// progress bar — one segment per stage, sized by its epoch weight, filled by that
// stage's own progress. Completed stages read full, the current one fills live,
// upcoming ones sit empty; it doubles as a preview during tokenization (all empty)
// and reads "all full" once the run is done. Progress is derived from the global
// epoch position (monotonic across stages), so it stays smooth even though each
// stage reports its own chunk counts.
function CurriculumProgress({
  stages,
  train,
  done,
}: {
  stages: CurriculumStage[]
  train: Partial<TrainStats>
  done: boolean
}) {
  const weights = stages.map((s) => Math.max(1, s.epochs))
  const starts: number[] = []
  let acc = 0
  for (const w of weights) {
    starts.push(acc)
    acc += w
  }
  const totalEpochs = acc

  // Continuous position in epoch-units: completed global epochs plus the fraction
  // of the current epoch done (its chunk / the stage's chunk count). Pinned to the
  // end once the run finishes so every segment reads full.
  const chunkFrac =
    train.total_chunks && train.total_chunks > 0
      ? Math.min(1, (train.chunk ?? 0) / train.total_chunks)
      : 0
  const epochPos = done
    ? totalEpochs
    : Math.max(0, (train.epoch ?? 0) - 1 + chunkFrac)

  // The authoritative current stage comes from the backend frame; fall back to the
  // epoch position (e.g. during tokenization, before the first train frame).
  const currentStage =
    train.stage != null
      ? train.stage
      : Math.min(
          stages.length - 1,
          starts.findIndex(
            (start, i) => epochPos >= start && epochPos < start + weights[i]
          )
        )
  const activeIdx = currentStage < 0 ? 0 : currentStage
  const activeName =
    stages[activeIdx]?.name?.trim() ||
    train.stage_name?.trim() ||
    `Stage ${activeIdx + 1}`

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-xs font-medium text-muted-foreground">
          Curriculum
        </span>
        <span className="min-w-0 truncate text-xs text-muted-foreground tabular-nums">
          Stage {activeIdx + 1} / {stages.length}
          <span className="mx-1.5 text-muted-foreground">·</span>
          <span className="text-foreground">{activeName}</span>
        </span>
      </div>

      <div className="flex items-stretch gap-1">
        {stages.map((stage, i) => {
          const frac = Math.min(
            1,
            Math.max(0, (epochPos - starts[i]) / weights[i])
          )
          const isActive = i === activeIdx && !done
          return (
            <div
              key={i}
              className="flex flex-col gap-1"
              style={{ flexGrow: weights[i], flexBasis: 0, minWidth: 0 }}
              title={`${stage.name?.trim() || `Stage ${i + 1}`} — ${stage.epochs} epoch${stage.epochs === 1 ? "" : "s"}`}
            >
              <div
                className={cn(
                  "h-2 overflow-hidden rounded-full bg-primary/15",
                  isActive && "ring-1 ring-primary/40"
                )}
              >
                <div
                  className="h-full rounded-full bg-primary transition-[width] duration-300"
                  style={{ width: `${frac * 100}%` }}
                />
              </div>
              <span
                className={cn(
                  "truncate text-xs tabular-nums",
                  isActive ? "text-foreground" : "text-muted-foreground"
                )}
              >
                {stage.name?.trim() || `Stage ${i + 1}`}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export interface TelemetryProps {
  status: RunPhase | null
  isSft: boolean
  tokenize: Partial<TokenizeStats>
  train: Partial<TrainStats>
  loss: LossPoint[]
  vocabSeries: VocabPoint[]
  batchSeries: BatchPoint[]
  // The run's curriculum stages (empty for a plain single-pass run), for the hero's
  // stage indicator + segmented curriculum progress bar.
  curriculum: CurriculumStage[]
  // Per-report 3-D embedding projections — the hero's small spinning globe. Empty
  // until the first diagnostics sample lands.
  emb3Frames: [number, number, number][][]
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
  curriculum,
  emb3Frames,
}: TelemetryProps) {
  const tokenizing = status === "tokenizing"
  // The training figure is the hero from the moment training starts through done.
  const trainPhase = status != null && status !== "tokenizing"
  // Show the curriculum figure only for a real staged run (>1 stage), whenever a
  // run is staged/active — it previews during tokenization and fills in training.
  const showCurriculum = status != null && (curriculum?.length ?? 0) > 1

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

  // The hero's spinning embedding globe: only once training is underway and the
  // first projection has landed, and given the horizontal room beside the stat run.
  const showGlobe = trainPhase && (emb3Frames.at(-1)?.length ?? 0) >= 2

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-6">
        <div className="min-w-0 flex-1">
          <StatRun stats={stats} />
        </div>
        {showGlobe && (
          <EmbeddingGlobe
            frames={emb3Frames}
            variant="mini"
            className="hidden size-24 shrink-0 self-start sm:block"
          />
        )}
      </div>

      <div className="space-y-1.5">
        <Progress value={pct} />
        <div className="text-right text-xs text-muted-foreground tabular-nums">
          {pct}%
        </div>
      </div>

      {showCurriculum && (
        <CurriculumProgress
          stages={curriculum}
          train={train}
          done={status === "done"}
        />
      )}

      {/* Hero figure — the loss overlay once training, else the phase-1 figure. */}
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
              className="aspect-auto h-64 w-full"
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
              : "Vocabulary growth (x = merges learned)"
          }
        >
          {!isSft && vocabSeries.length > 1 ? (
            <ChartContainer
              config={vocabChartConfig}
              className="aspect-auto h-64 w-full"
            >
              <LineChart
                data={vocabSeries}
                margin={{ left: 4, right: 4, top: 4 }}
              >
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="merges" {...axisProps} />
                <ChartTooltip
                  cursor={false}
                  content={<ChartTooltipContent />}
                />
                <Line
                  dataKey="vocab"
                  type="linear"
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
            <ChartContainer
              config={batchChartConfig}
              className="aspect-auto h-48 w-full"
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
            <FigurePlaceholder className="h-48">
              Waiting for batch reports…
            </FigurePlaceholder>
          )}
        </Figure>
      )}
    </div>
  )
}
