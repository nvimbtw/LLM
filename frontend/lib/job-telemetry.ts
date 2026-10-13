import type {
  CheckpointFrame,
  DiagPoint,
  TokenizeStats,
  TrainConfig,
  TrainStats,
  TrainingSnapshot,
} from "./training"

export interface LossPoint {
  epoch: number
  chunk: number
  loss: number
}
export interface BatchPoint {
  step: number
  min: number
  avg: number
  max: number
}
export interface BuildPoint {
  elapsed: number
  rate: number
  tokens: number
  documents: number
}

export interface JobTelemetry {
  tokenize: Partial<TokenizeStats>
  train: Partial<TrainStats>
  loss: LossPoint[]
  batchSeries: BatchPoint[]
  diag: DiagPoint[]
  buildSeries: BuildPoint[]
  checkpoints: CheckpointFrame[]
  log: string[]
  runConfig: TrainConfig | null
  stage: string | null
  fraction: number | null
  text: string
}

export type JobTelemetryEvent =
  | { type: "training-snapshot"; snapshot: TrainingSnapshot }
  | { type: "progress"; stage?: string; fraction?: number; text?: string }
  | { type: "tokenize"; stats: Partial<TokenizeStats> }
  | { type: "train"; stats: TrainStats }
  | { type: "checkpoint"; checkpoint: CheckpointFrame }
  | { type: "log"; line: string }
  | { type: "token"; value: string }

const MAX_LOSS_POINTS = 500
const MAX_BATCH_POINTS = 60
const MAX_DIAG_POINTS = 500
const MAX_BUILD_POINTS = 300
const MAX_LOG_LINES = 500

/** Creates independent empty telemetry for a newly selected job. */
export function createJobTelemetry(): JobTelemetry {
  return {
    tokenize: {},
    train: {},
    loss: [],
    batchSeries: [],
    diag: [],
    buildSeries: [],
    checkpoints: [],
    log: [],
    runConfig: null,
    stage: null,
    fraction: null,
    text: "",
  }
}

/**
 * Translates the existing relayed engine frame vocabulary into reducer events.
 * Training payloads retain the legacy wire assertions, including sparse RL diagnostics;
 * this compatibility seam does not claim complete runtime validation of engine-specific fields.
 */
export function jobTelemetryEvent(
  frame: Record<string, unknown>
): JobTelemetryEvent | null {
  switch (frame.type) {
    case "snapshot":
      if (Array.isArray(frame.loss_series))
        return {
          type: "training-snapshot",
          snapshot: frame as unknown as TrainingSnapshot,
        }
      return {
        type: "progress",
        stage: typeof frame.stage === "string" ? frame.stage : undefined,
        fraction:
          typeof frame.fraction === "number" ? frame.fraction : undefined,
        text: typeof frame.text === "string" ? frame.text : undefined,
      }
    case "progress":
      return {
        type: "progress",
        stage: typeof frame.stage === "string" ? frame.stage : undefined,
        fraction:
          typeof frame.fraction === "number" ? frame.fraction : undefined,
      }
    case "tokenize":
      return { type: "tokenize", stats: frame as Partial<TokenizeStats> }
    case "train":
      return { type: "train", stats: frame as unknown as TrainStats }
    case "checkpoint":
      return {
        type: "checkpoint",
        checkpoint: frame as unknown as CheckpointFrame,
      }
    case "log":
      return { type: "log", line: String(frame.line ?? "") }
    case "token":
      return { type: "token", value: String(frame.value ?? "") }
    default:
      return null
  }
}

/** Halves historical chart resolution until its point count fits the given positive cap. */
function downsample<T>(points: T[], cap: number): T[] {
  let result = points
  while (result.length > cap)
    result = result.filter((_, index) => index % 2 === 0)
  return result
}

/** Appends a live point after halving a full history, preserving the newest report at every cap boundary. */
function appendPoint<T>(points: T[], point: T, cap: number): T[] {
  const history =
    points.length >= cap ? points.filter((_, index) => index % 2 === 0) : points
  return [...history, point]
}

/** Inverts a 1-based global step into epoch/chunk coordinates; older snapshots without chunk counts use epoch one. */
function stepToPoint(
  step: number,
  loss: number,
  totalChunks: number | undefined
): LossPoint {
  if (!totalChunks || totalChunks <= 0) return { epoch: 1, chunk: step, loss }
  return {
    epoch: Math.floor((step - 1) / totalChunks) + 1,
    chunk: ((step - 1) % totalChunks) + 1,
    loss,
  }
}

/** Replaces snapshot-owned histories while leaving separately replayed build/checkpoint frames intact. */
function restoreSnapshot(
  previous: JobTelemetry,
  snapshot: TrainingSnapshot
): JobTelemetry {
  const train = snapshot.train ?? {}
  return {
    ...previous,
    runConfig: snapshot.config ?? null,
    tokenize: snapshot.tokenize ?? {},
    train,
    loss: downsample(
      snapshot.loss_series.map(([step, loss]) =>
        stepToPoint(step, loss, train.total_chunks)
      ),
      MAX_LOSS_POINTS
    ),
    batchSeries: (snapshot.batch_series ?? [])
      .map(([step, min, avg, max]) => ({ step, min, avg, max }))
      .slice(-MAX_BATCH_POINTS),
    diag: downsample(snapshot.diag_series ?? [], MAX_DIAG_POINTS),
    log: snapshot.log ?? [],
  }
}

/** Merges newly measured build counters and records speed only when the frame actually measured it. */
function recordBuild(
  previous: JobTelemetry,
  stats: Partial<TokenizeStats>
): JobTelemetry {
  const { elapsed_secs: elapsed, tokens_per_sec: rate } = stats
  const buildSeries =
    elapsed != null && rate != null
      ? appendPoint(
          previous.buildSeries,
          {
            elapsed,
            rate,
            tokens: stats.total_tokens ?? 0,
            documents: stats.documents_done ?? 0,
          },
          MAX_BUILD_POINTS
        )
      : previous.buildSeries
  return {
    ...previous,
    tokenize: { ...previous.tokenize, ...stats },
    buildSeries,
  }
}

/** Records a report's loss, batch spread and optional diagnostics without turning absent measurements into zero. */
function recordTraining(
  previous: JobTelemetry,
  stats: TrainStats
): JobTelemetry {
  return {
    ...previous,
    train: stats,
    loss: Number.isFinite(stats.loss)
      ? appendPoint(
          previous.loss,
          { epoch: stats.epoch, chunk: stats.chunk, loss: stats.loss },
          MAX_LOSS_POINTS
        )
      : previous.loss,
    batchSeries:
      !stats.epoch_done && Number.isFinite(stats.batch_avg)
        ? [
            ...previous.batchSeries,
            {
              step: stats.step,
              min: stats.batch_min,
              avg: stats.batch_avg,
              max: stats.batch_max,
            },
          ].slice(-MAX_BATCH_POINTS)
        : previous.batchSeries,
    diag: stats.diag
      ? appendPoint(previous.diag, stats.diag, MAX_DIAG_POINTS)
      : previous.diag,
  }
}

/** Folds one typed engine event into immutable chart/display state; terminal job status remains REST-owned. */
export function reduceJobTelemetry(
  previous: JobTelemetry,
  event: JobTelemetryEvent
): JobTelemetry {
  switch (event.type) {
    case "training-snapshot":
      return restoreSnapshot(previous, event.snapshot)
    case "tokenize":
      return recordBuild(previous, event.stats)
    case "train":
      return recordTraining(previous, event.stats)
    case "log":
      return {
        ...previous,
        log: [...previous.log, event.line].slice(-MAX_LOG_LINES),
      }
    case "token":
      return { ...previous, text: previous.text + event.value }
    case "progress":
      return {
        ...previous,
        stage: event.stage ?? previous.stage,
        fraction: event.fraction ?? previous.fraction,
        text: event.text ?? previous.text,
      }
    case "checkpoint": {
      const checkpoint = event.checkpoint
      return checkpoint.final ||
        previous.checkpoints.some(
          (item) => item.model_id === checkpoint.model_id
        )
        ? previous
        : { ...previous, checkpoints: [...previous.checkpoints, checkpoint] }
    }
  }
}
