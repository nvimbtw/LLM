"use client"

import * as React from "react"

import {
  attachTraining,
  listJobs,
  startTraining,
  type Attachment,
  type DiagPoint,
  type FinishedModel,
  type JobStatus,
  type TokenizeStats,
  type TrainConfig,
  type TrainStats,
} from "@/lib/training-socket"

export const DEFAULT_CONFIG: TrainConfig = {
  model_name: "my-model",
  dimensions: 64,
  ctx_window: 64,
  heads: 4,
  layers: 2,
  epochs: 5,
  // Pretrain default. Raised from 1e-5 → 1e-4 now that the f32 master-weight fix
  // (ISSUE-0052) lifted the old f16-quantization LR cap: at 1e-5 from-scratch
  // pretrains were badly undertrained. 1e-4 trains stably post-fix (ISSUE-0057).
  lr: 0.0001,
  ff_multiplier: 4,
  max_vocab: 1024,
  datasets: [],
  curriculum: [],
  // Shuffle documents each epoch by default: it isolates documents (an
  // <|endoftext|> separator between them) and removes fixed-order bias at no cost.
  shuffle: true,
  mode: "pretrain",
}

// Cap the number of loss points kept for the chart. Long runs emit thousands of
// frames; recharts chokes rendering them all, so we halve the resolution
// whenever we hit the cap — the curve's shape is preserved, the DOM stays light.
const MAX_LOSS_POINTS = 500

// The batch-spread chart is a short rolling window of the most recent reports.
const MAX_BATCH_POINTS = 60

// Diagnostics points mirror the loss cap: halve resolution when full so the
// whole-run trajectory is preserved without unbounded growth.
const MAX_DIAG_POINTS = 500

function downsample<T>(arr: T[], max: number): T[] {
  let out = arr
  while (out.length > max) out = out.filter((_, i) => i % 2 === 0)
  return out
}

// One recorded loss value, tagged with the epoch it belongs to and its position
// (chunk) within that epoch. The chart overlays each epoch as its own line using
// `chunk` as the shared x-axis, so the plot width stays bounded to a single
// epoch instead of growing with the whole run.
export interface LossPoint {
  epoch: number
  chunk: number
  loss: number
}

// Reconstruct (epoch, chunk) from a global step for the reattach snapshot, which
// only carries [step, loss]. The backend numbers steps 1-indexed globally as
// step = (epoch - 1) * total_chunks + chunk, so we invert that here.
function stepToPoint(
  step: number,
  loss: number,
  totalChunks: number | undefined
): LossPoint {
  if (!totalChunks || totalChunks <= 0) return { epoch: 1, chunk: step, loss }
  const idx = step - 1
  return {
    epoch: Math.floor(idx / totalChunks) + 1,
    chunk: (idx % totalChunks) + 1,
    loss,
  }
}

export interface VocabPoint {
  merges: number
  vocab: number
}

// One report's intra-batch loss spread, plotted over the global step on the
// rolling batch-spread chart.
export interface BatchPoint {
  step: number
  min: number
  avg: number
  max: number
}

// Owns all live state for the training page and keeps a single socket attached
// to the current job. Re-attaches to an in-flight run on mount so a reload (or a
// new tab) resumes watching instead of starting over.
export function useTraining() {
  const [jobId, setJobId] = React.useState<string | null>(null)
  const [status, setStatus] = React.useState<JobStatus | null>(null)
  const [tokenize, setTokenize] = React.useState<Partial<TokenizeStats>>({})
  const [train, setTrain] = React.useState<Partial<TrainStats>>({})
  const [loss, setLoss] = React.useState<LossPoint[]>([])
  const [vocabSeries, setVocabSeries] = React.useState<VocabPoint[]>([])
  const [batchSeries, setBatchSeries] = React.useState<BatchPoint[]>([])
  const [diag, setDiag] = React.useState<DiagPoint[]>([])
  const [checkpoints, setCheckpoints] = React.useState<string[]>([])
  const [log, setLog] = React.useState<string[]>([])
  const [error, setError] = React.useState<string | null>(null)
  // The config the current/last run was started with (or reattached from the
  // snapshot), so the shell can label the phase stepper by mode and recap the run.
  const [runConfig, setRunConfig] = React.useState<TrainConfig | null>(null)
  // The model a completed run produced, for the "Continue to SFT" flow. Null
  // until a run finishes successfully; cleared when a new run starts.
  const [finishedModel, setFinishedModel] =
    React.useState<FinishedModel | null>(null)
  // Sticky across a subsequent SFT start so the stepper keeps reading "Pretrain ✓"
  // through the continuation. Only a *fresh* pretrain resets it.
  const [pretrainDone, setPretrainDone] = React.useState(false)

  const attachment = React.useRef<Attachment | null>(null)

  const isActive = status === "tokenizing" || status === "training"

  const attach = React.useCallback((id: string) => {
    attachment.current?.detach()
    attachment.current = attachTraining(id, {
      onSnapshot: (snap) => {
        setStatus(snap.status)
        if (snap.config) setRunConfig(snap.config)
        // A run that finished while we were away carries the trained model's
        // identifiers on its snapshot — populate the same state onDone would, so a
        // reattached, just-finished pretrain still offers "Continue to SFT".
        if (snap.status === "done" && snap.id_model && snap.model_name) {
          setFinishedModel({ id: snap.id_model, name: snap.model_name })
          if (snap.config?.mode === "pretrain") setPretrainDone(true)
        }
        setTokenize(snap.tokenize ?? {})
        setTrain(snap.train ?? {})
        setLoss(
          downsample(
            (snap.loss_series ?? []).map(([step, l]) =>
              stepToPoint(step, l, snap.train?.total_chunks)
            ),
            MAX_LOSS_POINTS
          )
        )
        setVocabSeries(
          snap.tokenize?.vocab != null
            ? [
                {
                  merges: snap.tokenize.merges ?? 0,
                  vocab: snap.tokenize.vocab,
                },
              ]
            : []
        )
        setBatchSeries(
          (snap.batch_series ?? [])
            .map(([step, min, avg, max]) => ({ step, min, avg, max }))
            .slice(-MAX_BATCH_POINTS)
        )
        setDiag(downsample(snap.diag_series ?? [], MAX_DIAG_POINTS))
        setLog(snap.log ?? [])
        setError(snap.error ?? null)
      },
      onTokenize: (s) => {
        setTokenize(s)
        setVocabSeries((prev) =>
          [...prev, { merges: s.merges, vocab: s.vocab }].slice(-400)
        )
        // Tokenization finished → the run moves on to training. Flip the phase
        // now (before the first loss frame) so the training panel appears and a
        // slow/stalled GPU start is visible rather than looking stuck on
        // tokenization.
        if (s.done) setStatus("training")
      },
      onLog: (line) => setLog((prev) => [...prev, line].slice(-500)),
      onTrain: (s) => {
        setStatus("training")
        setTrain(s)
        if (Number.isFinite(s.loss)) {
          setLoss((prev) => {
            // Halve resolution when full so the array stays bounded instead of
            // growing without limit over a long run.
            const base =
              prev.length >= MAX_LOSS_POINTS
                ? prev.filter((_, i) => i % 2 === 0)
                : prev
            return [...base, { epoch: s.epoch, chunk: s.chunk, loss: s.loss }]
          })
        }
        // Epoch-boundary reports carry null batch stats (an epoch average, not a
        // window) — skip them so the spread chart stays purely per-window.
        if (!s.epoch_done && Number.isFinite(s.batch_avg)) {
          setBatchSeries((prev) =>
            [
              ...prev,
              {
                step: s.step,
                min: s.batch_min,
                avg: s.batch_avg,
                max: s.batch_max,
              },
            ].slice(-MAX_BATCH_POINTS)
          )
        }
        // Diagnostics ride mid-epoch report frames (null at epoch boundaries).
        if (s.diag) {
          const point = s.diag
          setDiag((prev) => {
            const base =
              prev.length >= MAX_DIAG_POINTS
                ? prev.filter((_, i) => i % 2 === 0)
                : prev
            return [...base, point]
          })
        }
        if (s.epoch_done && s.checkpoint_path) {
          const path = s.checkpoint_path
          setCheckpoints((prev) =>
            prev.includes(path) ? prev : [...prev, path]
          )
        }
      },
      onDone: (info) => {
        setStatus(info.status as JobStatus)
        // Only a successful run carries a registered model. Capture it for the
        // continuation flow, and remember that a pretrain succeeded so the stepper
        // stays "Pretrain ✓" even after an SFT run starts and resets live state.
        if (info.status === "done" && info.id_model && info.model_name) {
          setFinishedModel({ id: info.id_model, name: info.model_name })
          setRunConfig((cfg) => {
            if (cfg?.mode === "pretrain") setPretrainDone(true)
            return cfg
          })
        }
      },
      onError: (msg) => setError(msg),
    })
  }, [])

  const start = React.useCallback(
    async (config: TrainConfig) => {
      setTokenize({})
      setTrain({})
      setLoss([])
      setVocabSeries([])
      setBatchSeries([])
      setDiag([])
      setCheckpoints([])
      setLog([])
      setError(null)
      setStatus("tokenizing")
      setRunConfig(config)
      setFinishedModel(null)
      // A fresh pretrain clears the "Pretrain ✓" marker; starting an SFT run
      // (the continuation) keeps it so the stepper reads Pretrain ✓ → SFT active.
      if (config.mode === "pretrain") setPretrainDone(false)
      try {
        const id = await startTraining(config)
        setJobId(id)
        attach(id)
      } catch (e) {
        setStatus("failed")
        setError(e instanceof Error ? e.message : "Failed to start training.")
      }
    },
    [attach]
  )

  const cancel = React.useCallback(() => {
    attachment.current?.cancel()
  }, [])

  // On mount, re-attach to a run that's still tokenizing/training.
  React.useEffect(() => {
    let cancelled = false
    listJobs()
      .then((jobs) => {
        if (cancelled) return
        const active = jobs.find(
          (j) => j.status === "tokenizing" || j.status === "training"
        )
        if (active) {
          setJobId(active.job_id)
          setStatus(active.status)
          attach(active.job_id)
        }
      })
      .catch(() => {})
    return () => {
      cancelled = true
      attachment.current?.detach()
    }
  }, [attach])

  return {
    jobId,
    status,
    isActive,
    runConfig,
    finishedModel,
    pretrainDone,
    tokenize,
    train,
    loss,
    vocabSeries,
    batchSeries,
    diag,
    checkpoints,
    log,
    error,
    start,
    cancel,
  }
}
