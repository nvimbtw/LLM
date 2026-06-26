"use client"

import * as React from "react"

import {
  attachTraining,
  listJobs,
  startTraining,
  type Attachment,
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
  lr: 0.01,
  ff_multiplier: 4,
  max_vocab: 1024,
}

export interface LossPoint {
  step: number
  loss: number
}

export interface VocabPoint {
  merges: number
  vocab: number
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
  const [checkpoints, setCheckpoints] = React.useState<string[]>([])
  const [log, setLog] = React.useState<string[]>([])
  const [error, setError] = React.useState<string | null>(null)

  const attachment = React.useRef<Attachment | null>(null)

  const isActive = status === "tokenizing" || status === "training"

  const attach = React.useCallback((id: string) => {
    attachment.current?.detach()
    attachment.current = attachTraining(id, {
      onSnapshot: (snap) => {
        setStatus(snap.status)
        setTokenize(snap.tokenize ?? {})
        setTrain(snap.train ?? {})
        setLoss((snap.loss_series ?? []).map(([step, l]) => ({ step, loss: l })))
        setVocabSeries(
          snap.tokenize?.vocab != null
            ? [{ merges: snap.tokenize.merges ?? 0, vocab: snap.tokenize.vocab }]
            : []
        )
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
          setLoss((prev) => [...prev, { step: s.step, loss: s.loss }])
        }
        if (s.epoch_done && s.checkpoint_path) {
          const path = s.checkpoint_path
          setCheckpoints((prev) =>
            prev.includes(path) ? prev : [...prev, path]
          )
        }
      },
      onDone: (info) => setStatus(info.status as JobStatus),
      onError: (msg) => setError(msg),
    })
  }, [])

  const start = React.useCallback(
    async (config: TrainConfig) => {
      setTokenize({})
      setTrain({})
      setLoss([])
      setVocabSeries([])
      setCheckpoints([])
      setLog([])
      setError(null)
      setStatus("tokenizing")
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
    tokenize,
    train,
    loss,
    vocabSeries,
    checkpoints,
    log,
    error,
    start,
    cancel,
  }
}
