"use client"

import * as React from "react"

import type { BatchPoint, LossPoint, VocabPoint } from "@/hooks/use-training"
import {
  cancelJob,
  getJob,
  type Calibration,
  type CatalogueReport,
  type Job,
  type Verdict,
} from "@/lib/jobs"
import { attachJob, type Attachment } from "@/lib/jobs-socket"
import type {
  DiagPoint,
  TokenizeStats,
  TrainConfig,
  TrainStats,
} from "@/lib/training"

// Caps on what the charts hold. Long runs emit thousands of frames; recharts
// chokes rendering them all, so resolution is halved whenever a cap is hit — the
// curve's shape is preserved and the DOM stays light.
const MAX_LOSS_POINTS = 500
const MAX_BATCH_POINTS = 60
const MAX_DIAG_POINTS = 500
// Generated replies kept in full for the catalogue feed. A screenful and a bit —
// it is a live window, not a log.
const MAX_VERDICT_FEED = 30

function downsample<T>(arr: T[], max: number): T[] {
  let out = arr
  while (out.length > max) out = out.filter((_, i) => i % 2 === 0)
  return out
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

/**
 * One job: its durable row, plus whatever it is broadcasting right now.
 *
 * The row is fetched and refetched; the live frames arrive over a socket that
 * attaches only while the job holds the GPU. A queued or finished job simply has
 * no frames — the socket says `inactive` and the row is the whole story, which is
 * why every panel is written to render from the row first and treat the live
 * state as an enrichment.
 */
export function useJob(idJob: string) {
  const [job, setJob] = React.useState<Job | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  // Training telemetry, populated only for a `train` job.
  const [tokenize, setTokenize] = React.useState<Partial<TokenizeStats>>({})
  const [train, setTrain] = React.useState<Partial<TrainStats>>({})
  const [loss, setLoss] = React.useState<LossPoint[]>([])
  const [vocabSeries, setVocabSeries] = React.useState<VocabPoint[]>([])
  const [batchSeries, setBatchSeries] = React.useState<BatchPoint[]>([])
  const [diag, setDiag] = React.useState<DiagPoint[]>([])
  const [checkpoints, setCheckpoints] = React.useState<string[]>([])
  const [log, setLog] = React.useState<string[]>([])
  const [runConfig, setRunConfig] = React.useState<TrainConfig | null>(null)

  // Bumped once per reply a gather job produces. A nudge, not a payload: the
  // sample itself is already a row, and the rating panel refetches — so there is
  // one description of a pending sample rather than a REST shape and a frame shape
  // that drift apart.
  const [sampleTick, setSampleTick] = React.useState(0)

  // The catalogue report an auto-labelling run emits when it finishes: the
  // calibration it ran against your ratings, and what it wrote. Live only — it
  // arrives as one frame at the end and is not stored on the job, so a reload
  // falls back to the running catalogue count from /signals.
  const [catalogue, setCatalogue] = React.useState<CatalogueReport | null>(null)

  // The calibration, which arrives once before any labelling, and every verdict as
  // it is decided. Unlike the final report these are available *during* the run,
  // which is the point: a catalogue pass is minutes long and writes nothing until
  // it ends.
  const [calibration, setCalibration] = React.useState<Calibration | null>(null)
  // Only the most recent replies are kept in full. The text is what makes the feed
  // worth watching and also what makes it heavy, and nothing reads back past the
  // last screenful.
  const [verdicts, setVerdicts] = React.useState<Verdict[]>([])
  // Every score, though — they are two numbers each, and the distribution needs
  // all of them to be the distribution.
  const [verdictScores, setVerdictScores] = React.useState<
    { score: number; good: boolean }[]
  >([])

  // The live stage line for a benchmark, and the text so far for a chat turn.
  const [stage, setStage] = React.useState<string | null>(null)
  const [fraction, setFraction] = React.useState<number | null>(null)
  const [text, setText] = React.useState("")

  const attachment = React.useRef<Attachment | null>(null)

  const refresh = React.useCallback(async () => {
    try {
      setJob(await getJob(idJob))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load the job.")
    } finally {
      setIsLoading(false)
    }
  }, [idJob])

  const onFrame = React.useCallback((frame: Record<string, unknown>) => {
    switch (frame.type) {
      case "snapshot": {
        // A training snapshot carries the whole accumulated run; the simple kinds
        // carry a stage line. Which one it is is evident from the fields present.
        const snap = frame as Record<string, unknown>
        if (Array.isArray(snap.loss_series)) {
          const t = (snap.train ?? {}) as Partial<TrainStats>
          setRunConfig((snap.config as TrainConfig) ?? null)
          setTokenize((snap.tokenize as Partial<TokenizeStats>) ?? {})
          setTrain(t)
          setLoss(
            downsample(
              (snap.loss_series as [number, number][]).map(([step, l]) =>
                stepToPoint(step, l, t.total_chunks)
              ),
              MAX_LOSS_POINTS
            )
          )
          const tok = (snap.tokenize ?? {}) as Partial<TokenizeStats>
          setVocabSeries(
            tok.vocab != null ? [{ merges: tok.merges ?? 0, vocab: tok.vocab }] : []
          )
          setBatchSeries(
            ((snap.batch_series as [number, number, number, number][]) ?? [])
              .map(([step, min, avg, max]) => ({ step, min, avg, max }))
              .slice(-MAX_BATCH_POINTS)
          )
          setDiag(downsample((snap.diag_series as DiagPoint[]) ?? [], MAX_DIAG_POINTS))
          setLog((snap.log as string[]) ?? [])
        } else {
          if (typeof snap.stage === "string") setStage(snap.stage)
          if (typeof snap.fraction === "number") setFraction(snap.fraction)
          if (typeof snap.text === "string") setText(snap.text)
        }
        break
      }
      case "tokenize": {
        const s = frame as unknown as TokenizeStats
        setTokenize(s)
        setVocabSeries((prev) =>
          [...prev, { merges: s.merges, vocab: s.vocab }].slice(-400)
        )
        break
      }
      case "log":
        setLog((prev) => [...prev, String(frame.line ?? "")].slice(-500))
        break
      case "train": {
        const s = frame as unknown as TrainStats
        setTrain(s)
        if (Number.isFinite(s.loss)) {
          setLoss((prev) => {
            // Halve resolution when full so the array stays bounded instead of
            // growing without limit over a long run.
            const base =
              prev.length >= MAX_LOSS_POINTS ? prev.filter((_, i) => i % 2 === 0) : prev
            return [...base, { epoch: s.epoch, chunk: s.chunk, loss: s.loss }]
          })
        }
        // Epoch-boundary reports carry null batch stats (an epoch average, not a
        // window) — skip them so the spread chart stays purely per-window.
        if (!s.epoch_done && Number.isFinite(s.batch_avg)) {
          setBatchSeries((prev) =>
            [
              ...prev,
              { step: s.step, min: s.batch_min, avg: s.batch_avg, max: s.batch_max },
            ].slice(-MAX_BATCH_POINTS)
          )
        }
        // Diagnostics ride mid-epoch report frames (null at epoch boundaries).
        if (s.diag) {
          const point = s.diag
          setDiag((prev) => {
            const base =
              prev.length >= MAX_DIAG_POINTS ? prev.filter((_, i) => i % 2 === 0) : prev
            return [...base, point]
          })
        }
        if (s.epoch_done && s.checkpoint_path) {
          const path = s.checkpoint_path
          setCheckpoints((prev) => (prev.includes(path) ? prev : [...prev, path]))
        }
        break
      }
      case "progress":
        if (typeof frame.stage === "string") setStage(frame.stage)
        if (typeof frame.fraction === "number") setFraction(frame.fraction)
        break
      case "token":
        setText((prev) => prev + String(frame.value ?? ""))
        break
      case "sample":
        setSampleTick((n) => n + 1)
        break
      case "autolabel":
        setCatalogue(frame as unknown as CatalogueReport)
        break
      case "calibration":
        setCalibration(frame as unknown as Calibration)
        break
      case "verdict": {
        const verdict = frame as unknown as Verdict
        setVerdicts((prev) => [verdict, ...prev].slice(0, MAX_VERDICT_FEED))
        setVerdictScores((prev) => [
          ...prev,
          { score: verdict.score, good: verdict.good },
        ])
        break
      }
      case "done":
      case "error":
        // The row is authoritative for the terminal state — refetching it is what
        // moves the page out of "running" rather than trusting the frame.
        break
    }
  }, [])

  React.useEffect(() => {
    void refresh()
  }, [refresh])

  // Attached once per job, not once per status change: the socket resolves
  // `inactive` itself, and re-attaching on every refetch would thrash it.
  React.useEffect(() => {
    attachment.current?.detach()
    attachment.current = attachJob(idJob, {
      onFrame: (frame) => {
        onFrame(frame)
        if (frame.type === "done" || frame.type === "error") refresh()
      },
      // Queued or already finished: nothing streams, and the row already says so.
      onInactive: () => {},
      // A socket error is not a job error — the run continues without us. Say
      // nothing rather than mislabel a healthy run as broken.
      onError: () => {},
    })
    return () => {
      attachment.current?.detach()
      attachment.current = null
    }
  }, [idJob, onFrame, refresh])

  const cancel = React.useCallback(async () => {
    // Goes through REST rather than the socket frame: a queued job has no socket
    // attached, and this has to work in both states.
    try {
      await cancelJob(idJob)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to cancel the job.")
    } finally {
      await refresh()
    }
  }, [idJob, refresh])

  return {
    job,
    isLoading,
    error,
    refresh,
    cancel,
    // train
    runConfig,
    tokenize,
    train,
    loss,
    vocabSeries,
    batchSeries,
    diag,
    checkpoints,
    log,
    // benchmark + chat
    stage,
    fraction,
    text,
    // gather
    sampleTick,
    // autolabel
    catalogue,
    calibration,
    verdicts,
    verdictScores,
  }
}
