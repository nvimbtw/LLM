"use client"

import * as React from "react"

import type { BatchPoint, LossPoint, VocabPoint } from "@/hooks/use-training"
import {
  cancelJob,
  getJob,
  type Calibration,
  type CatalogueReport,
  type Job,
  type SynthAccepted,
  type SynthCall,
  type SynthReport,
  type TeachAccepted,
  type TeachCall,
  type TeachReport,
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
// Generated conversations kept in full. Matches the backend's own snapshot window,
// so a reattach and a live run show the same depth of feed.
const MAX_SYNTH_FEED = 24
// Triples kept in full. Fewer than the synth feed because each carries three
// answers, and it matches the backend's own `TEACH_FEED` so a reattach and a live
// run show the same depth.
const MAX_TEACH_FEED = 8

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

  // A generation run: one record per model call (which the figures are drawn
  // from), the recent conversations in full, and the report that lands at the end.
  // The calls are kept in full rather than downsampled — a run is tens of calls,
  // not thousands of steps.
  const [synthCalls, setSynthCalls] = React.useState<SynthCall[]>([])
  const [synthAccepted, setSynthAccepted] = React.useState<SynthAccepted[]>([])
  const [synthAdded, setSynthAdded] = React.useState(0)
  const [synth, setSynth] = React.useState<SynthReport | null>(null)

  // A local-teacher run. Same shape of accumulation as a generation run, kept in
  // its own state rather than folded into it: the unit is a question with up to
  // three answers, not a conversation, and the per-tier tally has no counterpart on
  // the synth side.
  //
  // Three counters because the run has three units, and holding only the first is
  // what let a "45 triples" headline sit beside tier bars summing to 75.
  const [teachCalls, setTeachCalls] = React.useState<TeachCall[]>([])
  const [teachAccepted, setTeachAccepted] = React.useState<TeachAccepted[]>([])
  const [teachQuestions, setTeachQuestions] = React.useState(0)
  const [teachAnswers, setTeachAnswers] = React.useState(0)
  const [teachComplete, setTeachComplete] = React.useState(0)
  // `[one tier, two tiers, all three]`, so the coverage figure draws the same
  // whether the page attached mid-run or is reading a finished report.
  const [teachByTiers, setTeachByTiers] = React.useState<number[]>([0, 0, 0])
  const [teachPerTier, setTeachPerTier] = React.useState<[string, number][]>([])
  const [teach, setTeach] = React.useState<TeachReport | null>(null)

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
            tok.vocab != null
              ? [{ merges: tok.merges ?? 0, vocab: tok.vocab }]
              : []
          )
          setBatchSeries(
            ((snap.batch_series as [number, number, number, number][]) ?? [])
              .map(([step, min, avg, max]) => ({ step, min, avg, max }))
              .slice(-MAX_BATCH_POINTS)
          )
          setDiag(
            downsample((snap.diag_series as DiagPoint[]) ?? [], MAX_DIAG_POINTS)
          )
          setLog((snap.log as string[]) ?? [])
        } else if (Array.isArray(snap.calls)) {
          // A generation snapshot: the whole accumulation, so a page attaching ten
          // minutes in draws the run so far rather than starting from empty.
          setSynthCalls(snap.calls as SynthCall[])
          setSynthAccepted((snap.accepted as SynthAccepted[]) ?? [])
          setSynthAdded(typeof snap.added === "number" ? snap.added : 0)
          if (typeof snap.stage === "string") setStage(snap.stage)
          if (typeof snap.fraction === "number") setFraction(snap.fraction)
        } else if (Array.isArray(snap.reports)) {
          // A teaching snapshot. Told apart by `reports` rather than `calls` — the
          // two accumulations are the same idea over different units, and a shared
          // field name would make a teach snapshot parse as a synth one.
          setTeachCalls(snap.reports as TeachCall[])
          setTeachAccepted((snap.accepted as TeachAccepted[]) ?? [])
          setTeachQuestions(
            typeof snap.questions === "number" ? snap.questions : 0
          )
          setTeachAnswers(typeof snap.answers === "number" ? snap.answers : 0)
          setTeachComplete(
            typeof snap.complete_triples === "number"
              ? snap.complete_triples
              : 0
          )
          setTeachByTiers(
            Array.isArray(snap.by_tiers)
              ? (snap.by_tiers as number[])
              : [0, 0, 0]
          )
          setTeachPerTier((snap.per_tier as [string, number][]) ?? [])
          if (typeof snap.stage === "string") setStage(snap.stage)
          if (typeof snap.fraction === "number") setFraction(snap.fraction)
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
      case "call": {
        const call = frame as unknown as SynthCall
        setSynthCalls((prev) => [...prev, call])
        setSynthAdded(call.total)
        break
      }
      case "accepted": {
        const accepted = frame as unknown as SynthAccepted
        setSynthAccepted((prev) => [accepted, ...prev].slice(0, MAX_SYNTH_FEED))
        break
      }
      case "synth":
        setSynth(frame as unknown as SynthReport)
        break
      case "teach_call": {
        const call = frame as unknown as TeachCall
        setTeachCalls((prev) => [...prev, call])
        setTeachQuestions(call.total_questions)
        setTeachAnswers(call.total_answers)
        setTeachComplete(call.total_complete)
        setTeachByTiers(call.total_by_tiers ?? [0, 0, 0])
        break
      }
      case "teach_triple": {
        const triple = frame as unknown as TeachAccepted
        setTeachAccepted((prev) => [triple, ...prev].slice(0, MAX_TEACH_FEED))
        // Tallied here rather than read off the final report, so a page watching a
        // run sees whether the budgets are separating while there is still time to
        // stop a run that is putting every answer in one bucket.
        setTeachPerTier((prev) => {
          const next = new Map(prev)
          for (const answer of triple.answers) {
            next.set(answer.filed_as, (next.get(answer.filed_as) ?? 0) + 1)
          }
          return [...next.entries()]
        })
        break
      }
      case "teach":
        setTeach(frame as unknown as TeachReport)
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

  // Inlined rather than calling `refresh`, so no state is written synchronously
  // from the effect body; `refresh` stays for the socket's terminal frames and
  // for `cancel`. Same shape as `use-signals.ts`.
  React.useEffect(() => {
    let active = true
    getJob(idJob)
      .then((next) => {
        if (!active) return
        setJob(next)
        setError(null)
      })
      .catch((e: unknown) => {
        if (!active) return
        setError(e instanceof Error ? e.message : "Failed to load the job.")
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [idJob])

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
    // synth
    synthCalls,
    synthAccepted,
    synthAdded,
    synth,
    // teach
    teachCalls,
    teachAccepted,
    teachQuestions,
    teachAnswers,
    teachComplete,
    teachByTiers,
    teachPerTier,
    teach,
    // autolabel
    catalogue,
    calibration,
    verdicts,
    verdictScores,
  }
}
