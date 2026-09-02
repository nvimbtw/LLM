"use client"

import * as React from "react"

import {
  getGymEpisode,
  getGymRun,
  type GymEpisode,
  type GymRunDetail,
  type GymStep,
} from "@/lib/gyms"
import { streamGymRun, type GymSampling } from "@/lib/gym-socket"
import type { ToolAnswer } from "@/components/chat/trace"

/**
 * One run, live or as it was recorded.
 *
 * A run is watched over a socket while it is happening and read back from the
 * database afterwards, and the *same* shape has to come out either way — a page
 * that rendered a live episode differently from a reloaded one would be two
 * views nobody keeps honest. So a transcript is built here, from frames while
 * they arrive and from stored steps on a reload, and the page below only ever
 * sees `Transcript`.
 */
export interface Transcript {
  /** The reply of every round, in order. The last one is the answer. */
  rounds: string[]
  /** What each command came back with, in the order the calls closed. */
  answers: ToolAnswer[]
  /** Seconds spent generating, summed across the rounds. */
  elapsedSecs: number
  /** True while this episode is the one being generated. */
  streaming: boolean
}

const EMPTY: Transcript = { rounds: [], answers: [], elapsedSecs: 0, streaming: false }

export function useGymRun(runId: string) {
  const [run, setRun] = React.useState<GymRunDetail | null>(null)
  const [transcripts, setTranscripts] = React.useState<Record<string, Transcript>>({})
  const [live, setLive] = React.useState<string | null>(null)
  const [isRunning, setIsRunning] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const cancelRef = React.useRef<(() => void) | null>(null)
  // Episode ids this hook has already fetched. Never cleared when the run
  // changes, and it does not need to be: an episode id belongs to one run, so
  // an id carried over from the previous one can never match.
  const loaded = React.useRef(new Set<string>())

  // Reset during render rather than in an effect when the run changes, the way
  // every other view here does it: an effect would render one frame of the
  // previous run's transcripts under the new run's id.
  const [loadedFor, setLoadedFor] = React.useState(runId)
  if (loadedFor !== runId) {
    setLoadedFor(runId)
    setRun(null)
    setTranscripts({})
    setLive(null)
  }

  React.useEffect(() => {
    let active = true
    getGymRun(runId)
      .then((detail) => active && setRun(detail))
      .catch((e) => active && setError(message(e)))
    return () => {
      active = false
    }
  }, [runId])

  /**
   * Load one episode's steps, once, and rebuild its transcript from them.
   *
   * A run listing carries no steps — a batch of thirty episodes with their
   * command output in it would be a page-load's worth of text nobody reads —
   * so the trajectory arrives when an episode is actually opened. An episode
   * that ran in this session already has a live transcript and is left alone:
   * what is in memory is what was streamed, which is the same thing and
   * already there.
   */
  const open = React.useCallback(
    (episodeId: string) => {
      if (loaded.current.has(episodeId)) return
      loaded.current.add(episodeId)
      getGymEpisode(episodeId)
        .then((episode) => {
          setRun((current) => current && replace(current, episode))
          setTranscripts((all) =>
            all[episodeId] ? all : { ...all, [episodeId]: stored(episode) }
          )
        })
        .catch(() => loaded.current.delete(episodeId))
    },
    []
  )

  // Closing the socket abandons the run, so a page that navigates away must do
  // it deliberately rather than leave a rollout running behind a closed tab.
  React.useEffect(() => () => cancelRef.current?.(), [])

  const start = React.useCallback(
    (sampling: GymSampling) => {
      if (isRunning) return
      setError(null)
      setIsRunning(true)

      cancelRef.current = streamGymRun(runId, sampling, {
        onEpisodeStart: ({ episodeId }) => {
          setLive(episodeId)
          setTranscripts((all) => ({
            ...all,
            [episodeId]: { ...EMPTY, rounds: [""], streaming: true },
          }))
        },
        onToken: (episodeId, token) =>
          setTranscripts((all) => amend(all, episodeId, (t) => ({
            ...t,
            rounds: [...t.rounds.slice(0, -1), (t.rounds.at(-1) ?? "") + token],
          }))),
        // The call is shown the moment it closes, with no body yet — `Trace`
        // renders a step with a null body as the one currently running.
        onToolCall: ({ episodeId, tool, argument }) =>
          setTranscripts((all) => amend(all, episodeId, (t) => ({
            ...t,
            answers: [...t.answers, { tool, argument, body: "" }],
          }))),
        onToolResult: ({ episodeId, tool, argument, body }) =>
          setTranscripts((all) => amend(all, episodeId, (t) => ({
            ...t,
            answers: t.answers.map((a, i) =>
              i === t.answers.length - 1 ? { tool, argument, body } : a
            ),
          }))),
        onRoundDone: (episodeId, elapsedSecs) =>
          setTranscripts((all) => amend(all, episodeId, (t) => ({
            ...t,
            elapsedSecs: t.elapsedSecs + (elapsedSecs ?? 0),
            // A round that ends the episode is the last one; the empty string
            // opened here is dropped by `onEpisodeDone` if nothing follows.
            rounds: [...t.rounds, ""],
          }))),
        onEpisodeDone: (verdict) => {
          setLive(null)
          setTranscripts((all) => amend(all, verdict.episodeId, (t) => ({
            ...t,
            streaming: false,
            rounds: t.rounds.filter((r, i) => r !== "" || i === 0),
          })))
          setRun((current) => current && score(current, verdict))
        },
        onRunDone: (totals) => {
          setIsRunning(false)
          setLive(null)
          setRun((current) =>
            current && {
              ...current,
              status: "done",
              score: totals.score,
              correct: totals.correct,
              graded: totals.graded,
            }
          )
        },
        onError: (e) => {
          setIsRunning(false)
          setLive(null)
          setError(e)
        },
      })
    },
    [runId, isRunning]
  )

  const stop = React.useCallback(() => {
    cancelRef.current?.()
    cancelRef.current = null
    setIsRunning(false)
    setLive(null)
  }, [])

  return { run, transcripts, live, isRunning, error, start, stop, open }
}

/** A finished episode, folded back into the run it belongs to. */
function score(
  run: GymRunDetail,
  verdict: { episodeId: string; score: number | null; answer: string | null; groundTruth: string | null }
): GymRunDetail {
  return {
    ...run,
    episodes: run.episodes.map((e) =>
      e.id === verdict.episodeId
        ? {
            ...e,
            status: "graded" as const,
            score: verdict.score,
            answer: verdict.answer,
            ground_truth: verdict.groundTruth ?? undefined,
          }
        : e
    ),
  }
}

/**
 * The transcript a reload rebuilds from stored rows.
 *
 * A stored step carries the reply of the round that made it, so the rounds come
 * back in order and the answers with them. The final reply is on the episode
 * rather than on any step, because it is the round that called nothing.
 *
 * Generation time is not stored, so a reloaded trace says how many commands ran
 * and not how long they took — better than a zero presented as a measurement.
 */
function stored(episode: GymEpisode): Transcript {
  const steps = episode.steps ?? []
  return {
    rounds: [
      ...steps.map((s) => s.reply ?? ""),
      ...(episode.reply ? [episode.reply] : []),
    ],
    answers: steps.map(answerOf),
    elapsedSecs: 0,
    streaming: false,
  }
}

/** One episode's freshly loaded detail, folded back into the run. */
function replace(run: GymRunDetail, episode: GymEpisode): GymRunDetail {
  return {
    ...run,
    episodes: run.episodes.map((e) => (e.id === episode.id ? { ...e, ...episode } : e)),
  }
}

/** A stored step, in the shape the live path produces. The body is rebuilt from
 * the recorded output rather than stored twice — what the model read is that
 * output inside a code block, which is what `Trace` renders. */
function answerOf(step: GymStep): ToolAnswer {
  const output = [step.stdout, step.stderr].filter((s) => s?.trim()).join("\n")
  return {
    tool: "bash",
    argument: step.action,
    body: output
      ? `<|codeblock|>${output}<|/codeblock|>`
      : `<|warn|>The command finished with no output.<|/warn|>`,
  }
}

function amend(
  all: Record<string, Transcript>,
  episodeId: string,
  change: (t: Transcript) => Transcript
): Record<string, Transcript> {
  const current = all[episodeId] ?? { ...EMPTY, rounds: [""] }
  return { ...all, [episodeId]: change(current) }
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : "Something went wrong."
}
