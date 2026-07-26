"use client"

import * as React from "react"

import {
  listPendingSamples,
  rateSample,
  skipSample,
  type PendingSample,
  type Verdict,
} from "@/lib/signals"

/**
 * The queue of replies waiting for a verdict, one at a time.
 *
 * Rating is a rhythm — read, judge, next — so the queue is held locally and the
 * current sample is dropped from it the moment it is judged, rather than
 * refetching between every click and making the person wait on a round trip to see
 * the next reply. The write still goes to the server; it just isn't in the way.
 *
 * `nudge` is how new samples arrive mid-batch: the gather job broadcasts a frame
 * per generated reply, and the panel bumps this to pull them in. Same
 * nudge-and-refetch shape the jobs board uses — the frame says *that* something
 * changed, the fetch says what.
 */
export function useRatingQueue(idJob?: string, nudge?: unknown) {
  const [queue, setQueue] = React.useState<PendingSample[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  // Verdicts recorded in this session, so the panel can show progress without
  // refetching the global count after every click.
  const [ratedHere, setRatedHere] = React.useState(0)

  // Every sample this session has already acted on. Kept in a ref because the
  // merge below reads it during a state update, and it must be the current set
  // rather than the one captured when `refresh` was created.
  const judgedIds = React.useRef<Set<string>>(new Set())

  const refresh = React.useCallback(async () => {
    try {
      const pending = await listPendingSamples(idJob)
      // Merged rather than replaced: a refetch that lands between a click and its
      // PUT would otherwise resurrect the sample just judged, and it would be
      // rated twice.
      setQueue((prev) => {
        const seen = new Set(prev.map((s) => s.id_sample))
        const judged = judgedIds.current
        const fresh = pending.filter(
          (s) => !seen.has(s.id_sample) && !judged.has(s.id_sample)
        )
        return [...prev, ...fresh]
      })
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load samples to rate.")
    } finally {
      setIsLoading(false)
    }
  }, [idJob])

  React.useEffect(() => {
    void refresh()
  }, [refresh, nudge])

  const current = queue[0] ?? null

  // Removes the head and records the verdict. Optimistic on purpose: a failed
  // write puts it back at the front rather than losing the judgement silently.
  const judge = React.useCallback(
    async (verdict: Verdict | null) => {
      const sample = current
      if (!sample) return
      judgedIds.current.add(sample.id_sample)
      setQueue((prev) => prev.slice(1))
      try {
        if (verdict === null) {
          await skipSample(sample.id_sample)
        } else {
          await rateSample(sample.id_sample, verdict)
          setRatedHere((n) => n + 1)
        }
      } catch (e) {
        judgedIds.current.delete(sample.id_sample)
        setQueue((prev) => [sample, ...prev])
        setError(e instanceof Error ? e.message : "Failed to save the rating.")
      }
    },
    [current]
  )

  return {
    current,
    remaining: queue.length,
    ratedHere,
    isLoading,
    error,
    refresh,
    approve: React.useCallback(() => judge("up"), [judge]),
    reject: React.useCallback(() => judge("down"), [judge]),
    skip: React.useCallback(() => judge(null), [judge]),
  }
}
