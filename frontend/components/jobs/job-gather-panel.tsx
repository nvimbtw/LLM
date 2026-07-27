"use client"

import * as React from "react"
import Link from "next/link"
import { ThumbsDown, ThumbsUp } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Panel } from "@/components/ui/panel"
import { useJob } from "@/hooks/use-job"
import { useRatingQueue } from "@/hooks/use-rating-queue"
import { useSignals } from "@/hooks/use-signals"
import { isRunning, type Job } from "@/lib/jobs"

/**
 * A gather job's body: one generated reply at a time, with a verdict to give it.
 *
 * This is the half of preference tuning that needs a person. The GPU can generate
 * replies all day and the rule-based judge can score their surface, but whether a
 * reply is *good* is a judgement only someone reading it can make — so the job's
 * page is a rating station rather than a progress bar.
 *
 * Rating starts while the batch is still generating. Each finished reply is a row
 * the moment it exists, and the job broadcasts a nudge, so the queue below fills
 * as the run proceeds instead of making the person wait for the whole batch.
 */
export interface JobGatherPanelProps {
  job: Job
  run: ReturnType<typeof useJob>
}

export function JobGatherPanel({ job, run }: JobGatherPanelProps) {
  const queue = useRatingQueue(job.id_job, run.sampleTick)
  // Re-read the global count each time a verdict lands here, so the bar tracks
  // everything rated rather than just this batch.
  const { status } = useSignals(queue.ratedHere)

  const { approve, reject, skip, current } = queue

  // Keyboard verdicts. Rating is repetitive by nature and reaching for the mouse
  // for every reply is what makes a batch feel like a chore.
  React.useEffect(() => {
    if (!current) return
    const onKey = (e: KeyboardEvent) => {
      // Never steal a keystroke meant for a field elsewhere on the page.
      const target = e.target as HTMLElement | null
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return
      if (e.key === "ArrowUp" || e.key === "j") void approve()
      else if (e.key === "ArrowDown" || e.key === "k") void reject()
      else if (e.key === "s") void skip()
      else return
      e.preventDefault()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [current, approve, reject, skip])

  const ready = status?.ready ?? false

  return (
    <Panel
      title="Rate the replies"
      description={
        isRunning(job) ? "Generating — rate as they arrive" : "One verdict each"
      }
      bodyClassName="p-0"
    >
      {status && (
        <div className="border-b border-border/40 px-5 py-3">
          <SignalBar status={status} />
        </div>
      )}

      {current ? (
        <div className="px-5 py-4">
          <p className="text-xs tracking-wide text-muted-foreground uppercase">
            Prompt
          </p>
          <p className="mt-1 text-sm">{current.prompt}</p>

          <p className="mt-4 text-xs tracking-wide text-muted-foreground uppercase">
            Reply
          </p>
          <p className="mt-1 text-sm leading-relaxed whitespace-pre-wrap">
            {current.response}
          </p>

          <div className="mt-5 flex flex-wrap items-center gap-2">
            <Button onClick={() => void approve()} className="gap-2">
              <ThumbsUp className="size-3.5" />
              Good
            </Button>
            <Button
              variant="outline"
              onClick={() => void reject()}
              className="gap-2"
            >
              <ThumbsDown className="size-3.5" />
              Bad
            </Button>
            <Button variant="ghost" onClick={() => void skip()}>
              Skip
            </Button>
            <span className="ml-auto text-xs text-muted-foreground tabular-nums">
              {queue.remaining} left in this batch
            </span>
          </div>

          {/* A rejected reply is as useful as an approved one — KTO trains on
              both — so say so, because "bad" reads like something to avoid
              recording. */}
          <p className="mt-3 text-xs text-muted-foreground">
            Both verdicts are signal: the tune needs examples to push down as
            much as examples to push up. <Key>↑</Key> good, <Key>↓</Key> bad,{" "}
            <Key>s</Key> skip.
          </p>

          {queue.error && (
            <p className="mt-3 text-xs text-destructive">{queue.error}</p>
          )}
        </div>
      ) : (
        <p className="px-5 py-8 text-center text-sm text-muted-foreground">
          {queue.isLoading
            ? "Loading…"
            : isRunning(job)
              ? "Generating the first reply…"
              : job.status === "queued"
                ? "Waiting for the GPU."
                : "Everything from this batch has been rated."}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-3 border-t border-border/40 px-5 py-3">
        {ready ? (
          <>
            <span className="text-sm">Enough signal to tune.</span>
            <Link
              href="/jobs/new/train"
              className="text-sm underline-offset-4 hover:underline"
            >
              Start the preference run →
            </Link>
          </>
        ) : (
          <span className="text-sm text-muted-foreground">
            {status?.blocked_reason ??
              "Keep rating to unlock the preference run."}
          </span>
        )}
      </div>
    </Panel>
  )
}

/**
 * The count against the bar it has to clear. Both classes are shown separately
 * because a total alone hides the failure that actually blocks a run: plenty of
 * ratings, all of them the same verdict.
 */
export function SignalBar({
  status,
}: {
  status: {
    desirable: number
    undesirable: number
    total: number
    required: number
  }
}) {
  const pct = Math.min(100, (status.total / Math.max(1, status.required)) * 100)
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="text-muted-foreground">Signals</span>
        <span className="tabular-nums">
          {status.total} / {status.required}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-foreground transition-[width] duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-xs text-muted-foreground tabular-nums">
        {status.desirable} good · {status.undesirable} bad
      </p>
    </div>
  )
}

function Key({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded border border-border px-1 py-0.5 font-mono text-[0.7rem]">
      {children}
    </kbd>
  )
}
