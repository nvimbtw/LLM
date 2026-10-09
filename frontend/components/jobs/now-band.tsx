"use client"

import * as React from "react"
import Link from "next/link"

import { Button } from "@/components/ui/button"
import { Throbber } from "@/components/ui/throbber"
import {
  KIND_LABEL,
  elapsedSeconds,
  formatElapsed,
  jobHref,
  throbberMode,
  type Job,
} from "@/lib/jobs"
import { cn } from "@/lib/utils"

/**
 * What the GPU is doing right now, pinned above the queue.
 *
 * This is the one place the throbber's per-kind modes earn their keep: the mark
 * rises as one for training, takes readings for a benchmark, hands a peak along
 * for a chat turn. Because a single instance is given a changing `mode`, it
 * travels between those behaviours as the queue advances instead of cutting —
 * which is the difference between a status light and something that reads as one
 * machine moving from one job to the next.
 *
 * Progress is a hairline rule rather than a filled bar. At this weight the page
 * still reads as a ledger, and the bar isn't competing with the numbers beside it.
 */
export interface NowBandProps {
  job: Job | null
  onCancel: (id: string) => void
}

export function NowBand({ job, onCancel }: NowBandProps) {
  // Ticks only while something is running, so an idle page isn't re-rendering
  // once a second forever.
  const [now, setNow] = React.useState(() => Date.now())
  React.useEffect(() => {
    if (!job) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [job])

  if (!job) {
    return <div className="border-b border-border/60" />
  }

  const elapsed = elapsedSeconds(job, now)
  const percent = Math.round(Math.min(1, Math.max(0, job.progress)) * 100)
  const stopping = job.status === "cancelling"

  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-5 py-6">
        {/* The one place the mark keeps its per-kind mode. This band is the queue
            page's subject — what the GPU is doing right now — so it is worth a
            figure rather than a dot, and it is the single instance whose `mode`
            changes as the queue advances, which is what lets it morph between
            training, benchmarking and generating instead of cutting. */}
        <Throbber mode={throbberMode(job)} />

        <div className="min-w-0 flex-1">
          <Link
            href={jobHref(job)}
            className="block truncate font-display text-2xl hover:underline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring sm:text-3xl"
          >
            {job.label}
          </Link>
          <div className="mt-2 flex flex-wrap items-baseline gap-x-2 gap-y-1 text-xs text-muted-foreground">
            <span>{KIND_LABEL[job.kind]}</span>
            <span aria-hidden>·</span>
            <p className="min-w-0 break-words">
              {stopping
                ? "Stopping at the next step boundary…"
                : (job.stage ?? "Starting…")}
            </p>
          </div>
        </div>

        <div className="flex w-full items-center justify-end gap-5 sm:w-auto">
          <div className="text-right">
            <div className="figure text-2xl leading-none">{percent}%</div>
            {elapsed !== null && (
              <div className="mt-1 text-xs text-muted-foreground tabular-nums">
                {formatElapsed(elapsed)}
              </div>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            disabled={stopping}
            onClick={() => onCancel(job.id_job)}
          >
            {stopping ? "Stopping…" : "Cancel"}
          </Button>
        </div>
      </div>

      {/* The rule sits on the band's own bottom edge, so it reads as the band
          filling rather than as a separate widget parked inside it. */}
      <div
        role="progressbar"
        aria-label={`Progress for ${job.label}`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        className="h-px w-full bg-border/60"
      >
        <div
          className={cn(
            "h-px bg-primary transition-[width] duration-700 ease-out motion-reduce:transition-none",
            stopping && "bg-muted-foreground/50"
          )}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  )
}
