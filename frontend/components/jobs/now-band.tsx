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

export interface NowBandProps {
  job: Job | null
  onCancel: (id: string) => void
}

/** Shows the active execution's measured progress and elapsed time, retaining cancel until stopping begins. */
export function NowBand({ job, onCancel }: NowBandProps) {
  const [now, setNow] = React.useState(() => Date.now())
  React.useEffect(() => {
    if (!job) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [job])

  if (!job) {
    return null
  }

  const elapsed = elapsedSeconds(job, now)
  const percent = Math.round(Math.min(1, Math.max(0, job.progress)) * 100)
  const stopping = job.status === "cancelling"

  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-4 py-5">
        <Throbber mode={throbberMode(job)} />

        <div className="min-w-0 flex-1">
          <Link
            href={jobHref(job)}
            className="block font-display text-xl break-words hover:underline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring"
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
            <div className="figure text-lg leading-none">{percent}%</div>
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
