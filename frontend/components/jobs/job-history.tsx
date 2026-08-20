"use client"

import Link from "next/link"
import { MoreHorizontal, RotateCcw, Trash2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  KIND_CHIP,
  KIND_LABEL,
  STATUS_LABEL,
  durationSeconds,
  formatClock,
  formatDayGroup,
  formatElapsed,
  jobHref,
  type Job,
} from "@/lib/jobs"
import { cn } from "@/lib/utils"

/**
 * Everything that has finished, newest first, grouped by the day it was queued.
 *
 * Grouped rather than flat because the question asked of this list is almost
 * always "what did I run today" or "what did I run before the weekend" — a date on
 * every row answers that by making you read every row, a heading answers it by
 * position. Times stay on the rows for ordering within a day.
 */
export interface JobHistoryProps {
  jobs: Job[]
  onRetry: (job: Job) => void
  onDelete: (id: string) => void
}

export function JobHistory({ jobs, onRetry, onDelete }: JobHistoryProps) {
  if (jobs.length === 0) {
    return (
      <p className="px-5 py-10 text-center text-sm text-muted-foreground">
        Nothing has run yet.
      </p>
    )
  }

  // Preserves the server's newest-first order: keys are inserted in the order
  // they're first seen, and a Map iterates in insertion order.
  const groups = new Map<string, Job[]>()
  for (const job of jobs) {
    const day = formatDayGroup(job.created_at)
    const bucket = groups.get(day)
    if (bucket) bucket.push(job)
    else groups.set(day, [job])
  }

  return (
    <div>
      {[...groups].map(([day, dayJobs]) => (
        <section key={day}>
          <h3 className="eyebrow border-b border-border/40 bg-secondary/20 px-5 py-1.5 text-muted-foreground">
            {day}
          </h3>
          <ul>
            {dayJobs.map((job) => (
              <HistoryRow
                key={job.id_job}
                job={job}
                onRetry={onRetry}
                onDelete={onDelete}
              />
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}

function HistoryRow({
  job,
  onRetry,
  onDelete,
}: {
  job: Job
  onRetry: (job: Job) => void
  onDelete: (id: string) => void
}) {
  const ran = durationSeconds(job)

  return (
    <li className="group border-b border-border/40 last:border-0">
      <div className="flex items-center gap-3 px-5 py-3">
        <span className="w-14 shrink-0 text-xs text-muted-foreground tabular-nums">
          {formatClock(job.created_at)}
        </span>

        <div className="min-w-0 flex-1">
          <Link
            href={jobHref(job)}
            className="block truncate text-sm hover:underline"
          >
            {job.label}
          </Link>
          {job.error && (
            <span className="mt-0.5 block truncate text-xs text-destructive/70">
              {job.error}
            </span>
          )}
        </div>

        <span
          className={cn(
            "shrink-0 rounded-(--radius-marker) px-1.5 py-0.5 text-xs font-medium tracking-[0.06em]",
            KIND_CHIP[job.kind]
          )}
        >
          {KIND_LABEL[job.kind]}
        </span>

        <span className="hidden w-16 shrink-0 text-right text-xs text-muted-foreground tabular-nums sm:block">
          {ran === null ? "" : formatElapsed(ran)}
        </span>

        {/* Status carries the row's whole verdict, so it is the one place colour
            is spent — and only on the outcomes that aren't the expected one. */}
        <span
          className={cn(
            "w-20 shrink-0 text-right text-xs",
            job.status === "done" && "text-muted-foreground",
            job.status === "failed" && "text-destructive/70",
            job.status === "cancelled" && "text-muted-foreground"
          )}
        >
          {STATUS_LABEL[job.status]}
        </span>

        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Options for ${job.label}`}
                className="shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100 data-[popup-open]:opacity-100"
              >
                <MoreHorizontal />
              </Button>
            }
          />
          <DropdownMenuContent align="end">
            {/* A chat turn's reply already exists; re-running it would append a
                second answer to the same question. */}
            <DropdownMenuItem
              disabled={job.kind === "chat"}
              onClick={() => onRetry(job)}
            >
              <RotateCcw />
              Run again
            </DropdownMenuItem>
            <DropdownMenuItem
              variant="destructive"
              onClick={() => onDelete(job.id_job)}
            >
              <Trash2 />
              Remove
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </li>
  )
}
