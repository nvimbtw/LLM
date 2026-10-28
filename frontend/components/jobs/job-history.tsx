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
  KIND_LABEL,
  jobStatusLabel,
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
      <p className="border-y border-border/60 py-8 text-sm text-muted-foreground">
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
          <h3 className="border-y border-border/60 py-2 eyebrow text-muted-foreground">
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

/** Presents one finished lifecycle/verdict and its explicit retry/delete actions without treating invalid validation as successful evidence. */
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
      <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-x-4 gap-y-2 py-4 sm:flex">
        <span className="col-start-1 row-start-2 w-14 shrink-0 text-xs text-muted-foreground tabular-nums">
          {formatClock(job.created_at)}
        </span>

        <div className="col-start-1 row-start-1 min-w-0 flex-1">
          <Link
            href={jobHref(job)}
            className="block truncate text-sm hover:underline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring"
          >
            {job.label}
          </Link>
          {job.error && (
            <span className="mt-0.5 block truncate text-xs text-destructive/70">
              {job.error}
            </span>
          )}
        </div>

        <span className="col-start-2 row-start-2 shrink-0 text-right text-xs text-muted-foreground sm:w-20 sm:text-left">
          {KIND_LABEL[job.kind]}
        </span>

        <span className="hidden w-16 shrink-0 text-right text-xs text-muted-foreground tabular-nums sm:block">
          {ran === null ? "" : formatElapsed(ran)}
        </span>

        <span
          className={cn(
            "col-start-2 row-start-1 w-20 shrink-0 text-right text-xs",
            job.status === "done" && "text-muted-foreground",
            job.status === "failed" && "text-destructive/70",
            job.status === "cancelled" && "text-muted-foreground"
          )}
        >
          {jobStatusLabel(job)}
        </span>

        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Options for ${job.label}`}
                className="col-start-3 row-span-2 row-start-1 shrink-0 text-muted-foreground transition-opacity focus-visible:opacity-100 data-[popup-open]:opacity-100 sm:opacity-0 sm:group-hover:opacity-100"
              >
                <MoreHorizontal />
              </Button>
            }
          />
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => onRetry(job)}>
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
