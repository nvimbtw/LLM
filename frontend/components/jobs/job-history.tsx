"use client"

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
import {
  ResourceList,
  ResourceRow,
  ResourceMetadata,
  ResourceLink,
} from "@/components/ui/resource-list"

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

/** Groups the supplied bounded history rows by their recorded queue date, preserving server order. */
export function JobHistory({ jobs, onRetry, onDelete }: JobHistoryProps) {
  if (jobs.length === 0) {
    return (
      <p className="py-8 text-sm text-muted-foreground">Nothing has run yet.</p>
    )
  }

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
          <h3 className="py-2 eyebrow text-muted-foreground">{day}</h3>
          <ResourceList density="compact">
            {dayJobs.map((job) => (
              <HistoryRow
                key={job.id_job}
                job={job}
                onRetry={onRetry}
                onDelete={onDelete}
              />
            ))}
          </ResourceList>
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
    <ResourceRow density="compact" className="flex items-start gap-3">
      <div className="min-w-0 flex-1">
        <ResourceLink href={jobHref(job)} className="text-sm">
          {job.label}
        </ResourceLink>
        {job.error && (
          <p className="mt-1 text-xs break-words text-destructive">
            {job.error}
          </p>
        )}
        <ResourceMetadata className="mt-1 tabular-nums">
          <span>{formatClock(job.created_at)}</span>
          <span>{KIND_LABEL[job.kind]}</span>
          <span>
            {ran === null ? "Duration unavailable" : formatElapsed(ran)}
          </span>
          <span
            className={job.status === "failed" ? "text-destructive" : undefined}
          >
            {jobStatusLabel(job)}
          </span>
        </ResourceMetadata>
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={`Options for ${job.label}`}
              className="shrink-0 text-muted-foreground"
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
    </ResourceRow>
  )
}
