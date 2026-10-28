"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { RotateCcw, Trash2 } from "lucide-react"

import { PageHeader } from "@/components/ui/page-header"
import { Button } from "@/components/ui/button"
import { RunningDot } from "@/components/ui/running-dot"
import { useParentTrail } from "@/hooks/use-parent-trail"
import {
  jobHref,
  KIND_LABEL,
  jobStatusLabel,
  deleteJob,
  elapsedSeconds,
  formatElapsed,
  isRunning,
  isTerminal,
  retryJob,
  recoverJob,
  type Job,
} from "@/lib/jobs"
import { cn } from "@/lib/utils"
import { FormError } from "@/components/ui/error-state"
import { useJobSubmission } from "@/hooks/use-job-submission"

/**
 * A job's masthead: what it is, where it stands, and the three things you can do
 * to it. Identical for every kind, so the per-kind panel below can be purely about
 * the work rather than repeating the controls three times.
 */
export interface JobHeaderProps {
  job: Job
  onCancel: () => void
  onChanged: () => void
}

/** Composes the shared page header with job provenance, parent navigation and lifecycle actions. */
export function JobHeader({ job, onCancel, onChanged }: JobHeaderProps) {
  const router = useRouter()
  const parent = useParentTrail({ href: "/admin", label: "Admin" })
  const [busy, setBusy] = React.useState(false)
  const submission = useJobSubmission(`job:${job.id_job}`)
  const actionError = submission.state?.error

  /** Starts recovery from the latest artifact and opens the new job, preserving actionable failures. */
  const recover = async () => {
    const next = await submission.owner?.submit(
      `recover:${job.id_job}`,
      (request) => recoverJob(job.id_job, request)
    )
    if (next) router.push(jobHref(next))
  }

  const [now, setNow] = React.useState(() => Date.now())
  React.useEffect(() => {
    if (!isRunning(job)) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [job])

  const elapsed = elapsedSeconds(job, now)
  const stopping = job.status === "cancelling"

  /** Queues the same job configuration again and navigates to the new job. */
  const retry = async () => {
    const next = await submission.owner?.submit(
      `retry:${job.id_job}`,
      (request) => retryJob(job, request)
    )
    if (next) router.push(jobHref(next))
  }

  /** Deletes the job and returns to its resolved parent; failures refresh the current row. */
  const remove = async () => {
    setBusy(true)
    try {
      await deleteJob(job.id_job)
      router.push(parent.href)
    } catch {
      setBusy(false)
      onChanged()
    }
  }

  return (
    <PageHeader
      back={parent}
      contentClassName="min-w-0 flex-1 basis-64"
      title={
        <span className="flex items-center gap-3">
          <span className="min-w-0 break-words">{job.label}</span>
          {isRunning(job) && <RunningDot />}
        </span>
      }
      description={
        <>
          <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
            <span>{KIND_LABEL[job.kind]}</span>
            <span aria-hidden className="text-muted-foreground/30">
              ·
            </span>
            <span
              className={cn(
                job.status === "failed" && "text-destructive/80",
                job.status === "running" && "text-foreground"
              )}
            >
              {jobStatusLabel(job)}
            </span>
            {elapsed !== null && (
              <>
                <span aria-hidden className="text-muted-foreground/30">
                  ·
                </span>
                <span className="tabular-nums">{formatElapsed(elapsed)}</span>
              </>
            )}
          </p>

          {job.error && (
            <FormError className="mt-3 max-w-prose">{job.error}</FormError>
          )}
          {actionError && <FormError className="mt-3">{actionError}</FormError>}
        </>
      }
      action={
        <div className="flex flex-wrap items-center gap-2">
          {isTerminal(job) ? (
            <>
              {(job.kind === "train" || job.kind === "rl") &&
                job.status !== "done" && (
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={busy || submission.pending}
                    onClick={recover}
                  >
                    Recover latest
                  </Button>
                )}
              <Button
                size="sm"
                disabled={busy || submission.pending}
                onClick={retry}
                className="gap-1.5"
              >
                <RotateCcw className="size-3.5" />
                Run again
              </Button>
              <Button
                variant="ghost"
                size="sm"
                disabled={busy || submission.pending}
                onClick={remove}
                className="gap-1.5 text-muted-foreground hover:text-destructive"
              >
                <Trash2 className="size-3.5" />
                Delete
              </Button>
            </>
          ) : (
            <Button
              variant="outline"
              size="sm"
              disabled={stopping}
              onClick={onCancel}
            >
              {stopping ? "Stopping…" : "Cancel"}
            </Button>
          )}
        </div>
      }
    />
  )
}
