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
  STATUS_LABEL,
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
  const [actionError, setActionError] = React.useState<string | null>(null)

  /** Starts recovery from the latest artifact and opens the new job, preserving actionable failures. */
  const recover = async () => {
    setBusy(true)
    setActionError(null)
    try {
      const next = await recoverJob(job.id_job)
      router.push(jobHref(next))
    } catch (error) {
      setActionError(
        error instanceof Error ? error.message : "Recovery could not start."
      )
    } finally {
      setBusy(false)
    }
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
    setBusy(true)
    try {
      const next = await retryJob(job)
      router.push(jobHref(next))
    } finally {
      setBusy(false)
    }
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
              {STATUS_LABEL[job.status]}
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
                    disabled={busy}
                    onClick={recover}
                  >
                    Recover latest
                  </Button>
                )}
              <Button
                size="sm"
                disabled={busy}
                onClick={retry}
                className="gap-1.5"
              >
                <RotateCcw className="size-3.5" />
                Run again
              </Button>
              <Button
                variant="ghost"
                size="sm"
                disabled={busy}
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
