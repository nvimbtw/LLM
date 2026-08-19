"use client"

import * as React from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { ArrowLeft, RotateCcw, Trash2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { RunningDot } from "@/components/ui/running-dot"
import { useParentTrail } from "@/hooks/use-parent-trail"
import {
  KIND_LABEL,
  STATUS_LABEL,
  deleteJob,
  elapsedSeconds,
  formatElapsed,
  isRunning,
  isTerminal,
  retryJob,
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

export function JobHeader({ job, onCancel, onChanged }: JobHeaderProps) {
  const router = useRouter()
  const parent = useParentTrail({ href: "/admin", label: "Admin" })
  const [busy, setBusy] = React.useState(false)

  const [now, setNow] = React.useState(() => Date.now())
  React.useEffect(() => {
    if (!isRunning(job)) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [job])

  const elapsed = elapsedSeconds(job, now)
  const stopping = job.status === "cancelling"

  const retry = async () => {
    setBusy(true)
    try {
      const next = await retryJob(job)
      // Straight to the new job: the point of re-running is to watch the new one,
      // and leaving you on the finished original makes you find it yourself.
      router.push(`/jobs/${next.id_job}`)
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    setBusy(true)
    try {
      await deleteJob(job.id_job)
      // The deleted job's page is gone, so leaving goes wherever back would have.
      router.push(parent.href)
    } catch {
      setBusy(false)
      onChanged()
    }
  }

  return (
    <header className="mb-8 border-b border-border pb-6">
      {/* A job is reachable from the queue and from the cycle that queued it, so
          back follows whoever linked here rather than always the queue. */}
      <Link
        href={parent.href}
        className="mb-4 inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
      >
        <ArrowLeft className="size-3" />
        {parent.label}
      </Link>

      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <h1 className="truncate font-display text-3xl font-semibold tracking-tight">
              {job.label}
            </h1>
            {isRunning(job) && <RunningDot />}
          </div>

          {/* One line of provenance: kind, state, and how long it took. Dot
              separators rather than a table — three short facts don't need one. */}
          <p className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
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
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {isTerminal(job) ? (
            <>
              {/* Most runs are a tweak of the last one, so re-running is the
                  primary action on anything finished. A chat turn is the
                  exception: its answer already exists. */}
              {job.kind !== "chat" && (
                <Button
                  size="sm"
                  disabled={busy}
                  onClick={retry}
                  className="gap-1.5"
                >
                  <RotateCcw className="size-3.5" />
                  Run again
                </Button>
              )}
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
      </div>
    </header>
  )
}
