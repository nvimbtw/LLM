"use client"

import * as React from "react"

import { ConfigRecap } from "@/components/jobs/config-recap"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import { LoadError } from "@/components/ui/error-state"
import { BenchmarkReport } from "@/components/benchmark/benchmark-report-view"
import type { useJob } from "@/hooks/use-job"
import { getRun, type BenchmarkRunDetail } from "@/lib/benchmark"
import { isRunning, type Job } from "@/lib/jobs"

export interface JobBenchmarkPanelProps {
  job: Job
  run: ReturnType<typeof useJob>
}

/** Keeps queued/running diagnostics and opens the execution's exact output report after completion. */
export function JobBenchmarkPanel({ job, run }: JobBenchmarkPanelProps) {
  const [loaded, setLoaded] = React.useState<{
    id: string
    detail?: BenchmarkRunDetail
    error?: string
  } | null>(null)
  const result = loaded?.id === job.output.id ? loaded : null

  React.useEffect(() => {
    if (job.status !== "done" || !job.output.id) return
    let cancelled = false
    const controller = new AbortController()
    const id = job.output.id
    getRun(id, controller.signal)
      .then((d) => {
        if (!cancelled) setLoaded({ id, detail: d })
      })
      .catch((cause: unknown) => {
        if (!cancelled)
          setLoaded({
            id,
            error:
              cause instanceof Error
                ? cause.message
                : "The report could not be loaded.",
          })
      })
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [job.status, job.output.id])

  if (isRunning(job)) {
    return (
      <div className="grid gap-6 lg:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]">
        <ConfigRecap job={job} />

        <Panel variant="open" title="Scoring" description="In progress">
          <p className="text-sm text-muted-foreground">
            {run.stage ?? job.stage ?? "Measuring English language…"}
          </p>
          <p className="mt-4 text-sm text-muted-foreground">
            Measuring held-out text and generated samples. Results appear when
            the measurement completes.
          </p>
        </Panel>
      </div>
    )
  }

  if (job.status !== "done") {
    return (
      <div className="grid gap-6 lg:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]">
        <ConfigRecap job={job} />
        <Panel variant="open" title="No score" description={job.status}>
          <p className="text-sm text-muted-foreground">
            This run did not finish, so it produced no score.
          </p>
        </Panel>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <ConfigRecap job={job} />
      {result?.error ? (
        <LoadError title={result.error} />
      ) : result?.detail ? (
        <BenchmarkReport run={result.detail} />
      ) : job.output.id ? (
        <Skeleton className="h-48" />
      ) : (
        <p className="text-sm text-muted-foreground">No report recorded.</p>
      )}
    </div>
  )
}
