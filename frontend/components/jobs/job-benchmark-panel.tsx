"use client"

import * as React from "react"
import Link from "next/link"

import { ConfigRecap } from "@/components/jobs/config-recap"
import { Panel } from "@/components/ui/panel"
import type { useJob } from "@/hooks/use-job"
import {
  formatNumber,
  getRun,
  type BenchmarkRunDetail,
} from "@/lib/benchmark"
import { isRunning, type Job } from "@/lib/jobs"

export interface JobBenchmarkPanelProps {
  job: Job
  run: ReturnType<typeof useJob>
}

export function JobBenchmarkPanel({ job, run }: JobBenchmarkPanelProps) {
  const [detail, setDetail] = React.useState<BenchmarkRunDetail | null>(null)

  // The score lives on the benchmark run the job produced, not on the job row —
  // jobs record scheduling, the run records results.
  React.useEffect(() => {
    if (job.status !== "done" || !job.output.id) return
    let cancelled = false
    getRun(job.output.id)
      .then((d) => {
        if (!cancelled) setDetail(d)
      })
      .catch(() => {})
    return () => {
      cancelled = true
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
            Measuring held-out text and generated samples. Results appear when the measurement completes.
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
    <div className="grid gap-6 lg:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]">
      <ConfigRecap job={job} />

      <Panel
        variant="open"
        title="Score"
        description={detail?.model_name ?? "Benchmark"}
      >
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
          <span className="figure text-5xl leading-none">
            {formatNumber(detail?.score ?? null, 1)}
          </span>
          <span className="text-sm text-muted-foreground">/ 100</span>
        </div>
        {job.output.id && (
          <Link
            href="/benchmark"
            className="mt-6 inline-block text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
          >
            See the full report and how it compares →
          </Link>
        )}
      </Panel>
    </div>
  )
}
