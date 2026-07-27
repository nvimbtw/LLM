"use client"

import * as React from "react"
import Link from "next/link"

import { Panel } from "@/components/ui/panel"
import type { useJob } from "@/hooks/use-job"
import { formatNumber, getRun, type BenchmarkRunDetail } from "@/lib/benchmark"
import { isRunning, type Job } from "@/lib/jobs"

/**
 * A benchmark job's body.
 *
 * While it runs there is one number worth showing — how far along it is — so the
 * panel is a stage line and a rule, not a dashboard of empty slots. Once it
 * finishes the score is the answer, and the full report lives on the benchmark
 * page, which already renders it well; duplicating that here would mean two
 * places to keep in step for no new reading.
 */
export interface JobBenchmarkPanelProps {
  job: Job
  run: ReturnType<typeof useJob>
}

export function JobBenchmarkPanel({ job, run }: JobBenchmarkPanelProps) {
  const [detail, setDetail] = React.useState<BenchmarkRunDetail | null>(null)

  // The score lives on the benchmark run the job produced, not on the job row —
  // jobs record scheduling, the run records results.
  React.useEffect(() => {
    if (job.status !== "done" || !job.ref_id) return
    let cancelled = false
    getRun(job.ref_id)
      .then((d) => {
        if (!cancelled) setDetail(d)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [job.status, job.ref_id])

  if (isRunning(job)) {
    const fraction = run.fraction ?? job.progress
    const percent = Math.round(Math.min(1, Math.max(0, fraction)) * 100)
    return (
      <Panel title="Running" description={`${percent}%`}>
        <p className="text-sm text-muted-foreground">
          {run.stage ?? job.stage ?? "Loading model…"}
        </p>
        <div className="mt-4 h-px w-full bg-border/60">
          <div
            className="h-px bg-primary transition-[width] duration-700 ease-out"
            style={{ width: `${percent}%` }}
          />
        </div>
      </Panel>
    )
  }

  if (job.status !== "done") {
    return (
      <Panel title="No score" description={job.status}>
        <p className="text-sm text-muted-foreground">
          This run did not finish, so it produced no score.
        </p>
      </Panel>
    )
  }

  return (
    <Panel title="Score" description={detail?.model_name ?? "Benchmark"}>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
        <span className="font-display text-5xl leading-none font-semibold tabular-nums">
          {formatNumber(detail?.score ?? null, 1)}
        </span>
        <span className="text-sm text-muted-foreground">/ 100</span>
      </div>
      {job.ref_id && (
        <Link
          href="/benchmark"
          className="mt-6 inline-block text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
        >
          See the full report and how it compares →
        </Link>
      )}
    </Panel>
  )
}
