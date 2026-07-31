"use client"

import * as React from "react"
import Link from "next/link"

import { ConfigRecap } from "@/components/jobs/config-recap"
import { Panel } from "@/components/ui/panel"
import type { useJob } from "@/hooks/use-job"
import {
  benchmarkSections,
  formatNumber,
  getRun,
  sectionIndex,
  type BenchmarkRunDetail,
} from "@/lib/benchmark"
import { benchmarkSuiteOf, isRunning, type Job } from "@/lib/jobs"
import { cn } from "@/lib/utils"

/**
 * A benchmark job's body.
 *
 * While it runs, the suite's own running order is the figure: five sections, each
 * carrying a fixed share of the composite, worked through in order. A percentage
 * on its own could not say whether the number you are waiting for is minutes or an
 * hour away — grammar alone is 40% of the score and most of the wall clock — and
 * the section list can.
 *
 * Once it finishes the score is the answer, and the full report lives on the
 * benchmark page, which already renders it well; duplicating that here would mean
 * two places to keep in step for no new reading.
 */
export interface JobBenchmarkPanelProps {
  job: Job
  run: ReturnType<typeof useJob>
}

export function JobBenchmarkPanel({ job, run }: JobBenchmarkPanelProps) {
  const [detail, setDetail] = React.useState<BenchmarkRunDetail | null>(null)

  // Which suite this job is running. Stored in the job's config; absent on rows
  // queued before the split, which all meant the language suite.
  const suite = benchmarkSuiteOf(job)

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
    const fraction = Math.min(1, Math.max(0, run.fraction ?? job.progress))
    const percent = Math.round(fraction * 100)
    const index = sectionIndex(fraction, suite)

    return (
      <div className="grid gap-6 lg:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]">
        <ConfigRecap job={job} />

        <Panel title="Scoring" description={`${percent}%`}>
          <p className="text-sm text-muted-foreground">
            {run.stage ?? job.stage ?? "Loading the model…"}
          </p>
          <div className="mt-4 h-px w-full bg-border/60">
            <div
              className="h-px bg-primary transition-[width] duration-700 ease-out"
              style={{ width: `${percent}%` }}
            />
          </div>

          {/* The sections are listed whole, with the finished ones struck out of
              the foreground rather than removed: what is left to do is the
              question, and it only reads as "left" beside what is done. */}
          <ol className="mt-6 space-y-3 border-t border-border pt-5">
            {benchmarkSections(suite).map((section, i) => (
              <li key={section.name} className="flex items-baseline gap-3">
                <span
                  className={cn(
                    "w-16 shrink-0 text-xs tabular-nums",
                    i === index ? "text-foreground" : "text-muted-foreground/60"
                  )}
                >
                  {Math.round(section.weight * 100)}%
                </span>
                <span className="min-w-0">
                  <span
                    className={cn(
                      "block text-sm",
                      i < index && "text-muted-foreground line-through",
                      i > index && "text-muted-foreground"
                    )}
                  >
                    {section.name}
                  </span>
                  {i === index && (
                    <span className="mt-0.5 block text-xs text-muted-foreground">
                      {section.blurb}
                    </span>
                  )}
                </span>
              </li>
            ))}
          </ol>
        </Panel>
      </div>
    )
  }

  if (job.status !== "done") {
    return (
      <div className="grid gap-6 lg:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]">
        <ConfigRecap job={job} />
        <Panel title="No score" description={job.status}>
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
    </div>
  )
}
