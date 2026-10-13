"use client"


import { useJob } from "@/hooks/use-job"
import { JobBenchmarkPanel } from "./job-benchmark-panel"
import { JobHeader } from "./job-header"
import { JobQueuedPanel } from "./job-queued-panel"
import { JobRlPanel } from "./job-rl-panel"
import { JobTokenizePanel } from "./job-tokenize-panel"
import { JobTrainPanel } from "./job-train-panel"

/**
 * One job, in full. The header is the same for every kind; the body is whatever
 * that kind has to show — or, before it starts, the queued panel, which is the
 * same for every kind because waiting is.
 */
export interface JobDetailProps {
  idJob: string
}

export function JobDetail({ idJob }: JobDetailProps) {
  const run = useJob(idJob)
  const { job, isLoading, error } = run

  if (isLoading) return <p className="py-16 text-center text-sm text-muted-foreground">Loading…</p>

  if (!job) {
    return (
      <p className="py-16 text-center text-sm text-muted-foreground">
        {error ?? "This job no longer exists."}
      </p>
    )
  }

  return (
    <>
      <JobHeader job={job} onCancel={run.cancel} onChanged={run.refresh} />
      {/* The panels' vertical rhythm lives here, not inside each kind's component.
          A kind that renders more than one panel — teach renders four — had them
          touching, because a fragment of siblings has no spacing of its own and
          every kind would otherwise have to remember to add some. The header keeps
          its own `mb-8`, so it sits outside this. */}
      <div className="space-y-6">
        {/* Waiting shares this panel with queued: both are "hasn't started", and
            the only difference — what it is waiting on — is one panel inside it. */}
        {job.status === "queued" || job.status === "waiting" ? (
          <JobQueuedPanel job={job} />
        ) : (
          <>
            {job.kind === "train" && <JobTrainPanel job={job} run={run} />}
            {job.kind === "tokenize" && <JobTokenizePanel run={run} />}
            {job.kind === "rl" && <JobRlPanel run={run} />}
            {job.kind === "benchmark" && (
              <JobBenchmarkPanel job={job} run={run} />
            )}

          </>
        )}
      </div>
    </>
  )
}
