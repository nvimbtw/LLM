"use client"

import { useJob } from "@/hooks/use-job"
import { JobBenchmarkPanel } from "./job-benchmark-panel"
import { JobCataloguePanel } from "./job-catalogue-panel"
import { JobChatPanel } from "./job-chat-panel"
import { JobGatherPanel } from "./job-gather-panel"
import { JobHeader } from "./job-header"
import { JobSynthPanel } from "./job-synth-panel"
import { JobTrainPanel } from "./job-train-panel"

/**
 * One job, in full. The header is the same for every kind; the body is whatever
 * that kind has to show.
 */
export interface JobDetailProps {
  idJob: string
}

export function JobDetail({ idJob }: JobDetailProps) {
  const run = useJob(idJob)
  const { job, isLoading, error } = run

  if (isLoading) {
    return (
      <p className="py-16 text-center text-sm text-muted-foreground">
        Loading…
      </p>
    )
  }

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
      {job.kind === "train" && <JobTrainPanel job={job} run={run} />}
      {job.kind === "benchmark" && <JobBenchmarkPanel job={job} run={run} />}
      {job.kind === "chat" && <JobChatPanel job={job} run={run} />}
      {job.kind === "gather" && <JobGatherPanel job={job} run={run} />}
      {job.kind === "autolabel" && <JobCataloguePanel job={job} run={run} />}
      {job.kind === "synth" && <JobSynthPanel job={job} run={run} />}
    </>
  )
}
