"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { useJob } from "@/hooks/use-job"
import { chatId } from "@/lib/jobs"
import { JobBenchmarkPanel } from "./job-benchmark-panel"
import { JobCataloguePanel } from "./job-catalogue-panel"
import { JobChatPanel } from "./job-chat-panel"
import { JobGatherPanel } from "./job-gather-panel"
import { JobHeader } from "./job-header"
import { JobQueuedPanel } from "./job-queued-panel"
import { JobSynthPanel } from "./job-synth-panel"
import { JobTeachPanel } from "./job-teach-panel"
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
  const router = useRouter()
  const run = useJob(idJob)
  const { job, isLoading, error } = run

  // A chat turn is queued from a conversation, never from the jobs menu, and the
  // conversation is where its reply belongs — so its job page is a redirect rather
  // than a place. `replace`, not `push`: this page is a waypoint, and leaving it in
  // history would make back bounce straight through it again.
  const chat = job ? chatId(job) : null
  React.useEffect(() => {
    if (chat) router.replace(`/chat/${chat}`)
  }, [chat, router])

  if (isLoading || chat) {
    return (
      <p className="py-16 text-center text-sm text-muted-foreground">
        {chat ? "Opening the conversation…" : "Loading…"}
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
            {job.kind === "benchmark" && (
              <JobBenchmarkPanel job={job} run={run} />
            )}
            {/* Only reachable for an older chat job whose config names no
                conversation; anything current redirected above. */}
            {job.kind === "chat" && <JobChatPanel job={job} run={run} />}
            {job.kind === "gather" && <JobGatherPanel job={job} run={run} />}
            {job.kind === "autolabel" && (
              <JobCataloguePanel job={job} run={run} />
            )}
            {job.kind === "synth" && <JobSynthPanel job={job} run={run} />}
            {job.kind === "teach" && <JobTeachPanel job={job} run={run} />}
          </>
        )}
      </div>
    </>
  )
}
