"use client"

import Link from "next/link"

import { Panel } from "@/components/ui/panel"
import type { useJob } from "@/hooks/use-job"
import { chatId, isRunning, type Job } from "@/lib/jobs"

/**
 * A chat job's body: the reply, as it arrives.
 *
 * This page is not where you would normally read a chat turn — the conversation
 * is — but a queued turn is a real job that can be waiting behind an hours-long
 * training run, and it would be strange for the queue to list something it refuses
 * to show. It also makes the durability visible: the reply lands here whether or
 * not the tab that asked for it is still open.
 */
export interface JobChatPanelProps {
  job: Job
  run: ReturnType<typeof useJob>
}

export function JobChatPanel({ job, run }: JobChatPanelProps) {
  const id = chatId(job)
  const text = run.text.trim()

  return (
    <Panel
      title="Reply"
      description={isRunning(job) ? "Generating" : undefined}
      bodyClassName="p-0"
    >
      {text ? (
        <p className="whitespace-pre-wrap px-5 py-4 text-sm leading-relaxed">{text}</p>
      ) : (
        <p className="px-5 py-8 text-center text-sm text-muted-foreground">
          {job.status === "queued"
            ? "Waiting for the GPU."
            : // A finished turn shows nothing here after a reload: the tokens were
              // streamed, not stored on the job. The conversation has them.
              "Nothing streamed to this page."}
        </p>
      )}

      {id && (
        <div className="border-t border-border/40 px-5 py-3">
          <Link
            href={`/chat/${id}`}
            className="text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
          >
            Open the conversation →
          </Link>
        </div>
      )}
    </Panel>
  )
}
