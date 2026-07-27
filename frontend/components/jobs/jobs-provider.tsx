"use client"

import * as React from "react"

import { useSettings } from "@/components/settings/settings-provider"
import { toast } from "@/components/ui/toast"
import { useAuth } from "@/hooks/use-auth"
import { useJobs } from "@/hooks/use-jobs"
import { KIND_LABEL, type Job } from "@/lib/jobs"

/**
 * One subscription to the queue for the whole app.
 *
 * The sidebar shows the queue on every page, and /jobs shows it in full — two
 * `useJobs()` calls would mean two sockets and two refetches per change, for the
 * life of the session. Hoisting it here is the same move `ChatsProvider` makes for
 * the conversation list, and for the same reason.
 *
 * Gated on `isAdmin`: the queue endpoints are admin-only, so a normal session
 * would otherwise spend its whole life collecting 403s.
 */
type JobsContextValue = ReturnType<typeof useJobs>

const JobsContext = React.createContext<JobsContextValue | null>(null)

export function JobsProvider({ children }: { children: React.ReactNode }) {
  const { isAdmin } = useAuth()
  const value = useJobs(isAdmin)
  useFinishNotices(value.board.active, value.history)
  return <JobsContext.Provider value={value}>{children}</JobsContext.Provider>
}

/**
 * Announces jobs as they land, if the setting is on.
 *
 * Watched here because this is the one place subscribed to the board on every
 * page — a training run that finishes while you are reading the corpus is exactly
 * the case worth announcing, and a /jobs-only watcher would miss it.
 *
 * A transition is detected rather than a status: the board is refetched whole on
 * every nudge, so "is finished" is true for the entire history forever, and only
 * "was active a moment ago and is not now" happens once.
 */
function useFinishNotices(active: Job[], history: Job[]) {
  const { settings } = useSettings()
  const enabled = settings.notifyOnFinish

  const wasActive = React.useRef<Set<string>>(new Set())
  // Read inside the effect rather than depended upon, so switching the setting on
  // does not replay every job that finished while it was off.
  const enabledRef = React.useRef(enabled)
  React.useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  React.useEffect(() => {
    const previous = wasActive.current
    wasActive.current = new Set(active.map((job) => job.id_job))

    if (!enabledRef.current || previous.size === 0) return

    for (const job of history) {
      if (!previous.has(job.id_job) || wasActive.current.has(job.id_job))
        continue
      const what = `${KIND_LABEL[job.kind]} · ${job.label}`
      if (job.status === "done") toast.success("Finished", what)
      else if (job.status === "failed") toast.error("Failed", job.error ?? what)
      else if (job.status === "cancelled")
        toast({ title: "Cancelled", description: what })
    }
  }, [active, history])
}

export function useJobsContext(): JobsContextValue {
  const value = React.useContext(JobsContext)
  if (!value) {
    throw new Error("useJobsContext must be used inside <JobsProvider>")
  }
  return value
}
