"use client"

import * as React from "react"

import { useSettings } from "@/components/settings/settings-provider"
import { toast } from "@/components/ui/toast"
import { useAuth } from "@/hooks/use-auth"
import { useJobs } from "@/hooks/use-jobs"
import { KIND_LABEL, getJob, type Job } from "@/lib/jobs"

/**
 * One subscription to the queue for the whole app.
 *
 * The sidebar shows the queue on every page, and /admin shows it in full — two
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
  useFinishNotices(value.board.active)
  return <JobsContext.Provider value={value}>{children}</JobsContext.Provider>
}

/**
 * Announces jobs as they land, if the setting is on.
 *
 * Watched here because this is the one place subscribed to the board on every
 * page — a training run that finishes while you are reading the corpus is exactly
 * the case worth announcing, and an /admin-only watcher would miss it.
 *
 * A transition is detected rather than a status: the board carries only one page
 * of history, so "is finished" would be true for a whole page forever, and only
 * "was active a moment ago and is not now" happens once.
 *
 * The row is looked up by id rather than found in `history`, because history is
 * paged: a job that finishes while the reader is on page three lands on page one,
 * where this would never see it. One request per finished job, and jobs finish
 * minutes apart.
 */
function useFinishNotices(active: Job[]) {
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
    const current = new Set(active.map((job) => job.id_job))
    wasActive.current = current

    if (!enabledRef.current || previous.size === 0) return

    for (const id of previous) {
      if (current.has(id)) continue
      // A job that left the live band without a terminal row was deleted, and its
      // lookup 404s — which is not worth announcing, so the failure is swallowed.
      void getJob(id).then(announce).catch(noop)
    }
  }, [active])
}

function announce(job: Job) {
  const what = `${KIND_LABEL[job.kind]} · ${job.label}`
  if (job.status === "done") toast.success("Finished", what)
  else if (job.status === "failed") toast.error("Failed", job.error ?? what)
  else if (job.status === "cancelled")
    toast({ title: "Cancelled", description: what })
}

function noop() {}

export function useJobsContext(): JobsContextValue {
  const value = React.useContext(JobsContext)
  if (!value) {
    throw new Error("useJobsContext must be used inside <JobsProvider>")
  }
  return value
}
