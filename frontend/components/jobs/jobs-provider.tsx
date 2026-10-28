"use client"

import * as React from "react"

import { useSettings } from "@/components/settings/settings-provider"
import { toast } from "@/components/ui/toast"
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import { useJobs } from "@/hooks/use-jobs"
import {
  KIND_LABEL,
  getJob,
  jobStatusLabel,
  validationReport,
  type Job,
} from "@/lib/jobs"
import { getToken } from "@/lib/api"

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

export interface JobsProviderProps {
  children: React.ReactNode
}
interface JobsSessionProps extends JobsProviderProps {
  enabled: boolean
  token: string | null
}

/** Replaces queue state with the actual bearer/access identity so private validation rows never survive session replacement. */
export function JobsProvider({ children }: JobsProviderProps) {
  const { isAdmin } = useAuth()
  const token = useSessionToken()
  return (
    <JobsSession
      key={`${token ?? "anonymous"}:${isAdmin}`}
      enabled={Boolean(token && isAdmin)}
      token={token}
    >
      {children}
    </JobsSession>
  )
}

/** Owns one session's queue subscription and best-effort completion notices without sharing their lifetime with a replacement session. */
function JobsSession({ children, enabled, token }: JobsSessionProps) {
  const value = useJobs(enabled)
  useFinishNotices(value.board.active, token)
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
function useFinishNotices(active: Job[], token: string | null) {
  const { settings } = useSettings()
  const enabled = settings.notifyOnFinish

  const wasActive = React.useRef<Set<string>>(new Set())
  const lifetime = React.useRef({ active: false })
  React.useEffect(() => {
    const owner = { active: true }
    lifetime.current = owner
    return () => {
      owner.active = false
    }
  }, [])
  const enabledRef = React.useRef(enabled)
  React.useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  React.useEffect(() => {
    const previous = wasActive.current
    const current = new Set(active.map((job) => job.id_job))
    wasActive.current = current

    const owner = lifetime.current
    if (
      !enabledRef.current ||
      !owner.active ||
      !token ||
      getToken() !== token ||
      previous.size === 0
    )
      return

    for (const id of previous) {
      if (current.has(id)) continue
      void getJob(id)
        .then((job) => {
          if (owner.active && enabledRef.current && getToken() === token)
            announce(job)
        })
        .catch(noop)
    }
  }, [active, token])
}

/** Announces the actual completed verdict or lifecycle outcome only after the session owner approves delivery. */
function announce(job: Job) {
  const what = `${KIND_LABEL[job.kind]} · ${job.label}`
  if (job.status === "done" && validationReport(job))
    toast({ title: jobStatusLabel(job), description: what })
  else if (job.status === "done") toast.success("Finished", what)
  else if (job.status === "failed") toast.error("Failed", job.error ?? what)
  else if (job.status === "cancelled")
    toast({ title: "Cancelled", description: what })
}

/** Ignore best-effort notice lookup failures; the current queue remains the authoritative visible status. */
function noop() {}

/** Returns this mounted queue session, refusing consumers outside its explicit provider. */
export function useJobsContext(): JobsContextValue {
  const value = React.useContext(JobsContext)
  if (!value) {
    throw new Error("useJobsContext must be used inside <JobsProvider>")
  }
  return value
}
