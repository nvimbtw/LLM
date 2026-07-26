"use client"

import * as React from "react"

import { useAuth } from "@/hooks/use-auth"
import { useJobs } from "@/hooks/use-jobs"

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
  return <JobsContext.Provider value={value}>{children}</JobsContext.Provider>
}

export function useJobsContext(): JobsContextValue {
  const value = React.useContext(JobsContext)
  if (!value) {
    throw new Error("useJobsContext must be used inside <JobsProvider>")
  }
  return value
}
