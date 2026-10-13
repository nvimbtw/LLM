"use client"

import * as React from "react"

import { cancelJob, getJob, type Job } from "@/lib/jobs"
import { attachJob } from "@/lib/jobs-socket"
import {
  createJobTelemetry,
  jobTelemetryEvent,
  reduceJobTelemetry,
} from "@/lib/job-telemetry"

/**
 * Coordinates one job's durable REST row and its socket attachment; returns live telemetry as enrichment.
 * Socket failures leave the job running, while terminal frames refresh authoritative status.
 * Resource changes reset telemetry and ignore late reads/frames from the previous job.
 */
export function useJob(idJob: string) {
  const [job, setJob] = React.useState<Job | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [telemetry, setTelemetry] = React.useState(createJobTelemetry)
  const [loadedFor, setLoadedFor] = React.useState(idJob)
  const ownerRef = React.useRef({ idJob, active: true })
  if (loadedFor !== idJob) {
    setLoadedFor(idJob)
    setJob(null)
    setIsLoading(true)
    setError(null)
    setTelemetry(createJobTelemetry())
  }

  /** Fetches this job's authoritative row, ignoring results once its resource owner has detached. */
  const refresh = React.useCallback(async () => {
    const owner = ownerRef.current
    if (owner.idJob !== idJob) return
    try {
      const next = await getJob(idJob)
      if (owner.active) {
        setJob(next)
        setError(null)
      }
    } catch (failure) {
      if (owner.active)
        setError(
          failure instanceof Error ? failure.message : "Failed to load the job."
        )
    } finally {
      if (owner.active) setIsLoading(false)
    }
  }, [idJob])

  React.useEffect(() => {
    const owner = { idJob, active: true }
    ownerRef.current = owner
    let active = true
    getJob(idJob)
      .then((next) => {
        if (active) {
          setJob(next)
          setError(null)
        }
      })
      .catch((failure: unknown) => {
        if (active)
          setError(
            failure instanceof Error
              ? failure.message
              : "Failed to load the job."
          )
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
      owner.active = false
    }
  }, [idJob])

  React.useEffect(() => {
    let active = true
    /** Reduces live telemetry and refreshes terminal status only while this attachment still owns the job view. */
    function onFrame(frame: Record<string, unknown>): void {
      if (!active) return
      const event = jobTelemetryEvent(frame)
      if (event) setTelemetry((previous) => reduceJobTelemetry(previous, event))
      if (
        frame.type === "done" ||
        frame.type === "error" ||
        frame.type === "cancelled"
      )
        void refresh()
    }
    const attachment = attachJob(idJob, { onFrame })
    return () => {
      active = false
      attachment.detach()
    }
  }, [idJob, refresh])

  /** Cancels through REST so queued jobs work too, then refreshes status even when cancellation is refused. */
  const cancel = React.useCallback(async () => {
    const owner = ownerRef.current
    try {
      await cancelJob(idJob)
    } catch (failure) {
      if (owner.active)
        setError(
          failure instanceof Error
            ? failure.message
            : "Failed to cancel the job."
        )
    } finally {
      await refresh()
    }
  }, [idJob, refresh])

  return { job, isLoading, error, refresh, cancel, ...telemetry }
}
