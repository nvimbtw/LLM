"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { ValidationObserver } from "@/lib/architecture/validation-observer"
import { validationCommand } from "@/lib/architecture/http"
import {
  getModelValidation,
  getModelValidationRun,
  retryModelValidation,
} from "@/lib/model-validation"
import { attachJob } from "@/lib/jobs-socket"

/** Supplies an empty subscription before this session's observer has initialized. */
const noSubscription = () => () => {}
/** Keeps server/unloaded external-store snapshots stable and free of browser credentials. */
const noObservation = () => null

/** Observes one actual administrator session/run with serialized REST polling and bounded socket enrichment; disposal never cancels durable work. */
export function useValidationJob(id: string) {
  const token = useSessionToken()
  const resource = `${token}:${id}`
  const [holder, setHolder] = React.useState<{
    resource: string
    observer: ValidationObserver
  } | null>(null)
  const [connection, setConnection] = React.useState<{
    resource: string
    error: string
  } | null>(null)
  const [reconnect, retryConnection] = React.useReducer(
    (value: number) => value + 1,
    0
  )
  React.useEffect(() => {
    if (!token) return
    const observer = new ValidationObserver(id, {
      run: (signal) => getModelValidationRun(id, { token, signal }),
      report: (signal) => getModelValidation(id, { token, signal }),
      cancel: (signal) =>
        validationCommand(`/runs/${encodeURIComponent(id)}/cancel`, "POST", {
          token,
          signal,
        }),
      retry: (signal) => retryModelValidation(id, { token, signal }),
      remove: (signal) =>
        validationCommand(`/runs/${encodeURIComponent(id)}`, "DELETE", {
          token,
          signal,
        }),
    })
    let active = true
    void observer.refresh().then(() => {
      if (active) setHolder({ resource, observer })
    })
    return () => {
      active = false
      observer.dispose()
    }
  }, [token, resource, id])
  const observer = holder?.resource === resource ? holder.observer : null
  const state = React.useSyncExternalStore(
    observer?.subscribe ?? noSubscription,
    observer?.getSnapshot ?? noObservation,
    noObservation
  )
  const status = state?.run?.status
  React.useEffect(() => {
    if (
      !observer ||
      !token ||
      (status !== "running" && status !== "cancelling")
    )
      return
    let active = true
    let timer: ReturnType<typeof setTimeout> | null = null
    const attachment = attachJob(
      id,
      {
        onFrame: (frame) => {
          if (!active) return
          if (frame.schema_version === 1) {
            try {
              observer.event(frame)
              setConnection(null)
            } catch (error) {
              setConnection({
                resource,
                error:
                  error instanceof Error
                    ? error.message.slice(0, 2000)
                    : "Invalid validation event.",
              })
              void observer.refresh()
            }
          } else if (
            ["done", "error", "cancelled"].includes(String(frame.type))
          )
            void observer.refresh()
        },
        onInactive: () => {
          if (active) void observer.refresh()
        },
        onError: (error) => {
          if (active) setConnection({ resource, error: error.slice(0, 2000) })
        },
        onClose: () => {
          if (!active) return
          void observer.refresh()
          timer = setTimeout(() => {
            if (active) retryConnection()
          }, 5000)
        },
      },
      token
    )
    return () => {
      active = false
      if (timer !== null) clearTimeout(timer)
      attachment.detach()
    }
  }, [observer, token, resource, id, status, reconnect])
  return {
    observer,
    state,
    loading: !state || state.loading,
    connectionError:
      connection?.resource === resource &&
      (status === "running" || status === "cancelling")
        ? connection.error
        : null,
  }
}
