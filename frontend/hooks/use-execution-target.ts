"use client"

import * as React from "react"
import { getToken } from "@/lib/api"
import { resolveExecutionTarget, type ExecutionTarget } from "@/lib/routes"
import type { JobKind } from "@/lib/jobs"

/** Resolves one mounted session's execution route and cancels or ignores obsolete reads on navigation. */
export function useExecutionTarget(token: string, id: string, kind: JobKind) {
  const [target, setTarget] = React.useState<ExecutionTarget | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  React.useEffect(() => {
    let current = true
    const controller = new AbortController()
    resolveExecutionTarget(id, kind, controller.signal)
      .then((next) => {
        if (current && getToken() === token) setTarget(next)
      })
      .catch((cause: unknown) => {
        if (current && getToken() === token)
          setError(
            cause instanceof Error
              ? cause.message
              : "The execution could not be loaded."
          )
      })
    return () => {
      current = false
      controller.abort()
    }
  }, [token, id, kind])
  return { target, error }
}
