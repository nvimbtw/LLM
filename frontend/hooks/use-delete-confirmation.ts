"use client"

import * as React from "react"
import { getToken } from "@/lib/api"
import { useSessionToken } from "./use-auth"

/** Owns one confirmation's pending delete and actionable failure; duplicate clicks and stale-session completions cannot dismiss or navigate another resource. */
export function useDeleteConfirmation(
  remove: () => Promise<void>,
  onDeleted: () => void
) {
  const token = useSessionToken()
  const [pending, setPending] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const busy = React.useRef(false)
  const active = React.useRef(false)
  React.useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])

  /** Performs the confirmed deletion once, keeping the dialog open on failure and delivering success only to the original mounted session. */
  async function confirm() {
    if (busy.current || !token || getToken() !== token) return
    busy.current = true
    setPending(true)
    setError(null)
    try {
      await remove()
      if (active.current && getToken() === token) onDeleted()
    } catch (cause) {
      if (active.current && getToken() === token)
        setError(
          cause instanceof Error
            ? cause.message
            : "Deletion failed. Please try again."
        )
    } finally {
      busy.current = false
      if (active.current && getToken() === token) setPending(false)
    }
  }
  return { pending, error, confirm }
}
