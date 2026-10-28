"use client"

import * as React from "react"
import { getToken } from "@/lib/api"
import { useSessionToken } from "./use-auth"

/** Owns one explicit conversation-title edit. Failed saves retain attempted text, duplicate submits coalesce, and stale-session completions never close replacement editors. */
export function useConversationRename(
  title: string,
  save: (next: string) => void | Promise<void>,
  onDone: () => void
) {
  const token = useSessionToken()
  const [value, setValue] = React.useState(title)
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

  /** Saves a nonblank changed title, retaining the editable attempt on failure and closing only after acknowledgement. */
  async function submit() {
    if (busy.current) return
    const next = value.trim()
    if (!next) {
      setError("Enter a conversation title.")
      return
    }
    if (next === title) {
      onDone()
      return
    }
    busy.current = true
    setPending(true)
    setError(null)
    try {
      await save(next)
      if (active.current && getToken() === token) onDone()
    } catch (cause) {
      if (active.current && getToken() === token)
        setError(
          cause instanceof Error
            ? cause.message
            : "The conversation could not be renamed."
        )
    } finally {
      busy.current = false
      if (active.current && getToken() === token) setPending(false)
    }
  }
  return { value, setValue, pending, error, submit }
}
