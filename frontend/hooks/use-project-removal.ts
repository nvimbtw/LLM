"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { getToken } from "@/lib/api"
import { useProjects } from "@/components/projects/projects-provider"

/** Owns deletion of a project and its failure state; late account/navigation completions cannot redirect another view. */
export function useProjectRemoval(projectId: string, onRemoved: () => void) {
  const token = useSessionToken()
  const { remove } = useProjects()
  const [busy, setBusy] = React.useState(false)
  const busyRef = React.useRef(false)
  const [error, setError] = React.useState<string | null>(null)
  const active = React.useRef(true)
  React.useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])

  /** Deletes only the project, leaving chats intact and preserving the confirmation dialog on failure. */
  const confirm = async (): Promise<void> => {
    if (busyRef.current) return
    busyRef.current = true
    setBusy(true)
    setError(null)
    try {
      await remove(projectId)
      if (active.current && getToken() === token) onRemoved()
    } catch (failure) {
      if (active.current && getToken() === token)
        setError(
          failure instanceof Error
            ? failure.message
            : "Unable to delete this project."
        )
    } finally {
      busyRef.current = false
      if (active.current && getToken() === token) setBusy(false)
    }
  }
  return { busy, error, confirm }
}
