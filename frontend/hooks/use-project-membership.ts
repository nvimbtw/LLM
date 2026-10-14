"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { getToken } from "@/lib/api"
import { useProjects } from "@/components/projects/projects-provider"

/** Owns one membership write and its error, keeping account changes or closed dialogs from completing an obsolete action. */
export function useProjectMembership(onMoved: () => void) {
  const token = useSessionToken()
  const { moveChat } = useProjects()
  const [state, setState] = React.useState<{
    token: string | null
    busy: boolean
    error: string | null
  }>({ token, busy: false, error: null })
  const active = React.useRef(true)
  const busy = React.useRef(false)
  React.useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])

  /** Persists the chosen target project, preserving the dialog and its selection on failure. */
  const move = async (
    chatId: string,
    projectId: string | null
  ): Promise<void> => {
    if (busy.current) return
    busy.current = true
    setState({ token, busy: true, error: null })
    try {
      await moveChat(chatId, projectId)
      if (active.current && getToken() === token) onMoved()
    } catch (failure) {
      if (active.current && getToken() === token)
        setState({
          token,
          busy: false,
          error:
            failure instanceof Error
              ? failure.message
              : "Unable to move this chat.",
        })
    } finally {
      busy.current = false
      if (active.current && getToken() === token)
        setState((current) => ({ ...current, busy: false }))
    }
  }
  return {
    move,
    busy: state.token === token && state.busy,
    error: state.token === token ? state.error : null,
  }
}
