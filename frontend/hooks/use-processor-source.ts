"use client"

import * as React from "react"

import { useSessionToken } from "@/hooks/use-auth"
import { getToken } from "@/lib/api"
import {
  getProcessorSource,
  saveProcessorSource,
  type ProcessorSource,
} from "@/lib/data"

interface SourceState {
  key: string
  info: ProcessorSource | null
  draft: string
  error: string | null
  saving: boolean
}

interface SourceOwner {
  key: string
  controllers: Set<AbortController>
  saving: boolean
}

/** Owns source reads, draft edits and saves for one processor and bearer session.
 * Resource switches abort requests and suppress late save/preview callbacks.
 */
export function useProcessorSource(
  name: string,
  canEdit: boolean,
  onSaved: () => void
) {
  const token = useSessionToken()
  const key = JSON.stringify([token, name])
  const [state, setState] = React.useState<SourceState | null>(null)
  const [retry, setRetry] = React.useState(0)
  const owner = React.useRef<SourceOwner | null>(null)

  React.useEffect(() => {
    if (!token) return
    const controller = new AbortController()
    const current: SourceOwner = {
      key,
      controllers: new Set([controller]),
      saving: false,
    }
    owner.current = current
    getProcessorSource(name, controller.signal)
      .then((info) => {
        if (owner.current !== current || getToken() !== token) return
        setState({ key, info, draft: info.source, error: null, saving: false })
      })
      .catch((error: unknown) => {
        if (owner.current !== current || getToken() !== token) return
        setState({
          key,
          info: null,
          draft: "",
          error: error instanceof Error ? error.message : "Source did not load",
          saving: false,
        })
      })
    return () => {
      owner.current = null
      for (const request of current.controllers) request.abort()
    }
  }, [key, name, token, retry])

  const loaded = state?.key === key ? state : null

  /** Replaces this resource's local draft without performing network effects. */
  function setDraft(draft: string) {
    setState((current) =>
      current?.key === key ? { ...current, draft } : current
    )
  }

  /** Resets the local draft to saved disk bytes, which may differ from active code. */
  function reset() {
    setState((current) =>
      current?.key === key && current.info
        ? { ...current, draft: current.info.source, error: null }
        : current
    )
  }

  /** Saves one draft and refreshes preview only while its original resource remains current. */
  async function save() {
    const current = owner.current
    if (
      !canEdit ||
      !loaded?.info?.editable ||
      loaded.saving ||
      current?.key !== key ||
      current.saving ||
      getToken() !== token
    )
      return
    const controller = new AbortController()
    current.saving = true
    current.controllers.add(controller)
    const draft = loaded.draft
    setState({ ...loaded, saving: true, error: null })
    try {
      const result = await saveProcessorSource(name, draft, controller.signal)
      if (owner.current !== current || getToken() !== token) return
      setState({
        ...loaded,
        info: { ...loaded.info, ...result, source: draft },
        draft,
        saving: false,
        error: null,
      })
      onSaved()
    } catch (error) {
      if (owner.current !== current || getToken() !== token) return
      setState({
        ...loaded,
        saving: false,
        error: error instanceof Error ? error.message : "Source was not saved",
      })
    } finally {
      current.saving = false
      current.controllers.delete(controller)
    }
  }

  return {
    info: loaded?.info ?? null,
    draft: loaded?.draft ?? "",
    error: loaded?.error ?? null,
    saving: loaded?.saving ?? false,
    loading: loaded === null,
    setDraft,
    reset,
    save,
    reload: () => setRetry((value) => value + 1),
  }
}
