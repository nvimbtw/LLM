"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { readPageDraft, writePageDraft } from "@/lib/page-draft"
import { useDraftRetentionWarning } from "./use-draft-retention-warning"

/** Retains one session/resource's explicit-save fields across navigation and reload in this tab; failed retention warns before unloading. */
export function usePageDraft<T>(
  resource: string,
  saved: T,
  parse: (value: unknown) => T,
  ready = true
) {
  const token = useSessionToken()
  const identity = React.useMemo(
    () => (token ? { token, resource } : null),
    [token, resource]
  )
  const source = JSON.stringify(saved)
  const owner = `${token}:${resource}`
  const [state, setState] = React.useState(() => ({
    source,
    owner,
    ready,
    ...(ready
      ? readPageDraft(identity, saved, parse)
      : { fields: saved, error: null, retained: true }),
  }))
  if (state.ready !== ready || state.owner !== owner) {
    setState({ source, owner, ready, ...readPageDraft(identity, saved, parse) })
  } else if (state.source !== source) {
    const clean =
      JSON.stringify(state.fields) === state.source ||
      JSON.stringify(state.fields) === source
    setState({
      ...state,
      source,
      fields: clean ? saved : state.fields,
      error: clean ? null : state.error,
    })
  }
  const dirty = JSON.stringify(state.fields) !== source

  /** Applies fields and attempts synchronous tab-local persistence before a following navigation can run. */
  function change(fields: T) {
    let error: string | null = null
    try {
      writePageDraft(identity, saved, fields)
    } catch (cause) {
      error =
        cause instanceof Error
          ? cause.message
          : "This draft could not be retained while browsing."
    }
    setState({ source, owner, ready, fields, error, retained: !error })
  }

  /** Restores the server baseline and removes the unsaved tab-local copy. */
  function reset() {
    change(saved)
  }

  /** Removes an acknowledged draft without replacing fields still awaiting their refreshed server baseline. */
  function forget() {
    try {
      writePageDraft(identity, saved, saved)
      setState((current) => ({ ...current, error: null, retained: true }))
    } catch (cause) {
      setState((current) => ({
        ...current,
        error:
          cause instanceof Error
            ? cause.message
            : "The retained draft could not be cleared.",
      }))
    }
  }

  useDraftRetentionWarning(dirty && !state.retained)

  return {
    fields: state.fields,
    change,
    reset,
    forget,
    dirty,
    persistenceError: state.error,
  }
}
