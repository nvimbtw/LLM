"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { architectures } from "@/lib/architectures"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import type { ArchitectureRevisionView } from "@/lib/architecture/contracts.generated"

/** Resolves only the explicitly pinned recipe UUID for one actual session; replacement never falls back to a resource's latest revision or legacy defaults. */
export function useSelectedTrainingArchitecture(revision: string | null) {
  const token = useSessionToken()
  const [epoch, retry] = React.useReducer((value: number) => value + 1, 0)
  const resource = `${revision}:${epoch}`
  const [state, setState] = React.useState<{
    token: string
    resource: string
    value?: ArchitectureRevisionView
    error?: string
  } | null>(null)
  React.useEffect(() => {
    if (!token || !revision) return
    const owner = new ArchitectureReadOwner({ token, resource })
    void owner
      .read((signal) => architectures.selected(revision, { token, signal }))
      .then((value) => {
        if (value) setState({ token, resource, value })
      })
      .catch((cause: unknown) =>
        setState({
          token,
          resource,
          error:
            cause instanceof Error
              ? cause.message.slice(0, 2000)
              : "Selected recipe did not load.",
        })
      )
    return () => owner.dispose()
  }, [token, resource, revision])
  const current =
    state?.token === token && state.resource === resource ? state : null
  return {
    value: revision ? (current?.value ?? null) : null,
    error: revision ? (current?.error ?? null) : null,
    loading: Boolean(token && revision && !current),
    retry,
  }
}
