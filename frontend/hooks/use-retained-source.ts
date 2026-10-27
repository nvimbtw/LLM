"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import { getRetainedSource } from "@/lib/model-graph"
import type { RetainedSource } from "@/lib/architecture/contracts.generated"

export interface RetainedSourceRequest {
  kind: "models" | "validations"
  id: string
  revision: string
  definitionId?: string
  contentHash?: string
}

/** Owns frozen-source observation for one actual session and exact resource identity, masking replaced reads and aborting transport. */
export function useRetainedSource({
  kind,
  id,
  revision,
  definitionId,
  contentHash,
}: RetainedSourceRequest) {
  const token = useSessionToken()
  const [epoch, retry] = React.useReducer((value: number) => value + 1, 0)
  const identity = JSON.stringify([
    token,
    kind,
    id,
    revision,
    definitionId,
    contentHash,
    epoch,
  ])
  const [state, setState] = React.useState<{
    identity: string
    value?: RetainedSource
    error?: string
  } | null>(null)
  React.useEffect(() => {
    if (!token) return
    const owner = new ArchitectureReadOwner({ token, resource: identity })
    void owner
      .read((signal) =>
        getRetainedSource(
          kind,
          id,
          revision,
          { token, signal },
          definitionId ?? null,
          contentHash ?? null
        )
      )
      .then((value) => {
        if (value) setState({ identity, value })
      })
      .catch((cause: unknown) =>
        setState({
          identity,
          error:
            cause instanceof Error
              ? cause.message.slice(0, 2000)
              : "Retained source is unavailable.",
        })
      )
    return () => owner.dispose()
  }, [kind, id, definitionId, revision, contentHash, token, identity])
  const current = state?.identity === identity ? state : null
  return {
    value: current?.value ?? null,
    error: current?.error ?? null,
    loading: Boolean(token && !current),
    retry,
  }
}
