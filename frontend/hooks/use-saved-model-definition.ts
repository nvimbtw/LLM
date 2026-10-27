"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import { getModelDefinition } from "@/lib/model-graph"
import type { ModelDefinitionSummary } from "@/lib/architecture/contracts.generated"

/** Reads immutable saved semantics only for an already-authorized mounted view, masking responses across actual-session/model/definition changes. */
export function useSavedModelDefinition(id: string, definitionId: string) {
  const token = useSessionToken()
  const [epoch, retry] = React.useReducer((value: number) => value + 1, 0)
  const resource = `${id}:${definitionId}:${epoch}`
  const [state, setState] = React.useState<{
    token: string
    resource: string
    value?: ModelDefinitionSummary
    error?: string
  } | null>(null)
  React.useEffect(() => {
    if (!token) return
    const owner = new ArchitectureReadOwner({ token, resource })
    void owner
      .read(async (signal) => {
        const value = await getModelDefinition(id, { token, signal })
        if (value.definition_id !== definitionId)
          throw new Error(
            "Saved definition differs from the selected model identity."
          )
        return value
      })
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
              : "Saved definition did not load.",
        })
      )
    return () => owner.dispose()
  }, [token, resource, id, definitionId])
  const current =
    state?.token === token && state.resource === resource ? state : null
  return {
    value: current?.value ?? null,
    error: current?.error ?? null,
    loading: Boolean(token && !current),
    retry,
  }
}
