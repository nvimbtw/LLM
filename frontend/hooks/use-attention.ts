"use client"

import * as React from "react"
import { useSessionToken } from "@/hooks/use-auth"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import {
  catalogAfterMeasurement,
  fetchAttention,
  fetchInspectionPoints,
  type AttentionMatrix,
  type AttentionQuery,
  type InspectionCatalog,
} from "@/lib/attention"

interface AttentionState {
  owner: ArchitectureReadOwner
  catalog: InspectionCatalog | null
  matrix: AttentionMatrix | null
  loading: boolean
  error: string | null
}

/** Owns explicit attention discovery/measurement for one real session/model and suppresses replaced selection responses. */
export function useAttention(modelId: string) {
  const token = useSessionToken()
  const resource = `attention:${modelId}`
  const ownerRef = React.useRef<ArchitectureReadOwner | null>(null)
  const [state, setState] = React.useState<AttentionState | null>(null)
  React.useEffect(() => {
    const owner = new ArchitectureReadOwner({ token: token ?? "", resource })
    ownerRef.current = owner
    return () => {
      owner.dispose()
      ownerRef.current = null
    }
  }, [token, resource])
  const current =
    state?.owner.identity.token === token &&
    state.owner.identity.resource === resource
      ? state
      : null
  const catalog = current?.catalog ?? null

  /** Cancels obsolete work and removes its matrix/error before controls acquire a new selection. */
  const clear = React.useCallback(() => {
    const owner = ownerRef.current
    if (!owner) return
    owner.cancel()
    setState((previous) => ({
      owner,
      catalog: previous?.owner === owner ? previous.catalog : null,
      matrix: null,
      loading: false,
      error: null,
    }))
  }, [])

  /** Loads candidates only on user request; a discovered declaration does not yet claim numerical support. */
  const discover = React.useCallback(async () => {
    const owner = ownerRef.current
    if (!token || !owner) return
    setState({ owner, catalog: null, matrix: null, loading: true, error: null })
    try {
      const value = await owner.read((signal) =>
        fetchInspectionPoints(modelId, { token, signal })
      )
      if (value)
        setState({
          owner,
          catalog: value,
          matrix: null,
          loading: false,
          error: null,
        })
    } catch (error) {
      setState({
        owner,
        catalog: null,
        matrix: null,
        loading: false,
        error:
          error instanceof Error
            ? error.message
            : "Unable to discover attention views.",
      })
    }
  }, [token, modelId])

  /** Replaces an in-flight measurement and updates only the current view's actual proof after successful decoding. */
  const load = React.useCallback(
    async (query: AttentionQuery) => {
      const owner = ownerRef.current
      if (!token || !catalog || !owner) return
      setState({ owner, catalog, matrix: null, loading: true, error: null })
      try {
        const value = await owner.read(async (signal) => {
          const matrix = await fetchAttention(modelId, query, { token, signal })
          return { matrix, catalog: catalogAfterMeasurement(catalog, matrix) }
        })
        if (value) setState({ owner, ...value, loading: false, error: null })
      } catch (error) {
        setState({
          owner,
          catalog,
          matrix: null,
          loading: false,
          error:
            error instanceof Error
              ? error.message
              : "Unable to read this model's attention.",
        })
      }
    },
    [token, modelId, catalog]
  )

  return {
    catalog,
    matrix: current?.matrix ?? null,
    isLoading: current?.loading ?? false,
    error: current?.error ?? null,
    discover,
    load,
    clear,
  }
}
