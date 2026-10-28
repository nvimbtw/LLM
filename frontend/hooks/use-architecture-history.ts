"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import { architectures } from "@/lib/architectures"
import { blocks } from "@/lib/blocks"
import type { RevisionPage } from "@/lib/architecture/contracts.generated"
import type { ArchitectureKind } from "./use-architecture-catalog"

/** Owns one bounded revision-history page independently from unsaved editor bytes; newly accepted heads reset pagination without reloading the draft. */
export function useArchitectureHistory(
  kind: ArchitectureKind,
  id: string,
  accepted: string | null
) {
  const token = useSessionToken()
  const resource = `${kind}:${id}:${accepted}`
  const [page, setPage] = React.useState<{
    resource: string
    cursor: string | null
  }>({ resource, cursor: null })
  const [epoch, retry] = React.useReducer((value: number) => value + 1, 0)
  const cursor = page.resource === resource ? page.cursor : null
  const key = `${resource}:${cursor}:${epoch}`
  const [state, setState] = React.useState<{
    token: string
    key: string
    value?: RevisionPage
    error?: string
  } | null>(null)
  React.useEffect(() => {
    if (!token) return
    const owner = new ArchitectureReadOwner({ token, resource: key })
    void owner
      .read(async (signal) => {
        const value = await (
          kind === "block" ? blocks : architectures
        ).revisions(id, { token, signal }, cursor)
        if (
          value.items.some(
            (item) =>
              item.reference.resource_id !== id || item.reference.kind !== kind
          )
        )
          throw new Error("History belongs to another resource.")
        return value
      })
      .then((value) => {
        if (value) setState({ token, key, value })
      })
      .catch((cause: unknown) =>
        setState({
          token,
          key,
          error:
            cause instanceof Error
              ? cause.message.slice(0, 2000)
              : "Accepted history did not load.",
        })
      )
    return () => owner.dispose()
  }, [token, key, kind, id, cursor])
  const current = state?.token === token && state.key === key ? state : null
  /** Returns to the newest bounded history page without replacing local source. */
  function first() {
    setPage({ resource, cursor: null })
  }
  /** Reads the server's opaque next-page cursor only for this current immutable history scope. */
  function next() {
    if (current?.value?.next_cursor)
      setPage({ resource, cursor: current.value.next_cursor })
  }
  return {
    value: current?.value ?? null,
    error: current?.error ?? null,
    loading: Boolean(token && !current),
    cursor,
    retry,
    first,
    next,
  }
}
