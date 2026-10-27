"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { architectures } from "@/lib/architectures"
import { blocks } from "@/lib/blocks"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import type {
  CatalogPage,
  CatalogQuery,
} from "@/lib/architecture/contracts.generated"

export type ArchitectureKind = "block" | "architecture"

interface CatalogState {
  token: string
  resource: string
  page?: CatalogPage
  error?: string
}

/** Reads one bounded catalog page for the actual administrator session/filter identity and discards replaced reads. */
export function useArchitectureCatalog(kind: ArchitectureKind) {
  const token = useSessionToken()
  const [query, setQuery] = React.useState<CatalogQuery>({
    search: "",
    origin: null,
    archived: false,
    limit: 50,
    cursor: null,
  })
  const [search, setSearch] = React.useState("")
  const [epoch, refresh] = React.useReducer((value: number) => value + 1, 0)
  const [state, setState] = React.useState<CatalogState | null>(null)
  const resource = `${kind}:${JSON.stringify(query)}:${epoch}`
  React.useEffect(() => {
    if (!token) return
    const owner = new ArchitectureReadOwner({ token, resource })
    owner
      .read(async (signal) => {
        const page = await (kind === "block" ? blocks : architectures).list(
          query,
          { token, signal }
        )
        if (page.items.some((item) => item.kind !== kind))
          throw new Error("Catalog contains a different resource family.")
        return page
      })
      .then((page) => {
        if (page) setState({ token, resource, page })
      })
      .catch((error: unknown) =>
        setState({
          token,
          resource,
          error:
            error instanceof Error ? error.message : "Catalog did not load.",
        })
      )
    return () => owner.dispose()
  }, [token, resource, kind, query])
  const current =
    state?.token === token && state.resource === resource ? state : null

  /** Starts a new filter scope from its first page; search submits explicitly instead of issuing a request per keystroke. */
  function filter(patch: Partial<CatalogQuery>) {
    setQuery((previous) => ({ ...previous, ...patch, cursor: null }))
  }

  return {
    query,
    search,
    setSearch,
    filter,
    refresh,
    next: () => {
      if (current?.page?.next_cursor)
        setQuery((previous) => ({
          ...previous,
          cursor: current.page!.next_cursor,
        }))
    },
    page: current?.page ?? null,
    error: current?.error ?? null,
    loading: Boolean(token && !current),
  }
}
