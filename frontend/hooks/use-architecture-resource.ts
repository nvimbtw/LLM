"use client"
import { useEffect, useState } from "react"
import { useSessionToken } from "./use-auth"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import { blocks } from "@/lib/blocks"
import { architectures } from "@/lib/architectures"
import type { ResourceDetail } from "@/lib/architecture/contracts.generated"
interface ResourceState {
  token: string
  resource: string
  value?: ResourceDetail
  error?: string
}
/** Loads catalog metadata for the actual session/resource and aborts stale reads on identity replacement. */
export function useArchitectureResource(
  kind: "block" | "architecture",
  id: string
) {
  const token = useSessionToken()
  const resource = `${kind}:${id}`
  const [state, setState] = useState<ResourceState | null>(null)
  useEffect(() => {
    if (!token) return
    const owner = new ArchitectureReadOwner({ token, resource })
    owner
      .read((signal) =>
        (kind === "block" ? blocks : architectures).detail(id, {
          token,
          signal,
        })
      )
      .then((value) => {
        if (value) setState({ token, resource, value })
      })
      .catch((error: unknown) =>
        setState({
          token,
          resource,
          error:
            error instanceof Error ? error.message : "Could not load resource.",
        })
      )
    return () => owner.dispose()
  }, [token, resource, kind, id])
  const current =
    state?.token === token && state?.resource === resource ? state : null
  return {
    value: current?.value ?? null,
    error: current?.error ?? null,
    loading: Boolean(token && !current),
  }
}
