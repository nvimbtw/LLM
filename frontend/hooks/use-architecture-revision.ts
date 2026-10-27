"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import { draftDigest } from "@/lib/architecture/draft"
import { architectures } from "@/lib/architectures"
import { blocks } from "@/lib/blocks"
import type {
  ArchitectureRevisionView,
  BlockRevisionView,
} from "@/lib/architecture/contracts.generated"
import type { ArchitectureKind } from "./use-architecture-catalog"

export interface RevisionSelection {
  kind: ArchitectureKind
  id: string
  revision: string | null
  origin: "builtin" | "custom"
}

/** Reads exact immutable accepted source for one actual-session selection; builtin hashes identify implementation manifests, while custom hashes identify source bytes. */
export function useArchitectureRevision({
  kind,
  id,
  revision,
  origin,
}: RevisionSelection) {
  const token = useSessionToken()
  const [epoch, retry] = React.useReducer((value: number) => value + 1, 0)
  const resource = `${kind}:${id}:${revision}:${origin}:${epoch}`
  const [state, setState] = React.useState<{
    token: string
    resource: string
    value?: ArchitectureRevisionView | BlockRevisionView
    error?: string
  } | null>(null)
  React.useEffect(() => {
    if (!token || !revision) return
    const owner = new ArchitectureReadOwner({ token, resource })
    void owner
      .read(async (signal) => {
        const value = await (
          kind === "block" ? blocks : architectures
        ).revision(id, revision, { token, signal })
        const reference = value.accepted.reference
        const source = "source" in value ? value.source : value.yaml
        if (
          reference.resource_id !== id ||
          reference.revision_id !== revision ||
          reference.kind !== kind ||
          (origin === "custom" &&
            reference.content_hash !== draftDigest(source))
        )
          throw new Error(
            "Accepted source differs from its selected immutable revision."
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
              : "Accepted source did not load.",
        })
      )
    return () => owner.dispose()
  }, [token, resource, kind, id, revision, origin, epoch])
  const current =
    state?.token === token && state.resource === resource ? state : null
  return {
    value: revision ? (current?.value ?? null) : null,
    error: revision ? (current?.error ?? null) : null,
    loading: Boolean(token && revision && !current),
    retry,
  }
}
