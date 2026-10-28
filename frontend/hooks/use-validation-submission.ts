"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { ValidationSubmission } from "@/lib/architecture/validation-submission"
import { blocks } from "@/lib/blocks"
import { architectures } from "@/lib/architectures"
import type { ArchitectureKind } from "./use-architecture-catalog"

/** Supplies a stable absence while this editor has no matching actual-session submission owner. */
const noSnapshot = () => null
/** Supplies an empty subscription before the matching submission owner is mounted. */
const noSubscription = () => () => {}

/** Binds one explicit validation submitter to the actual administrator session/family/resource, replacing all pending state when that identity changes. */
export function useValidationSubmission(kind: ArchitectureKind, id: string) {
  const token = useSessionToken()
  const resource = `${kind}:${id}`
  const [holder, setHolder] = React.useState<{
    token: string
    resource: string
    owner: ValidationSubmission
  } | null>(null)
  React.useEffect(() => {
    if (!token) return
    const owner = new ValidationSubmission((request, signal) =>
      (kind === "block" ? blocks : architectures).validate(id, request, {
        token,
        signal,
      })
    )
    let active = true
    queueMicrotask(() => {
      if (active) setHolder({ token, resource, owner })
    })
    return () => {
      active = false
      owner.dispose()
    }
  }, [kind, id, token, resource])
  const owner =
    holder?.token === token && holder.resource === resource
      ? holder.owner
      : null
  const state = React.useSyncExternalStore(
    owner?.subscribe ?? noSubscription,
    owner?.getSnapshot ?? noSnapshot,
    noSnapshot
  )
  return { owner, state }
}
