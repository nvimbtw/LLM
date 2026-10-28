"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import type { ArchitectureKind } from "./use-architecture-catalog"
import { architectures } from "@/lib/architectures"
import { blocks } from "@/lib/blocks"
import { ArchitectureDraft } from "@/lib/architecture/draft"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import type { ResourceDetail } from "@/lib/architecture/contracts.generated"

interface WorkspaceState {
  token: string
  resource: string
  detail?: ResourceDetail
  document?: ArchitectureDraft
  error?: string
}

/** Supplies no subscription before this resource's private draft has loaded. */
const noSubscription = () => () => {}
/** Keeps the unloaded external-store snapshot stable across renders. */
const noDraft = () => null

/** Loads one private resource with its exact draft and owns its session-scoped save controller; history has an independent read lifetime. */
export function useArchitectureWorkspace(kind: ArchitectureKind, id: string) {
  const token = useSessionToken()
  const [epoch, reload] = React.useReducer((value: number) => value + 1, 0)
  const [state, setState] = React.useState<WorkspaceState | null>(null)
  const resource = `${kind}:${id}:${epoch}`
  React.useEffect(() => {
    if (!token) return
    const owner = new ArchitectureReadOwner({ token, resource })
    let document: ArchitectureDraft | undefined
    let active = true
    owner
      .read(async (signal) => {
        const api = kind === "block" ? blocks : architectures
        const [detail, view] = await Promise.all([
          api.detail(id, { token, signal }),
          api.draft(id, { token, signal }),
        ])
        if (
          detail.summary.resource_id !== id ||
          detail.summary.kind !== kind ||
          view.resource_id !== id ||
          detail.summary.draft_version !== view.draft_version
        )
          throw new Error(
            "Draft changed while loading or belongs to another resource. Reload to read the current version."
          )
        return { detail, view }
      })
      .then((value) => {
        if (!value || !active) return
        const { detail, view } = value
        const text = "source" in view ? view.source : view.yaml
        document = new ArchitectureDraft(
          {
            resource_id: view.resource_id,
            draft_version: view.draft_version,
            content_hash: view.content_hash,
          },
          text,
          (version, bytes, signal) =>
            kind === "block"
              ? blocks.save(
                  id,
                  { expected_version: version, source: bytes },
                  { token, signal }
                )
              : architectures.save(
                  id,
                  { expected_version: version, yaml: bytes },
                  { token, signal }
                )
        )
        setState({ token, resource, detail, document })
      })
      .catch((error: unknown) => {
        if (active)
          setState({
            token,
            resource,
            error:
              error instanceof Error
                ? error.message
                : "Workspace did not load.",
          })
      })
    return () => {
      active = false
      owner.dispose()
      document?.dispose()
    }
  }, [token, resource, kind, id])
  const current =
    state?.token === token && state.resource === resource ? state : null
  const document = current?.document
  const draft = React.useSyncExternalStore(
    document?.subscribe ?? noSubscription,
    document?.getSnapshot ?? noDraft,
    noDraft
  )
  /** Replaces only current metadata after a confirmed mutation, retaining the editor's independently versioned local bytes. */
  function updateMetadata(detail: ResourceDetail) {
    if (detail.summary.resource_id !== id || detail.summary.kind !== kind)
      throw new Error("Metadata belongs to another resource.")
    setState((previous) =>
      previous?.token === token && previous.resource === resource
        ? { ...previous, detail }
        : previous
    )
  }
  return {
    updateMetadata,
    detail: current?.detail ?? null,
    draft,
    document,
    reload,
    error: current?.error ?? null,
    loading: Boolean(token && !current),
  }
}
