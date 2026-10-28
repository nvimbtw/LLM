"use client"

import * as React from "react"
import { useSessionToken } from "@/hooks/use-auth"
import { GraphScopeOwner } from "@/lib/architecture-graph"
import { getModelGraph } from "@/lib/model-graph"
import { decodeArchitecture } from "@/lib/architecture/decoder"
import type { ValidationReceipt } from "@/lib/architecture/contracts.generated"
import {
  ArchitectureGraphView,
  type ArchitectureGraphViewProps,
} from "./graph-view"
import {
  RetainedSourceDialog,
  type RetainedSourceSelection,
} from "./retained-source"

export interface SavedArchitectureGraphProps {
  kind: "models" | "validations"
  id: string
  receipt: ValidationReceipt
  architectureView?: boolean
  stale?: boolean
  onSource?: ArchitectureGraphViewProps["onSource"]
}

/** Supplies stable empty graph observation before a captured-session owner has completed its first read. */
const noSnapshot = () => null
/** Supplies an empty graph subscription while no matching session/resource owner exists. */
const noSubscription = () => () => {}

/** Reads retained engine graph scopes with one owner per actual session/resource/receipt; changing presentation never executes or cancels a model. */
export function SavedArchitectureGraph({
  kind,
  id,
  receipt,
  architectureView = true,
  stale,
  onSource,
}: SavedArchitectureGraphProps) {
  const token = useSessionToken()
  const [source, selectSource] = React.useState<RetainedSourceSelection | null>(
    null
  )
  const resource = `${kind}:${id}:${receipt.identity_hash}:${architectureView}`
  const [holder, setHolder] = React.useState<{
    token: string
    resource: string
    owner: GraphScopeOwner
  } | null>(null)
  const initial = React.useMemo(
    () =>
      decodeArchitecture("GraphQuery", {
        identity_hash: receipt.identity_hash,
        view: architectureView ? "architecture" : "operations",
        mode: "eval",
        signature: "full_prefix",
        scope: "root",
        limit: 256,
      }),
    [receipt.identity_hash, architectureView]
  )
  const available = Boolean(receipt.graph_manifest_hash)
  React.useEffect(() => {
    if (!token || !available) return
    const owner = new GraphScopeOwner(
      { token, resource },
      initial,
      (query, signal) => getModelGraph(kind, id, query, { token, signal })
    )
    let active = true
    void owner.load(initial).then(() => {
      if (active) setHolder({ token, resource, owner })
    })
    return () => {
      active = false
      owner.dispose()
    }
  }, [token, resource, initial, available, kind, id])
  const owner =
    holder?.token === token && holder.resource === resource
      ? holder.owner
      : null
  const state = React.useSyncExternalStore(
    owner?.subscribe ?? noSubscription,
    owner?.snapshot ?? noSnapshot,
    noSnapshot
  )
  if (!available)
    return (
      <p className="text-sm text-muted-foreground">
        No captured graph is available for this receipt.
      </p>
    )
  return (
    <>
      <ArchitectureGraphView
        key={resource}
        architectureView={architectureView}
        query={state?.query ?? initial}
        fragment={state?.fragment ?? null}
        loading={!state || state.loading}
        error={state?.error ?? null}
        stale={stale}
        onQuery={(query) => {
          void owner?.load(query)
        }}
        onSource={
          onSource ??
          (kind === "validations"
            ? (revision, line) => selectSource({ revision, line })
            : undefined)
        }
      />
      {kind === "validations" && !onSource && (
        <RetainedSourceDialog
          kind={kind}
          id={id}
          selection={source}
          onClose={() => selectSource(null)}
        />
      )}
    </>
  )
}
