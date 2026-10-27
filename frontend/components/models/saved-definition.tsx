"use client"

import * as React from "react"
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import { useSavedModelDefinition } from "@/hooks/use-saved-model-definition"
import { SavedArchitectureGraph } from "@/components/architecture-graph/saved-graph"
import { Panel } from "@/components/ui/panel"
import { LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import {
  RetainedSourceDialog,
  type RetainedSourceSelection,
} from "@/components/architecture-graph/retained-source"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { formatParams } from "./format"

export interface SavedModelDefinitionProps {
  modelId: string
  definitionId: string
  inherited?: boolean
}

/** Mounts private saved-definition/graph reads only for the actual administrator session; ordinary model/chat views keep public scalar facts. */
export function SavedModelDefinition(props: SavedModelDefinitionProps) {
  const { isAdmin, isLoading } = useAuth()
  const token = useSessionToken()
  return !isLoading && isAdmin && token ? (
    <SavedDefinitionContent
      key={`${token}:${props.modelId}:${props.definitionId}`}
      {...props}
    />
  ) : null
}

/** Presents exact saved parameters, dependency identities, capabilities and retained profile graphs without consulting current catalog defaults. */
function SavedDefinitionContent({
  modelId,
  definitionId,
  inherited,
}: SavedModelDefinitionProps) {
  const saved = useSavedModelDefinition(modelId, definitionId)
  const [source, selectSource] = React.useState<RetainedSourceSelection | null>(
    null
  )
  const [profile, selectProfile] = React.useState<string | null>(null)
  if (saved.loading)
    return (
      <Panel variant="open" title="Saved architecture">
        <Skeleton className="h-48 w-full" />
      </Panel>
    )
  if (saved.error || !saved.value)
    return (
      <Panel variant="open" title="Saved architecture">
        <LoadError
          title="Saved definition did not load."
          hint={saved.error ?? undefined}
          onRetry={saved.retry}
        />
      </Panel>
    )
  const value = saved.value
  const receipts = value.validation_receipts ?? []
  const selected =
    receipts.find((item) => item.identity_hash === profile) ?? receipts[0]

  /** Opens frozen package bytes independently of catalog availability, preserving the selected line. */
  function openSource(revision: string, line: number | null) {
    const reference = [value.architecture, ...value.dependencies].find(
      (item) => item.revision_id === revision
    )
    selectSource({ revision, line, contentHash: reference?.content_hash })
  }
  return (
    <Panel
      variant="open"
      title="Saved architecture"
      aside={inherited ? "Inherited" : undefined}
      description={`${formatParams(value.parameter_count ?? null)} parameters`}
    >
      <div className="grid gap-5">
        <RetainedSourceDialog
          kind="models"
          id={modelId}
          definitionId={definitionId}
          selection={source}
          onClose={() => selectSource(null)}
        />
        <p className="font-mono text-xs break-all">
          Definition {value.definition_id}
        </p>
        <dl className="grid gap-x-8 sm:grid-cols-2">
          {[
            {
              label: "Context window",
              value: `${value.context_window.toLocaleString()} tokens`,
            },
            {
              label: "Vocabulary",
              value: `${value.vocabulary_size.toLocaleString()} tokens`,
            },
            ...Object.entries(value.parameters).map(([label, value]) => ({
              label,
              value: String(value),
            })),
          ].map((fact) => (
            <div
              key={fact.label}
              className="flex justify-between gap-3 border-b border-border/40 py-3 text-sm"
            >
              <dt>{fact.label}</dt>
              <dd className="text-right break-all">{fact.value}</dd>
            </div>
          ))}
        </dl>
        <p className="text-sm text-muted-foreground">
          {inherited
            ? "This run inherits these saved bindings and weights. Architecture and vocabulary are read-only."
            : "These are the bindings retained with this model and its checkpoints."}
        </p>
        <details>
          <summary className="text-sm">Pinned source identities</summary>
          <ul className="grid gap-2 pt-3 text-xs break-all">
            {[value.architecture, ...value.dependencies].map((reference) => (
              <li key={reference.revision_id}>
                <Button
                  variant="link"
                  onClick={() => openSource(reference.revision_id, null)}
                >
                  {reference.kind} · {reference.resource_id}
                </Button>
                <p className="font-mono">
                  Revision {reference.revision_id} · {reference.content_hash}
                </p>
              </li>
            ))}
          </ul>
        </details>
        {(value.capabilities?.length ?? 0) > 0 && (
          <details>
            <summary className="text-sm">Optional capabilities</summary>
            <ul className="grid gap-2 pt-3 text-sm">
              {value.capabilities!.map((item) => (
                <li key={item.name}>
                  {item.name}:{" "}
                  {item.supported
                    ? "Validated for the retained profiles"
                    : (item.reason ?? "Unavailable")}
                </li>
              ))}
            </ul>
          </details>
        )}
        {receipts.length > 0 ? (
          <>
            <label className="grid gap-2 text-sm">
              Retained validation profile
              <Select
                value={selected.identity_hash}
                onValueChange={selectProfile}
              >
                <SelectTrigger>
                  <SelectValue>
                    {() =>
                      `${selected.identity.profile.device} · ${selected.identity.profile.precision} · ${selected.identity_hash.slice(0, 12)}`
                    }
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {receipts.map((item) => (
                    <SelectItem
                      key={item.identity_hash}
                      value={item.identity_hash}
                    >
                      {item.identity.profile.device} ·{" "}
                      {item.identity.profile.precision} ·{" "}
                      {item.identity_hash.slice(0, 12)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </label>
            <details>
              <summary className="text-sm">
                Exact runtime profile and bindings
              </summary>
              <pre className="max-h-96 overflow-auto p-3 text-xs">
                {JSON.stringify(selected.identity, null, 2)}
              </pre>
            </details>
            <SavedArchitectureGraph
              key={selected.identity_hash}
              kind="models"
              id={modelId}
              receipt={selected}
              onSource={openSource}
            />
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            No retained validation graph is available.
          </p>
        )}
      </div>
    </Panel>
  )
}
