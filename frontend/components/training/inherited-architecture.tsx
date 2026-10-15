"use client"

import { ArchitectureFlow } from "@/components/training/architecture-flow"
import { compact } from "@/components/training/stat-run"
import { LoadError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import { useSessionToken } from "@/hooks/use-auth"
import { useModel } from "@/hooks/use-model"
import type { TrainConfig } from "@/lib/training"

export interface InheritedArchitectureProps {
  cfg: TrainConfig
  modelId: string
}

/** Shows the chosen model's inherited shape without editing the draft. Reads and animation state reset when session or model identity changes. */
export function InheritedArchitecture(props: InheritedArchitectureProps) {
  const token = useSessionToken()
  return token ? (
    <InheritedArchitectureDetails
      key={`${token}:${props.modelId}`}
      {...props}
    />
  ) : null
}

/** Loads the owning model's resolved architecture and vocabulary, then renders read-only facts, a parameter estimate and the training loop. Failed or absent metadata never falls back to the draft's defaults. */
function InheritedArchitectureDetails({
  cfg,
  modelId,
}: InheritedArchitectureProps) {
  const { model, isLoading, error, notFound } = useModel(modelId)
  if (isLoading)
    return (
      <Panel variant="open" title="Architecture">
        <Skeleton className="h-40" />
      </Panel>
    )
  const architecture = model?.architecture
  if (error || notFound || !architecture)
    return (
      <Panel variant="open" title="Architecture">
        <LoadError
          title={error?.message ?? "This model's architecture is unavailable."}
        />
      </Panel>
    )
  const vocabularySize = model.vocabularySize
  const parameterCount = model.params || null
  const facts = [
    { label: "Dimensions", value: architecture.dimensions.toLocaleString() },
    {
      label: "Context window",
      value: `${architecture.ctx_window.toLocaleString()} tokens`,
    },
    { label: "Heads", value: architecture.heads.toLocaleString() },
    { label: "Layers", value: architecture.layers.toLocaleString() },
    {
      label: "FF multiplier",
      value: architecture.ff_multiplier.toLocaleString(),
    },
    { label: "Vocabulary", value: vocabularySize?.toLocaleString() ?? "—" },
  ]
  return (
    <Panel
      variant="open"
      title="Architecture"
      descriptionLayout="stacked"
      description={`${parameterCount === null ? "—" : `≈ ${compact(parameterCount)}`} parameters`}
      aside="Inherited"
    >
      <dl className="grid gap-x-8 sm:grid-cols-2">
        {facts.map((fact) => (
          <div
            key={fact.label}
            className="flex items-baseline justify-between gap-3 border-b border-border/40 py-4"
          >
            <dt className="text-sm">{fact.label}</dt>
            <dd className="text-sm tabular-nums">{fact.value}</dd>
          </div>
        ))}
      </dl>
      <ArchitectureFlow view="training" cfg={{ ...cfg, ...architecture }} />
    </Panel>
  )
}
