import { Checkbox } from "@/components/ui/checkbox"
import {
  ResourceRow,
  ResourceLink,
  ResourceTitle,
  ResourceMetadata,
  ResourceFacts,
} from "@/components/ui/resource-list"
import type { Model } from "@/hooks/use-models"
import { cn } from "@/lib/utils"
import { formatContext, formatParams } from "./format"

export interface ModelCardProps {
  model: Model
  selectable?: boolean
  selected?: boolean
  onSelectedChange?: (next: boolean) => void
}

/** Presents model identity and measured facts as a catalogue row, preserving explicit bulk selection. */
export function ModelCard({
  model,
  selectable = false,
  selected = false,
  onSelectedChange,
}: ModelCardProps) {
  const content = <ModelContent model={model} />
  return (
    <ResourceRow
      className={cn(
        "flex flex-wrap items-start gap-x-8 gap-y-4",
        selected && "bg-accent/40"
      )}
    >
      {selectable ? (
        <label className="flex min-w-0 flex-1 cursor-pointer items-start gap-4 focus-within:ring-2 focus-within:ring-ring">
          <Checkbox
            className="mt-1"
            checked={selected}
            onCheckedChange={(next) => onSelectedChange?.(Boolean(next))}
            aria-label={`Select ${model.name}`}
          />
          {content}
        </label>
      ) : (
        <ResourceLink href={`/models/${model.id}`}>{content}</ResourceLink>
      )}
      <ResourceFacts className="gap-x-8 sm:pt-1">
        <Stat
          label="Parameters"
          value={formatParams(model.params, model.parameterCountEstimated)}
        />
        <Stat
          label="Context"
          value={
            model.contextWindow
              ? `${formatContext(model.contextWindow)} tokens`
              : "—"
          }
        />
      </ResourceFacts>
    </ResourceRow>
  )
}

/** Groups a model's wrapping name, description and explicit provenance without invented measurements. */
function ModelContent({ model }: { model: Model }) {
  return (
    <div className="min-w-0 flex-1">
      <ResourceTitle>{model.name}</ResourceTitle>
      {model.description && (
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          {model.description}
        </p>
      )}
      <ResourceMetadata>
        <span className="text-(--kind-model)">{model.family || "Custom"}</span>
        {!model.published && <span>Private</span>}
        {model.tier && <span className="capitalize">{model.tier}</span>}
      </ResourceMetadata>
    </div>
  )
}

/** Displays one measured model fact with an explicit label. */
function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs">{label}</div>
      <div className="mt-1 text-foreground">{value}</div>
    </div>
  )
}
