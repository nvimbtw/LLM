import Link from "next/link"
import { ArrowUpRight } from "lucide-react"

import { Checkbox } from "@/components/ui/checkbox"
import type { Model } from "@/hooks/use-models"
import { cn } from "@/lib/utils"
import { formatContext, formatParams } from "./format"

export interface ModelCardProps {
  model: Model
  selectable?: boolean
  selected?: boolean
  onSelectedChange?: (next: boolean) => void
}

/** Presents a model as an open editorial entry or an accessible selection label for bulk deletion. */
export function ModelCard({
  model,
  selectable = false,
  selected = false,
  onSelectedChange,
}: ModelCardProps) {
  const content = (
    <CardBody
      model={model}
      selected={selected}
      corner={
        selectable ? (
          <Checkbox
            checked={selected}
            onCheckedChange={(next) => onSelectedChange?.(Boolean(next))}
            aria-label={`Select ${model.name}`}
          />
        ) : (
          <ArrowUpRight className="size-4 text-muted-foreground transition-colors group-hover:text-foreground" />
        )
      }
    />
  )
  const className =
    "group block min-w-0 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring focus-within:ring-ring"
  return selectable ? (
    <label
      className={cn(
        className,
        "cursor-pointer focus-within:outline-2 focus-within:outline-ring"
      )}
    >
      {content}
    </label>
  ) : (
    <Link href={`/models/${model.id}`} className={className}>
      {content}
    </Link>
  )
}

/** Renders model identity and measured figures without a filled tile or decorative size meter. Explicit categories use quiet text. */
function CardBody({
  model,
  selected,
  corner,
}: {
  model: Model
  selected: boolean
  corner: React.ReactNode
}) {
  return (
    <article
      className={cn(
        "flex h-full flex-col border-t py-5",
        selected
          ? "border-primary"
          : "border-border transition-colors group-hover:border-foreground/40"
      )}
    >
      <div className="mb-3 flex items-center justify-between gap-3 text-xs text-muted-foreground">
        <span className="truncate">
          {model.family || "Custom"}
          {!model.published && " · Private"}
        </span>
        <span className="flex shrink-0 items-center gap-3">
          {model.tier && (
            <span className="capitalize italic">{model.tier}</span>
          )}
          {corner}
        </span>
      </div>
      <h2 className="line-clamp-2 font-display text-xl leading-snug tracking-tight transition-colors group-hover:text-primary">
        {model.name}
      </h2>
      {model.description && (
        <p className="mt-2 line-clamp-2 text-sm leading-relaxed text-muted-foreground">
          {model.description}
        </p>
      )}
      <dl className="mt-auto flex flex-wrap gap-x-8 gap-y-3 pt-5">
        <Stat
          label="Parameters"
          value={model.params ? formatParams(model.params) : "—"}
        />
        <Stat
          label="Context"
          value={
            model.contextWindow
              ? `${formatContext(model.contextWindow)} tokens`
              : "—"
          }
        />
      </dl>
    </article>
  )
}

/** Displays a compact model measurement with a prose label. */
function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="mt-1 font-display text-lg tabular-nums">{value}</dd>
    </div>
  )
}
