import Link from "next/link"

import { Checkbox } from "@/components/ui/checkbox"
import type { Model } from "@/hooks/use-models"
import { cn } from "@/lib/utils"
import { ModelSizeMeter } from "./model-glyph"
import { formatContext, formatParams } from "./format"

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="eyebrow text-muted-foreground">
        {label}
      </dt>
      <dd className="figure mt-0.5 text-base">
        {value}
      </dd>
    </div>
  )
}

export interface ModelCardProps {
  model: Model
  /**
   * Selecting instead of navigating. The whole card becomes the checkbox's label
   * rather than a link — deliberately either/or, because a checkbox nested inside
   * a link is ambiguous to click and worse to reach by keyboard.
   */
  selectable?: boolean
  selected?: boolean
  onSelectedChange?: (next: boolean) => void
}

export function ModelCard({
  model,
  selectable = false,
  selected = false,
  onSelectedChange,
}: ModelCardProps) {
  if (selectable) {
    return (
      <label className="group block cursor-pointer rounded-(--radius-sheet) focus-within:ring-2 focus-within:ring-ring">
        <CardBody
          model={model}
          selected={selected}
          corner={
            <Checkbox
              checked={selected}
              onCheckedChange={(next) => onSelectedChange?.(Boolean(next))}
              aria-label={`Select ${model.name}`}
            />
          }
        />
      </label>
    )
  }

  return (
    <Link
      href={`/models/${model.id}`}
      className="group block rounded-(--radius-sheet) focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
    >
      <CardBody model={model} />
    </Link>
  )
}

/**
 * The card itself. Shared by both modes so selecting does not redraw the registry
 * into a different-looking thing — only the top-right corner and the border change.
 */
function CardBody({
  model,
  selected = false,
  corner,
}: {
  model: Model
  selected?: boolean
  corner?: React.ReactNode
}) {
  return (
    <div
      className={cn(
        // A sheet: the card surface (this used to be `bg-background`, which made
        // a tile the page colour with a line round it — the whole registry read
        // flat because the only thing separating a card from the page was its
        // border). The lift is a single pixel; anything more turns a grid of
        // eight into a fairground.
        "flex h-full flex-col rounded-(--radius-sheet) border bg-card p-5 shadow-sheet",
        "transition-[box-shadow,border-color,translate] duration-150",
        "group-hover:-translate-y-px group-hover:shadow-sheet-hover",
        "motion-reduce:transition-none motion-reduce:group-hover:translate-y-0",
        selected
          ? "border-foreground/40 bg-muted/40"
          : "border-border/45 group-hover:border-border"
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="min-w-0 flex-1 truncate font-display text-lg leading-tight font-semibold tracking-tight">
          {model.name}
        </h3>
        <div className="flex shrink-0 flex-col items-end gap-1.5 pt-0.5">
          {corner ?? (
            <>
              <ModelSizeMeter tier={model.tier} />
              <span className="eyebrow text-muted-foreground">
                {model.tier}
              </span>
            </>
          )}
        </div>
      </div>

      {/* The grid is flat now, so the card carries the two facts the family and
          visibility headings used to: which family it belongs to, and whether it
          is visible to anyone but an admin. Private is called out; published is
          the expected state and says nothing by being silent. */}
      <div className="eyebrow mt-1.5 flex items-center gap-2 text-muted-foreground">
        <span className="truncate">{model.family || "Custom"}</span>
        {!model.published && (
          <>
            <span className="text-muted-foreground/30">·</span>
            <span className="shrink-0">Private</span>
          </>
        )}
      </div>

      <p className="mt-3 line-clamp-2 min-h-[2.5rem] text-sm text-muted-foreground">
        {model.description}
      </p>

      <dl className="mt-auto flex gap-8 border-t border-border pt-3">
        <Stat label="Params" value={formatParams(model.params)} />
        <Stat label="Context" value={formatContext(model.contextWindow)} />
      </dl>
    </div>
  )
}
