import Link from "next/link"

import type { Model } from "@/hooks/use-models"
import { ModelSizeMeter } from "./model-glyph"
import { formatContext, formatParams } from "./format"

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase">
        {label}
      </dt>
      <dd className="mt-0.5 font-display text-base font-semibold tabular-nums">
        {value}
      </dd>
    </div>
  )
}

export interface ModelCardProps {
  model: Model
}

export function ModelCard({ model }: ModelCardProps) {
  return (
    <Link
      href={`/models/${model.id}`}
      className="group block rounded-lg focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
    >
      <div className="flex h-full flex-col rounded-lg border border-border bg-background p-5 transition-colors group-hover:border-foreground/25">
        <div className="flex items-start justify-between gap-3">
          <h3 className="min-w-0 flex-1 truncate font-display text-lg leading-tight font-semibold tracking-tight">
            {model.name}
          </h3>
          <div className="flex shrink-0 flex-col items-end gap-1.5 pt-0.5">
            <ModelSizeMeter tier={model.tier} />
            <span className="text-[9px] font-medium tracking-[0.14em] text-muted-foreground uppercase">
              {model.tier}
            </span>
          </div>
        </div>

        {/* The grid is flat now, so the card carries the two facts the family and
            visibility headings used to: which family it belongs to, and whether it
            is visible to anyone but an admin. Private is called out; published is
            the expected state and says nothing by being silent. */}
        <div className="mt-1.5 flex items-center gap-2 text-[10px] font-medium tracking-[0.14em] text-muted-foreground uppercase">
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
    </Link>
  )
}
