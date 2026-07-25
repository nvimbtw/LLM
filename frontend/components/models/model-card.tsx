import Link from "next/link"

import type { Model } from "@/hooks/use-models"
import { ModelSizeMeter } from "./model-glyph"
import { formatContext, formatParams } from "./format"

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] font-medium uppercase tracking-[0.12em] text-muted-foreground/60">
        {label}
      </dt>
      <dd className="mt-0.5 font-display text-base font-semibold tabular-nums">{value}</dd>
    </div>
  )
}

export function ModelCard({ model }: { model: Model }) {
  return (
    <Link
      href={`/models/${model.id}`}
      className="group block rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <div className="flex h-full flex-col rounded-xl border border-border/60 bg-background p-5 transition-colors group-hover:border-foreground/25">
        <div className="flex items-start justify-between gap-3">
          <h3 className="min-w-0 flex-1 truncate font-display text-lg font-semibold leading-tight tracking-tight">
            {model.name}
          </h3>
          <div className="flex shrink-0 flex-col items-end gap-1.5 pt-0.5">
            <ModelSizeMeter tier={model.tier} />
            <span className="text-[9px] font-medium uppercase tracking-[0.14em] text-muted-foreground/50">
              {model.tier}
            </span>
          </div>
        </div>

        <p className="mt-3 line-clamp-2 min-h-[2.5rem] text-sm text-muted-foreground">
          {model.description}
        </p>

        <dl className="mt-auto flex gap-8 border-t border-border/60 pt-3">
          <Stat label="Params" value={formatParams(model.params)} />
          <Stat label="Context" value={formatContext(model.contextWindow)} />
        </dl>
      </div>
    </Link>
  )
}
