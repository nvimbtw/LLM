"use client"

import { useModels, type Model } from "@/hooks/use-models"
import { Skeleton } from "@/components/ui/skeleton"
import { ModelCard } from "./model-card"

// Group a status bucket by family (families alphabetical); within a family the
// largest models sit first, so size reads top-to-bottom.
function byFamily(models: Model[]): [string, Model[]][] {
  const map = new Map<string, Model[]>()
  for (const m of models) {
    const family = m.family || "Custom"
    const list = map.get(family)
    if (list) list.push(m)
    else map.set(family, [m])
  }
  return Array.from(map.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([family, list]) => [family, list.sort((a, b) => b.params - a.params)])
}

function StatusSection({ title, models }: { title: string; models: Model[] }) {
  const families = byFamily(models)
  return (
    <section>
      <header className="flex items-baseline justify-between gap-3 border-b border-border/60 pb-3">
        <h2 className="font-display text-xl font-semibold tracking-tight">{title}</h2>
        <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-muted-foreground/50 tabular-nums">
          {models.length} {models.length === 1 ? "model" : "models"}
        </span>
      </header>

      <div className="mt-6 flex flex-col gap-8">
        {families.map(([family, familyModels]) => (
          <div key={family}>
            <h3 className="mb-3 text-[11px] font-medium uppercase tracking-[0.16em] text-muted-foreground/60">
              {family}
            </h3>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {familyModels.map((model) => (
                <ModelCard key={model.id} model={model} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

export function ModelsGrid() {
  const { models, isLoading, error } = useModels()

  if (error) {
    return (
      <p className="text-sm text-destructive">Failed to load models. Please try again.</p>
    )
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <Skeleton className="h-8 w-40 rounded-md" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-44 rounded-xl" />
          ))}
        </div>
      </div>
    )
  }

  if (models.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-24 text-center">
        <p className="text-sm text-muted-foreground">No models yet.</p>
        <p className="max-w-xs text-xs text-muted-foreground/70">
          Train one and it will appear here, grouped by family.
        </p>
      </div>
    )
  }

  const published = models.filter((m) => m.published)
  const unpublished = models.filter((m) => !m.published)

  return (
    <div className="flex flex-col gap-12">
      {published.length > 0 && <StatusSection title="Published" models={published} />}
      {unpublished.length > 0 && <StatusSection title="Unpublished" models={unpublished} />}
    </div>
  )
}
