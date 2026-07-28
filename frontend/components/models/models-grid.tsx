"use client"

import * as React from "react"

import { Skeleton } from "@/components/ui/skeleton"
import { useAuth } from "@/hooks/use-auth"
import { useModels, type Model } from "@/hooks/use-models"
import { formatCount } from "./format"
import { ModelCard } from "./model-card"
import { ModelsFilter, type Visibility } from "./models-filter"

/**
 * The registry: a ledger of what exists, then one grid of it.
 *
 * Previously this nested a family section inside a visibility section inside the
 * grid — two levels of heading over a set that is usually a dozen rows, so the page
 * was mostly scaffolding and the same family appeared twice whenever it held both a
 * published and an unpublished model. Family and visibility are now filters
 * (`ModelsFilter`), which is what they always were: ways of narrowing one list, not
 * a hierarchy the list has.
 *
 * The grid itself is sorted largest-first. Size is the one ordering that reads
 * without a heading to explain it, and it puts the model most likely to be wanted
 * in the top-left.
 */
export function ModelsGrid() {
  const { models, isLoading, error } = useModels()
  const { isAdmin } = useAuth()

  const [family, setFamily] = React.useState<string | null>(null)
  const [visibility, setVisibility] = React.useState<Visibility>("all")
  const [query, setQuery] = React.useState("")

  const families = React.useMemo(() => familyCounts(models), [models])
  const counts = React.useMemo(
    () => ({
      all: models.length,
      published: models.filter((m) => m.published).length,
      private: models.filter((m) => !m.published).length,
    }),
    [models]
  )

  const shown = React.useMemo(() => {
    const needle = query.trim().toLowerCase()
    return models
      .filter((m) => family === null || familyOf(m) === family)
      .filter((m) =>
        visibility === "all" ? true : m.published === (visibility === "published")
      )
      .filter(
        (m) =>
          needle === "" ||
          m.name.toLowerCase().includes(needle) ||
          m.description.toLowerCase().includes(needle) ||
          familyOf(m).toLowerCase().includes(needle)
      )
      .sort((a, b) => b.params - a.params)
  }, [models, family, visibility, query])

  if (error) {
    return (
      <p className="text-sm text-destructive">
        Failed to load models. Please try again.
      </p>
    )
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <Skeleton className="h-20 rounded-lg" />
        <Skeleton className="h-8 w-72 rounded-md" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-44 rounded-lg" />
          ))}
        </div>
      </div>
    )
  }

  if (models.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-24 text-center">
        <p className="text-sm text-muted-foreground">No models yet.</p>
        <p className="max-w-xs text-xs text-muted-foreground">
          Train one and it will appear here.
        </p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <Ledger models={models} families={families.length} />

      <ModelsFilter
        families={families}
        family={family}
        onFamily={setFamily}
        visibility={visibility}
        onVisibility={setVisibility}
        counts={counts}
        query={query}
        onQuery={setQuery}
        showVisibility={isAdmin}
      />

      {shown.length === 0 ? (
        <p className="py-16 text-center text-sm text-muted-foreground">
          Nothing matches that.
        </p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {shown.map((model) => (
            <ModelCard key={model.id} model={model} />
          ))}
        </div>
      )}
    </div>
  )
}

// The headline figures, in the divided-ledger idiom the model detail page and the
// usage page share — so "how big is the registry" is answered in the same shape
// wherever it is asked.
function Ledger({ models, families }: { models: Model[]; families: number }) {
  const largest = models.reduce((max, m) => Math.max(max, m.params), 0)
  const context = models.reduce((max, m) => Math.max(max, m.contextWindow), 0)

  return (
    <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border/60 sm:grid-cols-4">
      <Figure label="Models" value={String(models.length)} />
      <Figure label="Families" value={String(families)} />
      <Figure
        label="Largest"
        value={largest ? formatCount(largest) : "—"}
        hint="parameters"
      />
      <Figure
        label="Longest context"
        value={context ? formatCount(context) : "—"}
        hint="tokens"
      />
    </dl>
  )
}

function Figure({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint?: string
}) {
  return (
    <div className="bg-background px-5 py-4">
      <dt className="text-xs font-medium tracking-[0.16em] text-muted-foreground uppercase">
        {label}
      </dt>
      <dd className="mt-1.5 font-display text-2xl font-semibold tracking-tight tabular-nums">
        {value}
      </dd>
      {hint && <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}

/** A model with no family recorded is its own bucket rather than a blank one. */
function familyOf(model: Model): string {
  return model.family || "Custom"
}

// Families with how many models each holds, alphabetical — the order a filter row
// is read in, unlike the grid, which is ordered by size.
function familyCounts(models: Model[]): { name: string; count: number }[] {
  const counts = new Map<string, number>()
  for (const model of models) {
    const name = familyOf(model)
    counts.set(name, (counts.get(name) ?? 0) + 1)
  }
  return Array.from(counts, ([name, count]) => ({ name, count })).sort((a, b) =>
    a.name.localeCompare(b.name)
  )
}
