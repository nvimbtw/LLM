"use client"

import * as React from "react"
import { Trash2, X } from "lucide-react"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/toast"
import { useAuth } from "@/hooks/use-auth"
import { deleteModel, useModels, type Model } from "@/hooks/use-models"
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
  const [refreshKey, setRefreshKey] = React.useState(0)
  const { models, isLoading, error } = useModels(refreshKey)
  const { isAdmin } = useAuth()

  const [family, setFamily] = React.useState<string | null>(null)
  const [visibility, setVisibility] = React.useState<Visibility>("all")
  const [query, setQuery] = React.useState("")
  const [selecting, setSelecting] = React.useState(false)
  const [selected, setSelected] = React.useState<ReadonlySet<string>>(new Set())

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
        visibility === "all"
          ? true
          : m.published === (visibility === "published")
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

  // Selection is held as ids and resolved against the *whole* registry, not the
  // filtered view: narrowing the filter must not silently drop what is already
  // ticked. The confirm dialog names every model it is about to delete, so a
  // selection scrolled off screen can still never be deleted unseen.
  const chosen = React.useMemo(
    () => models.filter((m) => selected.has(m.id)),
    [models, selected]
  )

  const toggle = React.useCallback((id: string, next: boolean) => {
    setSelected((prev) => {
      const copy = new Set(prev)
      if (next) copy.add(id)
      else copy.delete(id)
      return copy
    })
  }, [])

  const stopSelecting = React.useCallback(() => {
    setSelecting(false)
    setSelected(new Set())
  }, [])

  const deleteChosen = React.useCallback(async () => {
    // Sequential rather than all at once: each delete is a transaction plus an
    // `rm -rf` of the weights, and a batch is usually a post-train run's leftovers
    // — a dozen or more. Firing those concurrently buys nothing and makes a
    // partial failure harder to report honestly.
    const failures: string[] = []
    for (const model of chosen) {
      try {
        await deleteModel(model.id)
      } catch {
        failures.push(model.name)
      }
    }

    const deleted = chosen.length - failures.length
    if (failures.length === 0) {
      toast.success(`Deleted ${count(deleted, "model")}.`)
    } else if (deleted === 0) {
      toast.error(
        `Couldn't delete ${count(failures.length, "model")}.`,
        failures.join(", ")
      )
    } else {
      // Partial failure is the case worth stating precisely — the grid is about to
      // refresh and show a number that would otherwise need explaining.
      toast.error(
        `Deleted ${deleted} of ${chosen.length}.`,
        `Still there: ${failures.join(", ")}`
      )
    }

    stopSelecting()
    setRefreshKey((k) => k + 1)
  }, [chosen, stopSelecting])

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

      {isAdmin && (
        <SelectBar
          selecting={selecting}
          onStart={() => setSelecting(true)}
          onCancel={stopSelecting}
          chosen={chosen}
          shown={shown}
          onSelectShown={() =>
            setSelected((prev) => new Set([...prev, ...shown.map((m) => m.id)]))
          }
          onClear={() => setSelected(new Set())}
          onDelete={deleteChosen}
        />
      )}

      {shown.length === 0 ? (
        <p className="py-16 text-center text-sm text-muted-foreground">
          Nothing matches that.
        </p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {shown.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              selectable={selecting}
              selected={selected.has(model.id)}
              onSelectedChange={(next) => toggle(model.id, next)}
            />
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

/**
 * Selecting and deleting in bulk — the registry's one destructive affordance.
 *
 * Behind an explicit mode rather than a checkbox that is always on every card,
 * because the overwhelmingly common reason to be here is to open a model, and a
 * grid permanently dressed for deletion reads as a more dangerous page than it is.
 * A post-train run leaves a dozen models behind, though, and deleting those one
 * detail page at a time is the thing this exists to stop.
 */
function SelectBar({
  selecting,
  onStart,
  onCancel,
  chosen,
  shown,
  onSelectShown,
  onClear,
  onDelete,
}: {
  selecting: boolean
  onStart: () => void
  onCancel: () => void
  chosen: Model[]
  shown: Model[]
  onSelectShown: () => void
  onClear: () => void
  onDelete: () => Promise<void>
}) {
  const [isDeleting, setIsDeleting] = React.useState(false)

  if (!selecting) {
    return (
      <div className="flex justify-end">
        <Button variant="ghost" size="sm" className="gap-2" onClick={onStart}>
          <Trash2 className="size-3.5" />
          Select
        </Button>
      </div>
    )
  }

  const allShownChosen =
    shown.length > 0 && shown.every((m) => chosen.some((c) => c.id === m.id))

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-muted/30 px-4 py-2.5">
      <p className="text-sm tabular-nums">
        {chosen.length === 0
          ? "Nothing selected"
          : `${count(chosen.length, "model")} selected`}
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={allShownChosen ? onClear : onSelectShown}
          disabled={shown.length === 0 || isDeleting}
        >
          {allShownChosen ? "Clear" : `Select all ${shown.length}`}
        </Button>

        <AlertDialog>
          <AlertDialogTrigger
            render={
              <Button
                variant="outline"
                size="sm"
                className="gap-2"
                disabled={chosen.length === 0 || isDeleting}
              >
                <Trash2 className="size-3.5" />
                {isDeleting ? "Deleting…" : `Delete ${chosen.length}`}
              </Button>
            }
          />
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>
                Delete {count(chosen.length, "model")}?
              </AlertDialogTitle>
              <AlertDialogDescription>
                This permanently removes them from the registry and deletes their
                weights from disk. Existing messages are kept but detached. This
                cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>

            {/* Named, not just counted. Selection survives a filter change, so the
                list is the only thing that guarantees nothing is deleted unseen. */}
            <ul className="max-h-48 space-y-1 overflow-y-auto rounded-md border border-border bg-muted/30 px-3 py-2 text-sm">
              {chosen.map((m) => (
                <li key={m.id} className="truncate">
                  {m.name}
                </li>
              ))}
            </ul>

            <AlertDialogFooter>
              <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
              <AlertDialogAction
                disabled={isDeleting}
                className="bg-destructive text-white hover:bg-destructive/90"
                onClick={async () => {
                  setIsDeleting(true)
                  try {
                    await onDelete()
                  } finally {
                    setIsDeleting(false)
                  }
                }}
              >
                {isDeleting ? "Deleting…" : `Delete ${chosen.length}`}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <Button
          variant="ghost"
          size="sm"
          className="gap-2"
          onClick={onCancel}
          disabled={isDeleting}
        >
          <X className="size-3.5" />
          Done
        </Button>
      </div>
    </div>
  )
}

/** "1 model" / "3 models" — the count is always shown, so it never reads "models". */
function count(n: number, noun: string): string {
  return `${n} ${noun}${n === 1 ? "" : "s"}`
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
