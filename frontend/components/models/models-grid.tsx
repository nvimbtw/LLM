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
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import { getToken } from "@/lib/api"
import { deleteModel, useModels, type Model } from "@/hooks/use-models"
import { LoadError } from "@/components/ui/error-state"
import { ModelCard } from "./model-card"
import { ModelsFilter, type Visibility } from "./models-filter"

/** Scopes registry filters and selection to the current authenticated session. */
export function ModelsGrid() {
  const token = useSessionToken()
  return token ? <ModelsGridContent key={token} /> : null
}

/** Owns one session's registry filters, selection and sequential batch deletion. */
function ModelsGridContent() {
  const [refreshKey, setRefreshKey] = React.useState(0)
  const { models, isLoading, error } = useModels(refreshKey)
  const { isAdmin } = useAuth()

  const [visibility, setVisibility] = React.useState<Visibility>("all")
  const [query, setQuery] = React.useState("")
  const [selecting, setSelecting] = React.useState(false)
  const [selected, setSelected] = React.useState<ReadonlySet<string>>(new Set())

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
  }, [models, visibility, query])

  const chosen = React.useMemo(
    () => models.filter((m) => selected.has(m.id)),
    [models, selected]
  )

  /** Updates selection by model identity without dropping selections hidden by a filter. */
  const toggle = React.useCallback((id: string, next: boolean) => {
    setSelected((prev) => {
      const copy = new Set(prev)
      if (next) copy.add(id)
      else copy.delete(id)
      return copy
    })
  }, [])

  /** Exits selection mode and clears the chosen identities. */
  const stopSelecting = React.useCallback(() => {
    setSelecting(false)
    setSelected(new Set())
  }, [])

  /** Deletes chosen models sequentially, stops on account replacement and reports partial failures by name. */
  const deleteChosen = React.useCallback(async () => {
    const failures: string[] = []
    const token = getToken()
    for (const model of chosen) {
      if (getToken() !== token) return
      try {
        await deleteModel(model.id)
      } catch {
        failures.push(model.name)
      }
    }
    if (getToken() !== token) return

    const deleted = chosen.length - failures.length
    if (failures.length === 0) {
      toast.success(`Deleted ${count(deleted, "model")}.`)
    } else if (deleted === 0) {
      toast.error(
        `Couldn't delete ${count(failures.length, "model")}.`,
        failures.join(", ")
      )
    } else {
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
      <LoadError
        title="The models didn't load."
        hint="Nothing has changed — this is a read."
      />
    )
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <Skeleton className="h-8 w-72 rounded-(--radius-control-sm)" />
        <div className="grid gap-x-10 gap-y-4 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-44 rounded-(--radius-sheet)" />
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
      <ModelsFilter
        visibility={visibility}
        onVisibility={setVisibility}
        counts={counts}
        query={query}
        onQuery={setQuery}
        showVisibility={isAdmin}
        action={
          isAdmin &&
          !selecting && (
            <Button variant="text" size="sm" onClick={() => setSelecting(true)}>
              <Trash2 className="size-3.5" />
              Select
            </Button>
          )
        }
      />

      {isAdmin && selecting && (
        <SelectBar
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
        <div className="grid gap-x-10 gap-y-4 sm:grid-cols-2 xl:grid-cols-3">
          {shown.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              selectable={selecting && isAdmin}
              selected={selected.has(model.id)}
              onSelectedChange={(next) => toggle(model.id, next)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

/** Presents selection controls and confirms every chosen model before sequential bulk deletion. */
function SelectBar({
  onCancel,
  chosen,
  shown,
  onSelectShown,
  onClear,
  onDelete,
}: {
  onCancel: () => void
  chosen: Model[]
  shown: Model[]
  onSelectShown: () => void
  onClear: () => void
  onDelete: () => Promise<void>
}) {
  const [isDeleting, setIsDeleting] = React.useState(false)

  const allShownChosen =
    shown.length > 0 && shown.every((m) => chosen.some((c) => c.id === m.id))

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-(--radius-sheet) border border-border bg-muted/30 px-4 py-2.5">
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
                This permanently removes them from the registry and deletes
                their weights from disk. Existing messages are kept but
                detached. This cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>

            <ul className="max-h-48 space-y-1 overflow-y-auto rounded-(--radius-control-sm) border border-border bg-muted/30 px-3 py-2 text-sm">
              {chosen.map((m) => (
                <li key={m.id} className="truncate">
                  {m.name}
                </li>
              ))}
            </ul>

            <AlertDialogFooter>
              <AlertDialogCancel disabled={isDeleting}>
                Cancel
              </AlertDialogCancel>
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
