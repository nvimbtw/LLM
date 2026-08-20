"use client"

import * as React from "react"
import { ChevronDown, ChevronRight, ChevronUp, Plus, X } from "lucide-react"

import { DatasetPicker } from "@/components/training/dataset-picker"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import type { CurriculumStage, DatasetInfo } from "@/lib/training"

// The curriculum builder: an ordered list of stages, each its own dataset
// selection + epoch count, trained in sequence. Reuses DatasetPicker per stage so
// a stage reads exactly like the flat picker. Stages can be reordered (the order
// IS the curriculum) and removed; "Add stage" appends an empty one.
export function CurriculumBuilder({
  available,
  loading,
  value,
  disabled,
  onChange,
}: {
  available: DatasetInfo[]
  loading: boolean
  value: CurriculumStage[]
  disabled: boolean
  onChange: (next: CurriculumStage[]) => void
}) {
  // Collapsed stages, tracked by index — a purely visual state so the rail can be
  // kept compact. A freshly-added stage starts expanded so it can be configured.
  const [collapsed, setCollapsed] = React.useState<Set<number>>(new Set())

  const toggleCollapsed = (i: number) =>
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })

  const update = (i: number, patch: Partial<CurriculumStage>) =>
    onChange(value.map((s, idx) => (idx === i ? { ...s, ...patch } : s)))

  const remove = (i: number) => onChange(value.filter((_, idx) => idx !== i))

  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir
    if (j < 0 || j >= value.length) return
    const next = [...value]
    ;[next[i], next[j]] = [next[j], next[i]]
    onChange(next)
  }

  const add = () => onChange([...value, { datasets: [], epochs: 1 }])

  // One-line summary of a collapsed stage: its datasets and epoch count.
  const summary = (stage: CurriculumStage) => {
    // A selection saved before the dataset list loaded carries only an id; show
    // the head of it rather than the word "undefined".
    const names = stage.datasets.map((d) => d.name || d.id.slice(0, 8))
    const datasets = names.length > 0 ? names.join(", ") : "no datasets"
    return `${datasets} · ${stage.epochs} ep`
  }

  return (
    <div className="space-y-3">
      <span className="text-xs text-muted-foreground">Curriculum stages</span>

      {value.map((stage, i) => {
        const isCollapsed = collapsed.has(i)
        const title = stage.name?.trim() || `Stage ${i + 1}`
        return (
          <div key={i} className="rounded-(--radius-sheet) border">
            {/* Header: click the title to collapse/expand; reorder + remove stay
                on the right. Kept compact so a long curriculum doesn't dominate. */}
            <div className="flex items-center justify-between gap-2 p-2.5">
              <button
                type="button"
                onClick={() => toggleCollapsed(i)}
                aria-expanded={!isCollapsed}
                className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
              >
                {isCollapsed ? (
                  <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
                ) : (
                  <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
                )}
                <span className="min-w-0 truncate">
                  <span className="text-xs font-medium tabular-nums">
                    {i + 1}.
                  </span>{" "}
                  <span className="text-sm">{title}</span>
                  {isCollapsed && (
                    <span className="ml-1.5 truncate text-xs text-muted-foreground">
                      {summary(stage)}
                    </span>
                  )}
                </span>
              </button>
              <div className="flex shrink-0 items-center gap-0.5">
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className="size-7"
                  disabled={disabled || i === 0}
                  onClick={() => move(i, -1)}
                  aria-label="Move stage up"
                >
                  <ChevronUp className="size-4" />
                </Button>
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className="size-7"
                  disabled={disabled || i === value.length - 1}
                  onClick={() => move(i, 1)}
                  aria-label="Move stage down"
                >
                  <ChevronDown className="size-4" />
                </Button>
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className="size-7 text-muted-foreground hover:text-destructive"
                  disabled={disabled}
                  onClick={() => remove(i)}
                  aria-label="Remove stage"
                >
                  <X className="size-4" />
                </Button>
              </div>
            </div>

            {!isCollapsed && (
              <div className="space-y-2.5 border-t p-2.5">
                <Input
                  value={stage.name ?? ""}
                  onChange={(e) =>
                    update(i, { name: e.target.value || undefined })
                  }
                  placeholder={`Stage ${i + 1} label (optional)`}
                  disabled={disabled}
                  className="h-8 text-sm"
                />

                <DatasetPicker
                  available={available}
                  loading={loading}
                  value={stage.datasets}
                  disabled={disabled}
                  onChange={(datasets) => update(i, { datasets })}
                />

                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm text-muted-foreground">Epochs</span>
                  <Input
                    type="number"
                    min={1}
                    step={1}
                    value={String(stage.epochs)}
                    onChange={(e) =>
                      update(i, { epochs: Number(e.target.value) })
                    }
                    disabled={disabled}
                    className="h-8 w-24 text-right tabular-nums"
                  />
                </div>
              </div>
            )}
          </div>
        )
      })}

      <Button
        type="button"
        size="sm"
        variant="outline"
        className="w-full"
        disabled={disabled}
        onClick={add}
      >
        <Plus className="size-4" /> Add stage
      </Button>
    </div>
  )
}
