"use client"

import * as React from "react"
import { ChevronDown, ChevronRight, ChevronUp, Plus, X } from "lucide-react"

import { DatasetPicker } from "@/components/training/dataset-picker"
import { Button } from "@/components/ui/button"
import { Input, type InputProps } from "@/components/ui/input"
import type { CurriculumStage, DatasetInfo } from "@/lib/training"

export interface CurriculumBuilderProps {
  available: DatasetInfo[]
  loading: boolean
  value: CurriculumStage[]
  disabled: boolean
  onChange: (next: CurriculumStage[]) => void
  variant?: InputProps["variant"]
}

/** Edits an ordered curriculum with per-stage datasets, names and epochs; newly added stages open for editing. */
export function CurriculumBuilder({
  available,
  loading,
  value,
  disabled,
  onChange,
  variant = "default",
}: CurriculumBuilderProps) {
  const [collapsed, setCollapsed] = React.useState<Set<number>>(new Set())

  /** Toggles the visual expansion of the supplied stage index. */
  const toggleCollapsed = (i: number) =>
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })

  /** Applies a patch to the supplied stage and emits the updated curriculum. */
  const update = (i: number, patch: Partial<CurriculumStage>) =>
    onChange(value.map((s, idx) => (idx === i ? { ...s, ...patch } : s)))

  /** Removes the supplied stage index from the curriculum. */
  const remove = (i: number) => onChange(value.filter((_, idx) => idx !== i))

  /** Moves a stage one place when the requested neighbor exists. */
  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir
    if (j < 0 || j >= value.length) return
    const next = [...value]
    ;[next[i], next[j]] = [next[j], next[i]]
    onChange(next)
  }

  /** Appends an empty stage with one epoch. */
  const add = () => onChange([...value, { datasets: [], epochs: 1 }])

  /** Formats a collapsed stage from its selected datasets and epoch count. */
  const summary = (stage: CurriculumStage) => {
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
          <div key={i} className="border-t border-border first:border-t-0">
            <div className="flex items-center justify-between gap-2 py-3">
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
                  variant="text"
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
                  variant="text"
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
                  variant="text"
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
              <div className="space-y-3 pb-3">
                <Input
                  variant={variant}
                  aria-label={`Stage ${i + 1} name`}
                  value={stage.name ?? ""}
                  onChange={(e) =>
                    update(i, { name: e.target.value || undefined })
                  }
                  placeholder={`Stage ${i + 1} label (optional)`}
                  disabled={disabled}
                  className="h-8 text-sm"
                />

                <DatasetPicker
                  variant={variant}
                  available={available}
                  loading={loading}
                  value={stage.datasets}
                  disabled={disabled}
                  onChange={(datasets) => update(i, { datasets })}
                />

                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm text-muted-foreground">Epochs</span>
                  <Input
                    variant={variant}
                    aria-label={`Stage ${i + 1} epochs`}
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
        variant="text"
        className="px-0 text-muted-foreground hover:bg-transparent hover:text-foreground"
        disabled={disabled}
        onClick={add}
      >
        <Plus className="size-4" /> Add stage
      </Button>
    </div>
  )
}
