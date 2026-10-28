"use client"

import { ResourceList, ResourceRow } from "@/components/ui/resource-list"
import { Checkbox } from "@/components/ui/checkbox"
import { useId } from "react"
import { Input, type InputProps } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import type { DatasetInfo, DatasetSelection } from "@/lib/training"

// The two controls a run and a vocabulary build both need: which corpora, and
// how large a vocabulary. They lived in `params-rail.tsx` until the tokenizer
// composer wanted the same two, which is when a local stops being local.

// Exponent bounds: 2^0 (1) up through 2^13 (8192) is the default ceiling; fields
// that go larger (vocab) pass their own `maxExp`.
export const EXP_MIN = 0
export const EXP_MAX = 13

export interface Pow2InputProps {
  value: number
  disabled: boolean
  maxExp?: number
  ariaLabel?: string
  variant?: InputProps["variant"]
  onChange: (n: number) => void
}

/** Edits a bounded exponent and emits its power-of-two value, displaying the resolved value beside it. */
export function Pow2Input({
  value,
  disabled,
  maxExp = EXP_MAX,
  ariaLabel,
  variant = "default",
  onChange,
}: Pow2InputProps) {
  const exp = Math.max(0, Math.round(Math.log2(value > 0 ? value : 1)))
  const resolved = 2 ** Math.min(maxExp, Math.max(EXP_MIN, exp))

  /** Clamps an edited exponent to the supported range and emits its power-of-two value. */
  const setExp = (raw: number) => {
    if (!Number.isFinite(raw)) return
    const k = Math.min(maxExp, Math.max(EXP_MIN, Math.round(raw)))
    onChange(2 ** k)
  }

  return (
    <span className="flex items-center justify-end gap-1.5">
      <span className="text-xs text-muted-foreground tabular-nums">
        = {resolved}
      </span>
      <span className="text-sm text-muted-foreground select-none">2^</span>
      <Input
        variant={variant}
        aria-label={ariaLabel}
        type="number"
        step={1}
        min={EXP_MIN}
        max={maxExp}
        value={String(exp)}
        disabled={disabled}
        onChange={(e) => setExp(Number(e.target.value))}
        className="h-8 w-16 text-right tabular-nums"
      />
    </span>
  )
}

export interface DatasetPickerProps {
  available: DatasetInfo[]
  loading: boolean
  value: DatasetSelection[]
  disabled: boolean
  onChange: (next: DatasetSelection[]) => void
  emptyHint?: React.ReactNode
  variant?: InputProps["variant"]
}

/** Selects datasets by stable ID and bounds document limits to catalogue totals; zero means all documents. IDs are unique to each mounted picker. */
export function DatasetPicker({
  available,
  loading,
  value,
  disabled,
  onChange,
  emptyHint,
  variant = "default",
}: DatasetPickerProps) {
  const pickerId = useId()
  const selected = new Map(value.map((d) => [d.id, d.count]))

  /** Adds or removes a dataset selection, retaining its name for saved presets. */
  const toggle = (dataset: DatasetInfo, on: boolean) => {
    if (on)
      onChange([...value, { id: dataset.id, name: dataset.name, count: 0 }])
    else onChange(value.filter((d) => d.id !== dataset.id))
  }

  /** Bounds the edited document limit to this dataset and updates its selection. */
  const setCount = (id: string, count: number, max: number) => {
    const clamped = Number.isFinite(count)
      ? Math.min(max, Math.max(0, Math.floor(count)))
      : 0
    onChange(value.map((d) => (d.id === id ? { ...d, count: clamped } : d)))
  }

  const totalDocuments = value.reduce((sum, d) => {
    const info = available.find((a) => a.id === d.id)
    if (!info) return sum
    return (
      sum + (d.count > 0 ? Math.min(d.count, info.documents) : info.documents)
    )
  }, 0)

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Datasets</span>
        {value.length > 0 && (
          <span className="text-xs text-muted-foreground tabular-nums">
            {totalDocuments.toLocaleString()} documents
          </span>
        )}
      </div>

      {loading ? (
        <p className="py-3 text-xs text-muted-foreground">Loading datasets…</p>
      ) : available.length === 0 ? (
        <p className="py-3 text-xs text-muted-foreground">
          {emptyHint ?? (
            <>No datasets yet. Pull one from the Data view first.</>
          )}
        </p>
      ) : (
        <ResourceList density="compact">
          {available.map((d) => {
            const checked = selected.has(d.id)
            const count = selected.get(d.id) ?? 0
            const rowId = `${pickerId}-dataset-${d.id}`
            return (
              <ResourceRow
                density="compact"
                key={d.id}
                className="flex items-center justify-between gap-3"
              >
                <label
                  htmlFor={rowId}
                  className={cn(
                    "flex min-w-0 items-center gap-2",
                    !disabled && "cursor-pointer"
                  )}
                >
                  <Checkbox
                    id={rowId}
                    checked={checked}
                    disabled={disabled}
                    onCheckedChange={(on) => toggle(d, Boolean(on))}
                  />
                  <span className="truncate text-sm">{d.name}</span>
                </label>

                <span className="flex w-32 shrink-0 items-center justify-end gap-1.5 tabular-nums">
                  {checked ? (
                    <>
                      <Input
                        variant={variant}
                        aria-label={`${d.name} document limit`}
                        type="number"
                        min={0}
                        max={d.documents}
                        step={1}
                        value={count === 0 ? "" : String(count)}
                        placeholder="all"
                        disabled={disabled}
                        onChange={(e) =>
                          setCount(d.id, Number(e.target.value), d.documents)
                        }
                        className="h-8 w-16 text-right tabular-nums"
                      />
                      <span className="w-14 text-xs text-muted-foreground">
                        / {d.documents.toLocaleString()}
                      </span>
                    </>
                  ) : (
                    <span className="pr-1 text-xs text-muted-foreground">
                      {d.documents.toLocaleString()} docs
                    </span>
                  )}
                </span>
              </ResourceRow>
            )
          })}
        </ResourceList>
      )}
    </div>
  )
}
