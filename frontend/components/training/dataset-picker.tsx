"use client"

import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import type { DatasetInfo, DatasetSelection } from "@/lib/training"

// The two controls a run and a vocabulary build both need: which corpora, and
// how large a vocabulary. They lived in `params-rail.tsx` until the tokenizer
// composer wanted the same two, which is when a local stops being local.

// Exponent bounds: 2^0 (1) up through 2^13 (8192) is the default ceiling; fields
// that go larger (vocab) pass their own `maxExp`.
export const EXP_MIN = 0
export const EXP_MAX = 13

// A power-of-two field, edited as its exponent: the input holds `k` and the value
// sent upstream is 2^k. This is what makes cycling work — the native ±1 step (from
// the spinner buttons, the wheel, or ↑/↓) moves the exponent by one, which
// doubles/halves the value (… 16 · 32 · 64 …) instead of walking 65 · 66 · 67. A
// muted "= 64" shows the resolved value so the exponent is never cryptic.
export function Pow2Input({
  value,
  disabled,
  maxExp = EXP_MAX,
  onChange,
}: {
  value: number
  disabled: boolean
  maxExp?: number
  onChange: (n: number) => void
}) {
  const exp = Math.max(0, Math.round(Math.log2(value > 0 ? value : 1)))
  const resolved = 2 ** Math.min(maxExp, Math.max(EXP_MIN, exp))

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

// The dataset picker: one checkbox row per selectable dataset. Checking one adds
// it to the run; the count field then dials how many of its documents to take
// (blank/0 = all), with the total shown as "/ <documents>". Reads as an
// extension of the spec-sheet rail — a hairline list, tabular figures, muted
// labels — rather than a separate form.
//
// Serves both kinds. A selection carries the dataset's id: the engine selects
// by id because a name is something a person can rename in the data view
// without meaning to change what a run trains on.
export function DatasetPicker({
  available,
  loading,
  value,
  disabled,
  onChange,
  emptyHint,
}: {
  available: DatasetInfo[]
  loading: boolean
  value: DatasetSelection[]
  disabled: boolean
  onChange: (next: DatasetSelection[]) => void
  emptyHint?: React.ReactNode
}) {
  const selected = new Map(value.map((d) => [d.id, d.count]))

  const toggle = (dataset: DatasetInfo, on: boolean) => {
    // The name is stored alongside the id so a saved preset still reads as
    // something; the run resolves the id and takes the current name from there.
    if (on) onChange([...value, { id: dataset.id, name: dataset.name, count: 0 }])
    else onChange(value.filter((d) => d.id !== dataset.id))
  }

  const setCount = (id: string, count: number, max: number) => {
    const clamped = Number.isFinite(count)
      ? Math.min(max, Math.max(0, Math.floor(count)))
      : 0
    onChange(value.map((d) => (d.id === id ? { ...d, count: clamped } : d)))
  }

  const totalDocuments = value.reduce((sum, d) => {
    const info = available.find((a) => a.id === d.id)
    if (!info) return sum
    return sum + (d.count > 0 ? Math.min(d.count, info.documents) : info.documents)
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
        <p className="rounded-lg border border-dashed px-2.5 py-3 text-center text-xs text-muted-foreground">
          Loading datasets…
        </p>
      ) : available.length === 0 ? (
        <p className="rounded-lg border border-dashed px-2.5 py-3 text-center text-xs text-muted-foreground">
          {emptyHint ?? <>No datasets yet. Pull one from the Data view first.</>}
        </p>
      ) : (
        <ul className="divide-y rounded-lg border">
          {available.map((d) => {
            const checked = selected.has(d.id)
            const count = selected.get(d.id) ?? 0
            const rowId = `dataset-${d.id}`
            return (
              <li
                key={d.id}
                className="flex items-center justify-between gap-3 px-2.5 py-2"
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
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
