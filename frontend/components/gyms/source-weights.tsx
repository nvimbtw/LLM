"use client"

import { Input } from "@/components/ui/input"
import { type GymTaskType } from "@/lib/gyms"

interface SourceWeightsProps {
  tasks: GymTaskType[]
  selected: string[]
  weights: Record<string, number>
  onChange: (weights: Record<string, number>) => void
}

/** Edits bounded source weights for the currently selected gym tasks. */
export function SourceWeights({
  tasks,
  selected,
  weights,
  onChange,
}: SourceWeightsProps) {
  const sources = [
    ...new Set(
      tasks
        .filter((task) => selected.includes(task.id))
        .map((task) => task.gym_id)
    ),
  ]
  return (
    <fieldset className="space-y-3">
      <legend className="eyebrow text-muted-foreground">Source weights</legend>
      <p className="text-xs text-muted-foreground">
        Equal weights give each gym an equal share. A weight of 2 draws twice as
        often as 1.
      </p>
      <div className="flex flex-wrap gap-4">
        {sources.map((source) => (
          <label key={source} className="space-y-1">
            <span className="block eyebrow text-muted-foreground">
              {source}
            </span>
            <Input
              className="w-24"
              type="number"
              min={0.01}
              max={1_000_000}
              step={0.1}
              value={weights[source] ?? 1}
              onChange={(event) => {
                const value = event.target.valueAsNumber
                onChange({
                  ...weights,
                  [source]: Number.isFinite(value)
                    ? Math.max(0.01, Math.min(1_000_000, value))
                    : 1,
                })
              }}
            />
          </label>
        ))}
      </div>
    </fieldset>
  )
}
