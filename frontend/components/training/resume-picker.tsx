"use client"

import * as React from "react"

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  FINISHED_WEIGHTS,
  FROM_SCRATCH,
} from "@/components/training/run-config"
import { useCheckpointParent } from "@/hooks/use-checkpoint-parent"
import { useCheckpoints } from "@/hooks/use-checkpoints"
import { cn } from "@/lib/utils"
import type { Model } from "@/hooks/use-models"

/**
 * What a pretrain starts from: nothing, a finished model, or one of that
 * model's checkpoints.
 *
 * Two selects rather than one flat list, and that is the whole design. A run
 * leaves ~10 checkpoints, so a single list over every model and every
 * intermediate would be the flood the engine's `checkpoints=false` default
 * exists to prevent — the second select is only ever one model's, asked for by
 * name. It also matches how the two ids relate: a checkpoint is *of* a model,
 * and picking the model first is picking the run.
 *
 * Rendered by both arrangements of the composer — the board and the follow-up
 * rail — so neither can drift from the other on what may be resumed.
 */
export interface ResumePickerProps {
  /** `TrainConfig.resume_from`: a model id, a checkpoint id, or absent. */
  value: string | undefined
  onChange: (next: string | undefined) => void
  /** Finished models only — `useModels` never lists intermediates. */
  models: Model[]
  modelsLoading: boolean
  disabled: boolean
  /**
   * The rail's sizing rather than the board's — the small trigger and the small
   * radius its neighbours take. A prop and not a class, so the two arrangements
   * cannot end up drawing the same control at two different radii.
   */
  compact?: boolean
}

export function ResumePicker({
  value,
  onChange,
  models,
  modelsLoading,
  disabled,
  compact = false,
}: ResumePickerProps) {
  const size = compact ? "sm" : "default"
  const triggerClassName = compact ? "h-8 w-44" : "h-10 w-full"
  // Which model's intermediates the second select offers, for the one case the
  // answer is not already in hand: an id the model list does not hold, arriving
  // from a deep link off its parent's page or typed into the bar. An id that
  // resolves to nothing leaves the second select unopened — the config still
  // carries it, and the engine is the one entitled to refuse a run for it.
  const { parentId, record } = useCheckpointParent(value, models)

  const { checkpoints, isLoading: checkpointsLoading } = useCheckpoints(
    parentId || null
  )

  const pickParent = (next: string | null) => {
    onChange(!next || next === FROM_SCRATCH ? undefined : next)
  }

  const pickPoint = (next: string | null) => {
    if (!next || next === FINISHED_WEIGHTS) {
      onChange(parentId)
      return
    }
    // Recorded rather than looked up: this is the one moment the pairing is known
    // for certain, and writing it here is what stops the hook above spending a
    // request to re-learn what was just clicked.
    record(next, parentId)
    onChange(next)
  }

  return (
    <div
      className={cn(
        "flex flex-col gap-1.5",
        compact ? "items-end" : "items-stretch"
      )}
    >
      <Select
        value={parentId || FROM_SCRATCH}
        onValueChange={pickParent}
        disabled={disabled || modelsLoading}
      >
        <SelectTrigger size={size} className={triggerClassName}>
          <SelectValue>
            {(selected: string | null) =>
              !selected || selected === FROM_SCRATCH
                ? "Start from scratch"
                : (models.find((m) => m.id === selected)?.name ?? selected)
            }
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {/* A real option, not a placeholder: starting from random weights is
              what every pretrain did before a run could be picked up again, and
              it stays the default. */}
          <SelectItem value={FROM_SCRATCH}>Start from scratch</SelectItem>
          {models.map((m) => (
            <SelectItem key={m.id} value={m.id}>
              {m.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {parentId && !checkpointsLoading && checkpoints.length > 0 && (
        <Select
          value={value === parentId ? FINISHED_WEIGHTS : (value ?? "")}
          onValueChange={pickPoint}
          disabled={disabled}
        >
          <SelectTrigger size={size} className={triggerClassName}>
            <SelectValue>
              {(selected: string | null) => {
                if (!selected || selected === FINISHED_WEIGHTS)
                  return "Its finished weights"
                const found = checkpoints.find((c) => c.id === selected)
                return found ? `#${found.index} · step ${found.step}` : selected
              }}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={FINISHED_WEIGHTS}>
              Its finished weights
            </SelectItem>
            {checkpoints.map((c) => (
              <SelectItem key={c.id} value={c.id}>
                #{c.index} · step {c.step}
                {c.loss === null ? "" : ` · loss ${c.loss.toFixed(3)}`}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  )
}
