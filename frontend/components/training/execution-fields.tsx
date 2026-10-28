"use client"

import { useId } from "react"
import type { TrainConfig } from "@/lib/training"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export interface TrainingExecutionFieldsProps {
  config: TrainConfig
  disabled: boolean
  onChange: (config: TrainConfig) => void
}

/** Edits numerical execution controls separately from recipe parameters; the engine preview reports actual device/precision and final preflight remains authoritative. */
export function TrainingExecutionFields({
  config,
  disabled,
  onChange,
}: TrainingExecutionFieldsProps) {
  const id = useId()
  return (
    <fieldset
      disabled={disabled}
      className="grid gap-x-8 gap-y-4 py-4 sm:grid-cols-2"
    >
      <legend className="sr-only">Numerical execution</legend>
      <label
        className="flex min-w-0 items-center justify-between gap-3 text-sm"
        htmlFor={`${id}-precision`}
      >
        Precision
        <Select
          disabled={disabled}
          value={config.precision ?? "bf16"}
          onValueChange={(precision) => {
            if (
              precision === "fp32" ||
              precision === "fp16" ||
              precision === "bf16"
            )
              onChange({ ...config, precision })
          }}
        >
          <SelectTrigger
            variant="inline"
            className="h-9 w-auto"
            id={`${id}-precision`}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {["fp32", "fp16", "bf16"].map((precision) => (
              <SelectItem key={precision} value={precision}>
                {precision}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </label>
      <label className="flex min-w-0 items-center justify-between gap-3 text-sm">
        Numerical seed
        <Input
          variant="inline"
          className="h-9 w-24 text-right tabular-nums"
          aria-label="Numerical seed"
          type="number"
          step={1}
          min={0}
          max={Number.MAX_SAFE_INTEGER}
          value={Number.isFinite(config.seed ?? 0) ? (config.seed ?? 0) : ""}
          onChange={(event) =>
            onChange({
              ...config,
              seed:
                event.target.value === ""
                  ? Number.NaN
                  : Number(event.target.value),
            })
          }
        />
      </label>
      <label className="flex items-center justify-between gap-3 text-sm">
        <span>Allow TF32 on supported hardware</span>
        <Switch
          checked={config.allow_tf32 ?? true}
          disabled={disabled}
          onCheckedChange={(allow_tf32) => onChange({ ...config, allow_tf32 })}
        />
      </label>
      <p className="text-xs text-muted-foreground">
        The current device determines effective precision. A recipe profile read
        shows the values that will be checked.
      </p>
    </fieldset>
  )
}
