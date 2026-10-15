"use client"

import { useId } from "react"
import { CurriculumBuilder } from "@/components/training/curriculum-builder"
import { DatasetCompositionChart } from "@/components/training/dataset-composition"
import { DatasetPicker } from "@/components/training/dataset-picker"
import { Checkbox } from "@/components/ui/checkbox"
import { isFineTune, type DatasetInfo, type TrainConfig } from "@/lib/training"

export interface TrainingDataFieldsProps {
  cfg: TrainConfig
  onChange: (cfg: TrainConfig) => void
  disabled: boolean
  datasets: DatasetInfo[]
  loading: boolean
}

interface DataSettingProps {
  label: string
  hint: string
  checked: boolean
  disabled: boolean
  onChange: (on: boolean) => void
}

/** Edits datasets first, then applicable ordering/thinking policies, then displays their selected document composition. */
export function TrainingDataFields({
  cfg,
  onChange,
  disabled,
  datasets,
  loading,
}: TrainingDataFieldsProps) {
  const isSft = isFineTune(cfg.mode)
  const curriculumOn = !isSft && Boolean(cfg.curriculum?.length)
  return (
    <div className="space-y-6">
      {curriculumOn ? (
        <CurriculumBuilder
          variant="inline"
          available={datasets}
          loading={loading}
          value={cfg.curriculum ?? []}
          disabled={disabled}
          onChange={(curriculum) => onChange({ ...cfg, curriculum })}
        />
      ) : (
        <DatasetPicker
          variant="inline"
          available={datasets}
          loading={loading}
          value={cfg.datasets ?? []}
          disabled={disabled}
          emptyHint={
            isSft
              ? "No conversation datasets yet. Pull one from the Data view first."
              : undefined
          }
          onChange={(selection) => onChange({ ...cfg, datasets: selection })}
        />
      )}
      {!isSft && (
        <div
          className="grid gap-x-8 gap-y-4 sm:grid-cols-2"
          aria-label="Training data settings"
        >
          <DataSetting
            label="Shuffle each epoch"
            hint="Reorder documents between passes."
            checked={cfg.shuffle ?? true}
            disabled={disabled}
            onChange={(shuffle) => onChange({ ...cfg, shuffle })}
          />
          <DataSetting
            label="Curriculum"
            hint="Train datasets as ordered stages."
            checked={curriculumOn}
            disabled={disabled}
            onChange={(on) => onChange(withCurriculum(cfg, on))}
          />
        </div>
      )}
      {cfg.mode === "sft" && (
        <DataSetting
          label="Strip thinking"
          hint="Remove reasoning blocks from conversations."
          checked={cfg.strip_thinking ?? false}
          disabled={disabled}
          onChange={(strip_thinking) => onChange({ ...cfg, strip_thinking })}
        />
      )}
      <DatasetCompositionChart
        cfg={isSft ? { ...cfg, curriculum: [] } : cfg}
        datasets={datasets}
        loading={loading}
      />
    </div>
  )
}

/** Seeds curriculum from the flat selection when enabling it, or restores stage one's selection when disabling it. */
function withCurriculum(cfg: TrainConfig, on: boolean): TrainConfig {
  return on
    ? {
        ...cfg,
        curriculum: [{ datasets: cfg.datasets ?? [], epochs: cfg.epochs }],
      }
    : {
        ...cfg,
        curriculum: [],
        datasets: cfg.curriculum?.[0]?.datasets ?? cfg.datasets ?? [],
      }
}

/** Edits a boolean data policy through a labeled checkbox with a short explanation. */
function DataSetting({
  label,
  hint,
  checked,
  disabled,
  onChange,
}: DataSettingProps) {
  const hintId = useId()
  return (
    <Checkbox
      variant="block"
      aria-label={label}
      aria-describedby={hintId}
      checked={checked}
      disabled={disabled}
      onCheckedChange={(on) => onChange(on === true)}
    >
      <span className="min-w-0 text-sm">
        {label}
        <span
          id={hintId}
          className="mt-0.5 block text-xs text-muted-foreground"
        >
          {hint}
        </span>
      </span>
    </Checkbox>
  )
}
