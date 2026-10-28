"use client"

import * as React from "react"

import { Pow2Input } from "@/components/training/dataset-picker"
import { PresetControls } from "@/components/training/preset-controls"
import { ResumePicker } from "@/components/training/resume-picker"
import { BuildSheet } from "@/components/training/build-sheet"
import { TrainingDataFields } from "@/components/training/training-data-fields"
import { ArchitectureFlow } from "@/components/training/architecture-flow"
import { InheritedArchitecture } from "@/components/training/inherited-architecture"
import { ArchitectureChoice } from "@/components/training/architecture-choice"
import { TrainingExecutionFields } from "@/components/training/execution-fields"
import type { TrainingArchitectureCheck } from "@/lib/training-selection"
import { compact } from "@/components/training/stat-run"
import { estimateTrainingParameters } from "@/lib/training-architecture"
import {
  LEARN_WITH_RUN,
  STRING_FIELDS,
  configForMode,
  fieldsFor,
  hidesTokenizer,
  validate,
  type Field,
} from "@/components/training/run-config"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { EditableTitle } from "@/components/ui/editable-title"
import { Panel } from "@/components/ui/panel"
import { PageHeader } from "@/components/ui/page-header"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useDatasets } from "@/hooks/use-datasets"
import { useModels } from "@/hooks/use-models"
import { useTokenizers } from "@/hooks/use-tokenizers"
import { isFineTune, type TrainConfig, type TrainMode } from "@/lib/training"

export interface TrainBoardProps {
  cfg: TrainConfig
  onChange: (cfg: TrainConfig) => void
  disabled: boolean
  onStart: (cfg: TrainConfig) => void
  /** Advances when a model is registered, so the base picker refetches. */
  modelsRefreshKey?: unknown
  /** The follow-up list. Part of what the button submits, so it sits above it. */
  footer?: React.ReactNode
  /** Browser draft status and discard action beside the page title. */
  draftStatus?: React.ReactNode
}

/** Edits a run as page-native fields and queues its effective config after shared validation. Fine-tunes default to the first available base. */
export function TrainBoard({
  cfg,
  onChange,
  disabled,
  onStart,
  modelsRefreshKey,
  footer,
  draftStatus,
}: TrainBoardProps) {
  const { models, isLoading: modelsLoading } = useModels(modelsRefreshKey)
  const { datasets, isLoading: datasetsLoading } = useDatasets(
    isFineTune(cfg.mode)
  )
  const { tokenizers, isLoading: tokenizersLoading } = useTokenizers()

  const isSft = isFineTune(cfg.mode)
  const isCycleMode = cfg.mode === "kto" || cfg.mode === "raft"
  const { arch, run } = fieldsFor(cfg)

  const baseModelId = cfg.base_model_id || models[0]?.id
  const effective: TrainConfig = isSft
    ? { ...cfg, base_model_id: baseModelId }
    : cfg
  const [architectureCheck, setArchitectureCheck] =
    React.useState<TrainingArchitectureCheck | null>(null)
  const errors = validate(effective, false, architectureCheck)
  const vocabularySize = cfg.tokenizer_id
    ? (tokenizers.find((tokenizer) => tokenizer.id === cfg.tokenizer_id)
        ?.vocabSize ?? null)
    : cfg.max_vocab
  const parameterCount = cfg.architecture
    ? null
    : estimateTrainingParameters(cfg, vocabularySize)
  const inheritedModelId = isSft ? baseModelId : cfg.resume_from

  /** Parses a text edit into the parent config, preserving literal values for string fields. */
  const set = (key: keyof TrainConfig, value: string) =>
    onChange({ ...cfg, [key]: STRING_FIELDS.has(key) ? value : Number(value) })

  /** Stores a resolved numeric field value in the parent config. */
  const setNumber = (key: keyof TrainConfig, value: number) =>
    onChange({ ...cfg, [key]: value })

  /** Changes mode and adjusts only learning rates that still match a mode default. */
  const setMode = (mode: TrainMode) => onChange(configForMode(cfg, mode))

  /** Prevents browser submission and starts the validated effective run. */
  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (errors.length > 0) return
    onStart(effective)
  }

  return (
    <form onSubmit={submit} className="space-y-10">
      <PageHeader
        back={{ href: "/admin", label: "Admin" }}
        className="mb-0 border-b-0 pb-0"
        contentClassName="min-w-48 flex-1"
        description={draftStatus}
        title={
          <>
            <span className="sr-only">Train a model</span>
            <EditableTitle
              aria-label="Model name"
              value={cfg.model_name}
              onChange={(event) => set("model_name", event.target.value)}
              placeholder="Model name"
              disabled={disabled}
            />
          </>
        }
        action={
          <PresetControls
            currentConfig={effective}
            disabled={disabled}
            onLoad={onChange}
            modelsRefreshKey={modelsRefreshKey}
          />
        }
      />
      <div className="border-b border-border pt-5">
        <ModeToggle
          mode={cfg.mode}
          isCycleMode={isCycleMode}
          disabled={disabled}
          onChange={setMode}
        />
      </div>
      <section aria-label="Run identity">
        <div className="space-y-5">
          {isSft ? (
            <label className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)] items-center gap-6">
              <FieldLabel>Base model</FieldLabel>
              <div className="min-w-0">
                <Select
                  value={baseModelId ?? ""}
                  onValueChange={(v) => set("base_model_id", v ?? "")}
                  disabled={disabled || modelsLoading || models.length === 0}
                >
                  <SelectTrigger
                    variant="inline"
                    aria-label="Base model"
                    className="ml-auto h-9 max-w-full"
                  >
                    <SelectValue
                      className="text-right"
                      placeholder={modelsLoading ? "Loading…" : "No models"}
                    >
                      {(value: string | null) =>
                        models.find((m) => m.id === value)?.name ?? ""
                      }
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent
                    alignItemWithTrigger={false}
                    align="end"
                    className="w-64"
                  >
                    {models.map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        {m.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </label>
          ) : !hidesTokenizer(cfg) ? (
            <label className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)] items-center gap-6">
              <FieldLabel>Tokenizer</FieldLabel>
              <div className="min-w-0">
                <Select
                  value={cfg.tokenizer_id ?? LEARN_WITH_RUN}
                  onValueChange={(v) =>
                    onChange({
                      ...cfg,
                      tokenizer_id: !v || v === LEARN_WITH_RUN ? undefined : v,
                    })
                  }
                  disabled={disabled || tokenizersLoading}
                >
                  <SelectTrigger
                    variant="inline"
                    aria-label="Tokenizer"
                    className="ml-auto h-9 max-w-full"
                  >
                    <SelectValue className="text-right">
                      {(value: string | null) =>
                        !value || value === LEARN_WITH_RUN
                          ? "Learn one with the run"
                          : (tokenizers.find((t) => t.id === value)?.name ??
                            value)
                      }
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent
                    alignItemWithTrigger={false}
                    align="end"
                    className="w-64"
                  >
                    <SelectItem value={LEARN_WITH_RUN}>
                      Learn one with the run
                    </SelectItem>
                    {tokenizers
                      .filter((t) => t.hasVocab && !t.implicit)
                      .map((t) => (
                        <SelectItem key={t.id} value={t.id}>
                          {t.name}
                          {t.vocabSize
                            ? ` · ${t.vocabSize.toLocaleString()}`
                            : ""}
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
              </div>
            </label>
          ) : null}

          {!isSft && (
            <div className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)] items-center gap-6">
              <FieldLabel>Continue from</FieldLabel>
              <ResumePicker
                value={cfg.resume_from}
                onChange={(resume_from) => onChange({ ...cfg, resume_from })}
                models={models}
                modelsLoading={modelsLoading}
                disabled={disabled}
                variant="inline"
              />
            </div>
          )}
        </div>
      </section>
      <Panel variant="open" title="Training data">
        <TrainingDataFields
          cfg={cfg}
          onChange={onChange}
          disabled={disabled}
          datasets={datasets}
          loading={datasetsLoading}
        />
      </Panel>
      {!isSft && !cfg.resume_from ? (
        <Panel
          variant="open"
          title="Architecture"
          bodyClassName="p-0"
          descriptionLayout="stacked"
          description={
            cfg.architecture
              ? "Parameter count is measured during numerical validation."
              : `${parameterCount === null ? "—" : `≈ ${compact(parameterCount)}`} parameters`
          }
        >
          <ArchitectureChoice
            config={effective}
            onChange={onChange}
            disabled={disabled}
            onValidity={setArchitectureCheck}
          />
          <FieldGrid
            fields={arch}
            cfg={cfg}
            disabled={disabled}
            onText={set}
            onNumber={setNumber}
          />
          {!cfg.architecture && <ArchitectureFlow view="training" cfg={cfg} />}
        </Panel>
      ) : inheritedModelId ? (
        <InheritedArchitecture cfg={effective} modelId={inheritedModelId} />
      ) : null}
      <Panel variant="open" title="Optimization" bodyClassName="p-0">
        <FieldGrid
          fields={run}
          cfg={cfg}
          disabled={disabled}
          onText={set}
          onNumber={setNumber}
        />
        <TrainingExecutionFields
          config={effective}
          disabled={disabled}
          onChange={onChange}
        />
      </Panel>

      {footer}
      <BuildSheet
        cfg={effective}
        baseName={models.find((m) => m.id === baseModelId)?.name}
        resumeName={models.find((m) => m.id === cfg.resume_from)?.name}
        datasets={datasets}
        errors={errors}
        disabled={disabled}
      />
    </form>
  )
}

/** Pairs a field's visible label with an optional explanation beneath it. */
function FieldLabel({
  children,
  hint,
}: {
  children: React.ReactNode
  hint?: string
}) {
  return (
    <span className="block min-w-0">
      <span className="text-sm">{children}</span>
      {hint && (
        <span className="mt-0.5 block text-xs text-muted-foreground/80 normal-case">
          {hint}
        </span>
      )}
    </span>
  )
}

/** Renders the applicable numeric settings as inline values beside labels, stacking columns on narrow screens. */
function FieldGrid({
  fields,
  cfg,
  disabled,
  onText,
  onNumber,
}: {
  fields: Field[]
  cfg: TrainConfig
  disabled: boolean
  onText: (key: keyof TrainConfig, value: string) => void
  onNumber: (key: keyof TrainConfig, value: number) => void
}) {
  return (
    <div className="grid gap-x-8 sm:grid-cols-2">
      {fields.map((f) => (
        <label
          key={f.key}
          className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-3 py-4"
        >
          <FieldLabel hint={f.hint}>{f.label}</FieldLabel>
          {f.pow2 ? (
            <Pow2Input
              value={cfg[f.key] as number}
              disabled={disabled}
              maxExp={f.maxExp}
              variant="inline"
              ariaLabel={`${f.label} exponent`}
              onChange={(n) => onNumber(f.key, n)}
            />
          ) : (
            <Input
              variant="inline"
              aria-label={f.label}
              type="number"
              step={f.step ?? 1}
              value={String(cfg[f.key] ?? "")}
              onChange={(e) => onText(f.key, e.target.value)}
              disabled={disabled}
              className="h-9 w-24 text-right tabular-nums"
            />
          )}
        </label>
      ))}
    </div>
  )
}

/** Offers pretraining and fine-tuning as quiet tabs; seeded cycle modes are shown as a fact. */
function ModeToggle({
  mode,
  isCycleMode,
  disabled,
  onChange,
}: {
  mode: TrainMode
  isCycleMode: boolean
  disabled: boolean
  onChange: (mode: TrainMode) => void
}) {
  if (isCycleMode) {
    return (
      <p className="pb-3 text-xs text-muted-foreground">
        Seeded from a cycle&rsquo;s{" "}
        <span className="font-medium text-foreground">
          {mode === "raft" ? "self-train" : "preference"}
        </span>{" "}
        stage. Re-running it here is a one-off — cycles queue these themselves.
      </p>
    )
  }

  return (
    <div className="inline-flex gap-6">
      {(
        [
          ["pretrain", "Pretrain"],
          ["sft", "Fine-tune"],
        ] as [TrainMode, string][]
      ).map(([value, label]) => (
        <Button
          key={value}
          type="button"
          size="sm"
          variant="text"
          className={`h-auto rounded-none border-0 border-b-2 px-0 pb-3 hover:bg-transparent ${mode === value ? "border-foreground font-medium text-foreground" : "border-transparent text-muted-foreground"}`}
          disabled={disabled}
          aria-pressed={mode === value}
          onClick={() => onChange(value)}
        >
          {label}
        </Button>
      ))}
    </div>
  )
}
