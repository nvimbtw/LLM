"use client"

import * as React from "react"

import { DatasetPicker, Pow2Input } from "@/components/training/dataset-picker"
import { PresetControls } from "@/components/training/preset-controls"
import { BuildSheet } from "@/components/training/build-sheet"
import { CurriculumBuilder } from "@/components/training/curriculum-builder"
import {
  KTO_DEFAULT_BETA,
  LEARN_WITH_RUN,
  MODE_DEFAULT_LRS,
  STRING_FIELDS,
  defaultLrFor,
  fieldsFor,
  validate,
  type Field,
} from "@/components/training/run-config"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Panel } from "@/components/ui/panel"
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
import {
  isFineTune,
  type DatasetInfo,
  type TrainConfig,
  type TrainMode,
} from "@/lib/training"
import { cn } from "@/lib/utils"

/**
 * Composing a training run, as a board.
 *
 * The decisions run down the left in the order they depend on each other — what
 * kind of run, what it is, what it learns from, how hard it trains — and the
 * right column is the build sheet for the model those decisions describe. That
 * sheet is deliberately the same spec-sheet idiom `/models/[id]` uses for a
 * model that already exists, so the thing you are specifying and the thing you
 * later read back are recognisably one object at two moments of its life.
 *
 * A follow-up step renders the same fields as a narrow rail instead
 * (`ParamsRail`); both take their field list and their validation from
 * `run-config`, so the two arrangements cannot drift on what a run may be.
 */
export interface TrainBoardProps {
  cfg: TrainConfig
  onChange: (cfg: TrainConfig) => void
  disabled: boolean
  onStart: (cfg: TrainConfig) => void
  /** Advances when a model is registered, so the base picker refetches. */
  modelsRefreshKey?: unknown
  /** The follow-up list. Part of what the button submits, so it sits above it. */
  footer?: React.ReactNode
}

export function TrainBoard({
  cfg,
  onChange,
  disabled,
  onStart,
  modelsRefreshKey,
  footer,
}: TrainBoardProps) {
  const { models, isLoading: modelsLoading } = useModels(modelsRefreshKey)
  const { datasets, isLoading: datasetsLoading } = useDatasets(
    isFineTune(cfg.mode)
  )
  const { tokenizers, isLoading: tokenizersLoading } = useTokenizers()

  const isSft = isFineTune(cfg.mode)
  const isCycleMode = cfg.mode === "kto" || cfg.mode === "raft"
  const { arch, run } = fieldsFor(cfg)

  // The base id is derived: with none explicitly picked, default to the first
  // available model, keeping the select controlled and the submit correct.
  const baseModelId = cfg.base_model_id || models[0]?.id
  const effective: TrainConfig = isSft ? { ...cfg, base_model_id: baseModelId } : cfg
  const errors = validate(effective)

  const set = (key: keyof TrainConfig, value: string) =>
    onChange({ ...cfg, [key]: STRING_FIELDS.has(key) ? value : Number(value) })

  const setNumber = (key: keyof TrainConfig, value: number) =>
    onChange({ ...cfg, [key]: value })

  // Switching mode nudges the learning rate to that mode's sane default — but
  // only while it is still sitting on some mode's default, so a rate the user
  // actually typed is never overwritten.
  const setMode = (mode: TrainMode) => {
    if (cfg.mode === mode) return
    const next: TrainConfig = { ...cfg, mode }
    if (MODE_DEFAULT_LRS.includes(cfg.lr)) next.lr = defaultLrFor(mode)
    if (mode === "kto" && next.beta === undefined) next.beta = KTO_DEFAULT_BETA
    onChange(next)
  }

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (errors.length > 0) return
    onStart(effective)
  }

  return (
    <form onSubmit={submit}>
      <div className="mb-6 flex justify-end">
        <PresetControls
          currentConfig={effective}
          disabled={disabled}
          onLoad={onChange}
          modelsRefreshKey={modelsRefreshKey}
        />
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_21rem] xl:gap-8">
        <div className="min-w-0 space-y-6">
          <Panel title="Run" description="Name and starting point">
            <div className="space-y-5">
              <ModeToggle
                mode={cfg.mode}
                isCycleMode={isCycleMode}
                disabled={disabled}
                onChange={setMode}
              />

              <div className="grid gap-5 sm:grid-cols-2">
                <label className="space-y-1.5">
                  <FieldLabel hint="What the weights will be called">
                    Model name
                  </FieldLabel>
                  <Input
                    value={cfg.model_name}
                    onChange={(e) => set("model_name", e.target.value)}
                    // Weight classes are the size convention: Feather · Light ·
                    // Welter · Middle · Heavy, smallest to largest. Only a naming
                    // convention — nothing validates it.
                    placeholder="Slate Feather"
                    disabled={disabled}
                    className="h-10 font-heading text-base font-semibold"
                  />
                </label>

                {isSft ? (
                  <label className="space-y-1.5">
                    <FieldLabel hint="Its architecture is inherited">
                      Base model
                    </FieldLabel>
                    <Select
                      value={baseModelId ?? ""}
                      onValueChange={(v) => set("base_model_id", v ?? "")}
                      disabled={disabled || modelsLoading || models.length === 0}
                    >
                      <SelectTrigger className="h-10 w-full">
                        <SelectValue
                          placeholder={modelsLoading ? "Loading…" : "No models"}
                        >
                          {(value: string | null) =>
                            models.find((m) => m.id === value)?.name ?? ""
                          }
                        </SelectValue>
                      </SelectTrigger>
                      <SelectContent>
                        {models.map((m) => (
                          <SelectItem key={m.id} value={m.id}>
                            {m.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </label>
                ) : (
                  <label className="space-y-1.5">
                    <FieldLabel hint="Reuse a build, or learn one as it packs">
                      Tokenizer
                    </FieldLabel>
                    <Select
                      value={cfg.tokenizer_id ?? LEARN_WITH_RUN}
                      onValueChange={(v) =>
                        onChange({
                          ...cfg,
                          tokenizer_id:
                            !v || v === LEARN_WITH_RUN ? undefined : v,
                        })
                      }
                      disabled={disabled || tokenizersLoading}
                    >
                      <SelectTrigger className="h-10 w-full">
                        <SelectValue>
                          {(value: string | null) =>
                            !value || value === LEARN_WITH_RUN
                              ? "Learn one with the run"
                              : (tokenizers.find((t) => t.id === value)?.name ??
                                value)
                          }
                        </SelectValue>
                      </SelectTrigger>
                      <SelectContent>
                        {/* A real option, not a placeholder: learning one with
                            the run is what every run did before builds existed,
                            and it stays the default. */}
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
                  </label>
                )}
              </div>

            </div>
          </Panel>

          {arch.length > 0 && (
            <Panel
              title="Architecture"
              description="What the model is"
              bodyClassName="p-0"
            >
              <FieldGrid
                fields={arch}
                cfg={cfg}
                disabled={disabled}
                onText={set}
                onNumber={setNumber}
              />
            </Panel>
          )}

          <Panel
            title="Training data"
            description={isSft ? "Conversations" : "Corpora"}
          >
            <DataSection
              cfg={cfg}
              onChange={onChange}
              disabled={disabled}
              datasets={datasets}
              loading={datasetsLoading}
            />
          </Panel>

          <Panel
            title="Optimization"
            description="How hard it trains"
            bodyClassName="p-0"
          >
            <FieldGrid
              fields={run}
              cfg={cfg}
              disabled={disabled}
              onText={set}
              onNumber={setNumber}
            />
            {isSft && (
              <p className="border-t border-border px-5 py-3 text-xs text-muted-foreground">
                Dimensions, context, heads and layers come from the base model —
                only these apply.
              </p>
            )}
          </Panel>

          {/* Not wrapped in a Panel: the follow-up list frames and titles
              itself, and a panel around it reads as two headings for one thing. */}
          {footer}
        </div>

        <aside className="lg:sticky lg:top-6">
          <BuildSheet
            cfg={effective}
            baseName={models.find((m) => m.id === baseModelId)?.name}
            datasets={datasets}
            errors={errors}
            disabled={disabled}
          />
        </aside>
      </div>
    </form>
  )
}

function FieldLabel({
  children,
  hint,
}: {
  children: React.ReactNode
  hint?: string
}) {
  return (
    <span className="block">
      <span className="text-xs font-medium tracking-[0.08em] text-muted-foreground uppercase">
        {children}
      </span>
      {hint && (
        <span className="mt-0.5 block text-xs normal-case text-muted-foreground/80">
          {hint}
        </span>
      )}
    </span>
  )
}

/**
 * The numeric fields as a hairline grid rather than a stacked list.
 *
 * `gap-px` over a border-coloured background is what draws the rules between
 * cells: one element, no double borders where two cells meet, and it collapses
 * to a single column on a narrow screen without leaving a stray rule behind.
 */
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
    <div className="grid gap-px bg-border/60 sm:grid-cols-2 xl:grid-cols-3">
      {fields.map((f) => (
        <label key={f.key} className="space-y-2 bg-card p-5">
          <FieldLabel hint={f.hint}>{f.label}</FieldLabel>
          {f.pow2 ? (
            <Pow2Input
              value={cfg[f.key] as number}
              disabled={disabled}
              maxExp={f.maxExp}
              onChange={(n) => onNumber(f.key, n)}
            />
          ) : (
            <Input
              type="number"
              step={f.step ?? 1}
              value={String(cfg[f.key] ?? "")}
              onChange={(e) => onText(f.key, e.target.value)}
              disabled={disabled}
              className="h-9 w-full tabular-nums"
            />
          )}
        </label>
      ))}
    </div>
  )
}

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
  // Self-train (RAFT) and preference (KTO) are stages of a cycle, not runs a
  // person composes. They are still shown when `?from=` seeds this form from a
  // cycle's own job — that config is real and re-runnable, so the mode is stated
  // as a fact rather than silently rewritten to something the run was not.
  if (isCycleMode) {
    return (
      <p className="rounded-md border border-dashed border-border px-3 py-2 text-xs text-muted-foreground">
        Seeded from a cycle&rsquo;s{" "}
        <span className="font-medium text-foreground">
          {mode === "raft" ? "self-train" : "preference"}
        </span>{" "}
        stage. Re-running it here is a one-off — cycles queue these themselves.
      </p>
    )
  }

  return (
    <div className="inline-flex gap-0.5 rounded-md border p-0.5">
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
          // The selected half is a surface step and a weight change, not a filled
          // primary: painting it in the accent made the quietest decision on the
          // form its loudest mark, and it competed with the actual primary action.
          variant={mode === value ? "secondary" : "ghost"}
          className={cn(
            "w-32",
            mode === value ? "font-medium" : "text-muted-foreground"
          )}
          disabled={disabled}
          onClick={() => onChange(value)}
        >
          {label}
        </Button>
      ))}
    </div>
  )
}


/**
 * What the run learns from: the picker, plus the two switches that decide how
 * its documents are ordered.
 *
 * The switches sit above the picker rather than below it because curriculum
 * changes what the picker *is* — one selection, or one per stage — and a control
 * that reshapes the thing under it should not be underneath it.
 */
function DataSection({
  cfg,
  onChange,
  disabled,
  datasets,
  loading,
}: {
  cfg: TrainConfig
  onChange: (cfg: TrainConfig) => void
  disabled: boolean
  datasets: DatasetInfo[]
  loading: boolean
}) {
  const isSft = isFineTune(cfg.mode)
  const curriculumOn = (cfg.curriculum?.length ?? 0) > 0

  // Turning curriculum on seeds a first stage from the current flat selection so
  // nothing is lost; turning it off restores the flat selection from stage one.
  const setCurriculum = (on: boolean) => {
    if (on) {
      onChange({
        ...cfg,
        curriculum: [{ datasets: cfg.datasets ?? [], epochs: cfg.epochs }],
      })
    } else {
      onChange({
        ...cfg,
        curriculum: [],
        datasets: cfg.curriculum?.[0]?.datasets ?? cfg.datasets ?? [],
      })
    }
  }

  if (isSft) {
    return (
      <div className="space-y-3">
        <DatasetPicker
          available={datasets}
          loading={loading}
          value={cfg.datasets ?? []}
          disabled={disabled}
          emptyHint={
            <>No conversation datasets yet. Pull one from the Data view first.</>
          }
          onChange={(next) => onChange({ ...cfg, datasets: next })}
        />
        {(cfg.datasets?.length ?? 0) === 0 && datasets.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Pick at least one — a run names what it trains on. Pick it
            deliberately, too: a thinking set folded into a no-think run is how
            reasoning traces get stripped and trained as answer-only.
          </p>
        )}
        {cfg.mode === "sft" && (
          <label className="flex items-center justify-between gap-3 border-t border-border pt-3">
            <span className="min-w-0 text-sm">
              Strip thinking
              <span className="block text-xs text-muted-foreground">
                Drop the reasoning blocks, teaching it to answer directly.
              </span>
            </span>
            <Checkbox
              checked={cfg.strip_thinking ?? false}
              disabled={disabled}
              onCheckedChange={(on) =>
                onChange({ ...cfg, strip_thinking: on === true })
              }
            />
          </label>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-px overflow-hidden rounded-lg border border-border bg-border/60 sm:grid-cols-2">
        <Switch
          label="Shuffle each epoch"
          hint="Reorder documents every pass, separated by an end-of-text token."
          checked={cfg.shuffle ?? true}
          disabled={disabled}
          onChange={(on) => onChange({ ...cfg, shuffle: on })}
        />
        <Switch
          label="Curriculum"
          hint="Train staged datasets in order, each for its own epochs."
          checked={curriculumOn}
          disabled={disabled}
          onChange={setCurriculum}
        />
      </div>

      {curriculumOn ? (
        <CurriculumBuilder
          available={datasets}
          loading={loading}
          value={cfg.curriculum ?? []}
          disabled={disabled}
          onChange={(curriculum) => onChange({ ...cfg, curriculum })}
        />
      ) : (
        <DatasetPicker
          available={datasets}
          loading={loading}
          value={cfg.datasets ?? []}
          disabled={disabled}
          onChange={(next) => onChange({ ...cfg, datasets: next })}
        />
      )}
    </div>
  )
}

function Switch({
  label,
  hint,
  checked,
  disabled,
  onChange,
}: {
  label: string
  hint: string
  checked: boolean
  disabled: boolean
  onChange: (on: boolean) => void
}) {
  return (
    <label className="flex items-start justify-between gap-3 bg-card p-4">
      <span className="min-w-0 text-sm">
        {label}
        <span className="mt-0.5 block text-xs text-muted-foreground">
          {hint}
        </span>
      </span>
      <Checkbox
        checked={checked}
        disabled={disabled}
        onCheckedChange={(on) => onChange(on === true)}
      />
    </label>
  )
}
