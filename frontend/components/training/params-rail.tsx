"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { CurriculumBuilder } from "@/components/training/curriculum-builder"
import { DatasetPicker, Pow2Input } from "@/components/training/dataset-picker"
import {
  LEARN_WITH_RUN,
  STRING_FIELDS,
  configForMode,
  derived,
  fieldsFor,
  hidesTokenizer,
  inheritsArchitecture,
  validate,
} from "@/components/training/run-config"
import { Input } from "@/components/ui/input"
import { PresetControls } from "@/components/training/preset-controls"
import { ResumePicker } from "@/components/training/resume-picker"
import { useDatasets } from "@/hooks/use-datasets"
import { useTokenizers } from "@/hooks/use-tokenizers"
import { useModels } from "@/hooks/use-models"
import { cn } from "@/lib/utils"
import { isFineTune } from "@/lib/training"
import type { TrainConfig, TrainMode } from "@/lib/training"

/** Aligns the supplied field label and inline control in one quiet settings row. */
function Row({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-3 border-b border-border/40 py-3">
      <span className="text-sm text-muted-foreground">{label}</span>
      {children}
    </div>
  )
}

const fieldInputClass = "h-8 w-24 text-right tabular-nums"

export interface ParamsRailProps {
  cfg: TrainConfig
  onChange: (cfg: TrainConfig) => void
  disabled: boolean
  onStart: (cfg: TrainConfig) => void
  modelsRefreshKey?: unknown
  /**
   * Render as a *step* inside another composer rather than as a composer of its
   * own: a run queued to follow another one, which starts when that one lands.
   *
   * Three things follow from that and all three are the same fact. It emits no
   * `<form>` and no submit button, because it sits inside somebody else's form and
   * nested forms are not a thing. It shows no base-model picker, because the base
   * is whatever the job before it produces — that is what makes it a follow-up.
   * And it shows no presets, since a preset carries a base model and an
   * architecture, neither of which this step gets to choose.
   */
  followUp?: boolean
  /**
   * Rendered inside the form, immediately above the submit button.
   *
   * The one thing that has to sit there is the follow-up list: it is part of what
   * the button submits, so putting it after the button — which is where a sibling
   * of this component lands — would read as a control for something else.
   */
  footer?: React.ReactNode
  /**
   * Validation problems, reported upward. A follow-up has no submit button of its
   * own, so the composer it sits in is what has to refuse to submit — this is how
   * it finds out. Only called in `followUp` mode.
   */
  onValidity?: (errors: string[]) => void
}

/** Edits parent-owned training settings, sharing field rules with the main form. Follow-ups inherit their base and report validation without a nested form or submit. */
export function ParamsRail({
  cfg,
  onChange,
  disabled,
  onStart,
  modelsRefreshKey,
  followUp = false,
  onValidity,
  footer,
}: ParamsRailProps) {
  const { models, isLoading: modelsLoading } = useModels(modelsRefreshKey)
  const { datasets: availableDatasets, isLoading: datasetsLoading } =
    useDatasets(isFineTune(cfg.mode))
  const { tokenizers, isLoading: tokenizersLoading } = useTokenizers()
  const offerable = tokenizers.filter((t) => t.hasVocab && !t.implicit)

  /** Parses a text edit into the parent config, preserving literal string fields. */
  const setField = (key: keyof TrainConfig, value: string) => {
    onChange({ ...cfg, [key]: STRING_FIELDS.has(key) ? value : Number(value) })
  }
  /** Changes mode while preserving a learning rate the user has explicitly chosen. */
  const setMode = (mode: TrainMode) => onChange(configForMode(cfg, mode))

  const isKto = cfg.mode === "kto"
  const isRaft = cfg.mode === "raft"
  const isCycleMode = isKto || isRaft
  const isSft = isFineTune(cfg.mode)
  const isPlainSft = cfg.mode === "sft"
  const curriculumOn = (cfg.curriculum?.length ?? 0) > 0
  const { arch, run } = fieldsFor(cfg)
  const numFields = [...arch, ...run]
  /** Seeds curriculum from the flat selection or restores the first stage when disabling it. */
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
  const baseModelId = cfg.base_model_id || models[0]?.id
  const effectiveCfg: TrainConfig = followUp
    ? { ...cfg, base_model_id: undefined }
    : isSft
      ? { ...cfg, base_model_id: baseModelId }
      : cfg

  const errors = validate(effectiveCfg, followUp)
  const errorKey = errors.join("\n")
  const reportValidity = React.useEffectEvent(() =>
    onValidity?.(errorKey ? errorKey.split("\n") : [])
  )
  React.useEffect(() => {
    if (followUp) reportValidity()
  }, [followUp, errorKey])
  const { headDim, dFf } = derived(cfg)

  /** Prevents browser navigation and starts only the validated effective config. */
  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (errors.length > 0) return
    onStart(effectiveCfg)
  }
  const body = (
    <>
      {!followUp && (
        <PresetControls
          currentConfig={effectiveCfg}
          disabled={disabled}
          onLoad={onChange}
          modelsRefreshKey={modelsRefreshKey}
        />
      )}

      {isCycleMode ? (
        <p className="rounded-(--radius-control-sm) border border-dashed border-border px-3 py-2 text-xs text-muted-foreground">
          Seeded from a cycle&rsquo;s{" "}
          <span className="font-medium text-foreground">
            {isRaft ? "self-train" : "preference"}
          </span>{" "}
          stage. Re-running it here is a one-off — cycles queue these
          themselves.
        </p>
      ) : (
        <div className="inline-flex gap-6 border-b border-border/60">
          {(
            [
              ["pretrain", "Pretrain"],
              ["sft", "Fine-tune"],
            ] as [TrainMode, string][]
          ).map(([mode, label]) => (
            <Button
              key={mode}
              type="button"
              size="sm"
              variant="text"
              className={cn(
                "h-auto rounded-none border-0 border-b-2 px-0 pb-3 hover:bg-transparent",
                cfg.mode === mode
                  ? "border-foreground font-medium text-foreground"
                  : "border-transparent text-muted-foreground"
              )}
              disabled={disabled}
              onClick={() => setMode(mode)}
            >
              {label}
            </Button>
          ))}
        </div>
      )}

      <div className="space-y-1">
        <label className="text-xs text-muted-foreground">Model name</label>
        <Input
          variant="inline"
          aria-label="Follow-up model name"
          value={cfg.model_name}
          onChange={(e) => setField("model_name", e.target.value)}
          placeholder="Slate Feather"
          disabled={disabled}
          className="h-auto py-0 font-display text-xl md:text-xl"
        />
      </div>

      <div className="grid gap-x-8 sm:grid-cols-2">
        {isSft && !followUp && (
          <Row label="Base model">
            <Select
              value={baseModelId ?? ""}
              onValueChange={(v) => setField("base_model_id", v ?? "")}
              disabled={disabled || modelsLoading || models.length === 0}
            >
              <SelectTrigger
                variant="inline"
                aria-label="Base model"
                size="sm"
                className="h-8 w-44"
              >
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
          </Row>
        )}

        {!isSft && !hidesTokenizer(cfg) && (
          <Row label="Tokenizer">
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
                size="sm"
                className="h-8 w-44"
              >
                <SelectValue>
                  {(value: string | null) =>
                    !value || value === LEARN_WITH_RUN
                      ? "Learn one with the run"
                      : (offerable.find((t) => t.id === value)?.name ?? value)
                  }
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={LEARN_WITH_RUN}>
                  Learn one with the run
                </SelectItem>
                {offerable.map((t) => (
                  <SelectItem key={t.id} value={t.id}>
                    {t.name}
                    {t.vocabSize ? ` · ${t.vocabSize.toLocaleString()}` : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Row>
        )}

        {!isSft && (
          <Row label="Continue from">
            <ResumePicker
              value={cfg.resume_from}
              onChange={(resume_from) => onChange({ ...cfg, resume_from })}
              models={models}
              modelsLoading={modelsLoading}
              disabled={disabled}
              compact
              variant="inline"
            />
          </Row>
        )}

        {isSft && !followUp && (
          <Row label="Tokenizer">
            <span className="text-sm text-muted-foreground">
              Inherited from the base model
            </span>
          </Row>
        )}

        {isPlainSft && (
          <Row label="Strip thinking">
            <Checkbox
              aria-label="Strip thinking"
              checked={cfg.strip_thinking ?? false}
              disabled={disabled}
              onCheckedChange={(on) =>
                onChange({ ...cfg, strip_thinking: on === true })
              }
            />
          </Row>
        )}

        {numFields.map((f) =>
          f.pow2 ? (
            <Row key={f.key} label={f.label}>
              <Pow2Input
                variant="inline"
                ariaLabel={`${f.label} exponent`}
                value={cfg[f.key] as number}
                disabled={disabled}
                maxExp={f.maxExp}
                onChange={(n) => onChange({ ...cfg, [f.key]: n })}
              />
            </Row>
          ) : (
            <Row key={f.key} label={f.label}>
              <Input
                variant="inline"
                aria-label={f.label}
                type="number"
                step={f.step ?? 1}
                value={String(cfg[f.key])}
                onChange={(e) => setField(f.key, e.target.value)}
                disabled={disabled}
                className={fieldInputClass}
              />
            </Row>
          )
        )}
      </div>

      {isSft && (
        <div className="space-y-3">
          <DatasetPicker
            variant="inline"
            available={availableDatasets}
            loading={datasetsLoading}
            value={cfg.datasets ?? []}
            disabled={disabled}
            emptyHint={
              <>
                No conversation datasets yet. Pull one from the Data view first.
              </>
            }
            onChange={(datasets) => onChange({ ...cfg, datasets })}
          />
          {(cfg.datasets?.length ?? 0) === 0 &&
            availableDatasets.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Pick at least one — a run names what it trains on. Pick it
                deliberately, too: a thinking set folded into a no-think run is
                how reasoning traces get stripped and trained as answer-only.
              </p>
            )}
        </div>
      )}

      {!isSft && (
        <div className="space-y-3">
          <div className="space-y-3 border-y border-border/60 py-3">
            <label className="flex items-center justify-between gap-2">
              <span className="min-w-0 text-sm">
                Shuffle documents each epoch
                <span className="block text-xs text-muted-foreground">
                  Reorder docs every epoch, separated by an end-of-text token.
                </span>
              </span>
              <Checkbox
                checked={cfg.shuffle ?? true}
                disabled={disabled}
                onCheckedChange={(on) =>
                  onChange({ ...cfg, shuffle: on === true })
                }
              />
            </label>
            <label className="flex items-center justify-between gap-2 border-t pt-2">
              <span className="min-w-0 text-sm">
                Curriculum
                <span className="block text-xs text-muted-foreground">
                  Train staged datasets in order, each for its own epochs.
                </span>
              </span>
              <Checkbox
                checked={curriculumOn}
                disabled={disabled}
                onCheckedChange={(on) => setCurriculum(on === true)}
              />
            </label>
          </div>

          {curriculumOn ? (
            <CurriculumBuilder
              variant="inline"
              available={availableDatasets}
              loading={datasetsLoading}
              value={cfg.curriculum ?? []}
              disabled={disabled}
              onChange={(curriculum) => onChange({ ...cfg, curriculum })}
            />
          ) : (
            <DatasetPicker
              variant="inline"
              available={availableDatasets}
              loading={datasetsLoading}
              value={cfg.datasets ?? []}
              disabled={disabled}
              onChange={(datasets) => onChange({ ...cfg, datasets })}
            />
          )}
        </div>
      )}

      {inheritsArchitecture(cfg) ? (
        <p className="text-xs text-muted-foreground">
          Architecture (dimensions, context, heads, layers) is inherited from{" "}
          {isSft ? "the base model" : "the checkpoint"} — only the run knobs
          above apply.
          {!isSft && (
            <>
              {" "}
              Its tokenizer comes with them, so there is no vocabulary to pick.
            </>
          )}
          {isKto && (
            <>
              {" "}
              Trains on the thumbs up/down collected in chat and in a gather
              run. Needs both kinds: one-sided feedback has no contrast to learn
              from and the run will refuse.
            </>
          )}
          {isRaft && (
            <>
              {" "}
              Fine-tunes on the model&rsquo;s own replies that cleared the
              catalogue&rsquo;s bar — a dense signal, and the fastest way to
              teach a model at this stage to form English. Run it before a
              preference tune: afterwards the good replies already sit high, so
              KTO spends its gradient on suppressing the bad ones instead.
            </>
          )}
        </p>
      ) : (
        <div className="space-y-1 text-xs text-muted-foreground tabular-nums">
          <div className="flex justify-between">
            <span>Head size (dims ÷ heads)</span>
            <span className={cn("font-medium", headDim && "text-foreground")}>
              {headDim ?? "—"}
            </span>
          </div>
          <div className="flex justify-between">
            <span>FF hidden (dims × FF mult)</span>
            <span className={cn("font-medium", dFf && "text-foreground")}>
              {dFf || "—"}
            </span>
          </div>
        </div>
      )}

      {errors.length > 0 && !disabled && (
        <ul className="space-y-0.5 text-xs text-destructive">
          {errors.map((err) => (
            <li key={err}>{err}</li>
          ))}
        </ul>
      )}

      {footer}

      {!followUp && (
        <Button
          type="submit"
          disabled={disabled || errors.length > 0}
          className="w-full"
        >
          {disabled
            ? isSft
              ? "Fine-tuning…"
              : "Training…"
            : isSft
              ? "Start fine-tune"
              : "Start training"}
        </Button>
      )}
    </>
  )

  return followUp ? (
    <div className="space-y-4">{body}</div>
  ) : (
    <form onSubmit={submit} className="space-y-4">
      {body}
    </form>
  )
}
