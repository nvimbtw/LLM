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
  KTO_DEFAULT_BETA,
  LEARN_WITH_RUN,
  MODE_DEFAULT_LRS,
  STRING_FIELDS,
  defaultLrFor,
  derived,
  fieldsFor,
  validate,
} from "@/components/training/run-config"
import { Input } from "@/components/ui/input"
import { PresetControls } from "@/components/training/preset-controls"
import { useDatasets } from "@/hooks/use-datasets"
import { useTokenizers } from "@/hooks/use-tokenizers"
import { useModels } from "@/hooks/use-models"
import { cn } from "@/lib/utils"
import { isFineTune } from "@/lib/training"
import type { TrainConfig, TrainMode } from "@/lib/training"

// One definition-list row: a muted label on the left, an edit-in-place value
// right-aligned in tabular-nums. Borderless until focused, so at a glance the
// rail reads as a spec sheet rather than a form.
function Row({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="flex items-center justify-between gap-3 py-1.5">
      <span className="text-sm text-muted-foreground">{label}</span>
      {children}
    </div>
  )
}

// Proper input chrome (border, radius, padding from the base Input), just sized
// for the rail and right-aligned with tabular figures.
const fieldInputClass = "h-8 w-24 text-right tabular-nums"

export interface ParamsRailProps {
  cfg: TrainConfig
  onChange: (cfg: TrainConfig) => void
  disabled: boolean
  onStart: (cfg: TrainConfig) => void
  // Advances when a new model is registered, so the base-model picker refetches
  // and the just-trained model is selectable without a reload.
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

// The left-rail params panel: the run's identity + knobs as a hairline definition
// list. Config state is owned by the parent (so the continuation flow can seed it);
// this component edits it through `onChange` and validates on every render.
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
  // A fine-tune picks from the SFT conversation sets, a pretrain from the prose
  // corpora — two different folders on disk, so the catalogue follows the mode.
  const { datasets: availableDatasets, isLoading: datasetsLoading } =
    useDatasets(isFineTune(cfg.mode))
  const { tokenizers, isLoading: tokenizersLoading } = useTokenizers()
  // A vocabulary whose directory has gone cannot be trained against, and one a
  // run derived for itself was never a choice — it is listed on the tokenizers
  // page, not offered here.
  const offerable = tokenizers.filter((t) => t.hasVocab && !t.implicit)
  // Only meaningful in preference mode, but hooks can't be conditional — and it
  // is one small count, read once per mount.

  const setField = (key: keyof TrainConfig, value: string) => {
    onChange({ ...cfg, [key]: STRING_FIELDS.has(key) ? value : Number(value) })
  }

  // Switching mode nudges the learning rate to that mode's sane default — but only
  // while it is still sitting on some mode's default, so a rate the user actually
  // typed is never overwritten.
  const setMode = (mode: TrainMode) => {
    if (cfg.mode === mode) return
    const next: TrainConfig = { ...cfg, mode }
    if (MODE_DEFAULT_LRS.includes(cfg.lr)) next.lr = defaultLrFor(mode)
    if (mode === "kto" && next.beta === undefined) next.beta = KTO_DEFAULT_BETA
    onChange(next)
  }

  const isKto = cfg.mode === "kto"
  const isRaft = cfg.mode === "raft"
  // The two the cycle owns. Reachable here only by seeding from one of its jobs.
  const isCycleMode = isKto || isRaft
  const isSft = isFineTune(cfg.mode)
  // Strictly SFT, not the other two fine-tunes: RAFT and KTO train on the model's
  // own generations, which have no thinking block to strip, so offering the switch
  // there would be a control that does nothing.
  const isPlainSft = cfg.mode === "sft"
  const curriculumOn = (cfg.curriculum?.length ?? 0) > 0
  const { arch, run } = fieldsFor(cfg)
  const numFields = [...arch, ...run]

  // Toggle curriculum mode: turning it on seeds a first stage from the current
  // flat selection (so nothing is lost); turning it off clears the stages and
  // restores the flat selection from the first stage.
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

  // The base id is derived: if the user hasn't explicitly picked one, default to
  // the first available model, keeping the select controlled and the submit correct.
  const baseModelId = cfg.base_model_id || models[0]?.id
  const effectiveCfg: TrainConfig = followUp
    ? // Stripped rather than left alone: the server refuses a follow-up that names
      // its own base, so a stale id carried in from `?from=` would be a 400 rather
      // than a field nobody can see.
      { ...cfg, base_model_id: undefined }
    : isSft
      ? { ...cfg, base_model_id: baseModelId }
      : cfg

  const errors = validate(effectiveCfg, followUp)
  // Reported to the composer this step sits in, which owns the submit button.
  // Keyed on the joined text so an unchanged list doesn't re-notify every render.
  const errorKey = errors.join("\n")
  React.useEffect(() => {
    if (followUp) onValidity?.(errorKey ? errorKey.split("\n") : [])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [followUp, errorKey])
  const { headDim, dFf } = derived(cfg)

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (errors.length > 0) return
    onStart(effectiveCfg)
  }

  // Held as a value rather than returned directly, because a follow-up is a step
  // inside somebody else's form and so emits neither a form of its own (nested
  // forms are not valid) nor a submit button (the composer it sits in owns the one
  // submit). Everything between the two is identical, which is why this is a prop
  // rather than a second rail.
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

      {/* Mode toggle — two, not four.
          Self-train (RAFT) and preference (KTO) are stages of a cycle, not runs a
          person composes: the cycle decides when the catalogue is ready for each,
          and queues them with its own learning rates (`backend/src/cycle.rs`).
          Offering them here presented four peers when there are two decisions.

          They are still rendered when `?from=` seeds this form from a cycle's own
          job — that config is real and re-runnable, so the mode is shown as a fact
          rather than silently rewritten to something the run was not. */}
      {isCycleMode ? (
        <p className="rounded-md border border-dashed border-border px-3 py-2 text-xs text-muted-foreground">
          Seeded from a cycle&rsquo;s{" "}
          <span className="font-medium text-foreground">
            {isRaft ? "self-train" : "preference"}
          </span>{" "}
          stage. Re-running it here is a one-off — cycles queue these
          themselves.
        </p>
      ) : (
        <div className="inline-flex w-full gap-0.5 rounded-md border p-0.5">
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
              // The selected half is a surface step and a weight change, not a
              // filled primary. Two choices sitting inside a bordered track are
              // already legible as a pair; painting one of them in the accent
              // made the quietest decision on the form its loudest mark, and it
              // competed with the actual primary action at the bottom of it.
              variant={cfg.mode === mode ? "secondary" : "ghost"}
              className={cn(
                "flex-1",
                cfg.mode === mode ? "font-medium" : "text-muted-foreground"
              )}
              disabled={disabled}
              onClick={() => setMode(mode)}
            >
              {label}
            </Button>
          ))}
        </div>
      )}

      {/* Model name — the rail's headline / run identity */}
      <div className="space-y-1">
        <label className="text-xs text-muted-foreground">Model name</label>
        <Input
          value={cfg.model_name}
          onChange={(e) => setField("model_name", e.target.value)}
          // Weight classes are the size convention: Feather · Light · Welter ·
          // Middle · Heavy, smallest to largest. Only a naming convention — nothing
          // validates it, and the field takes any name.
          placeholder="Slate Feather"
          disabled={disabled}
          className="h-10 font-heading text-lg font-semibold"
        />
      </div>

      <dl>
        {isSft && !followUp && (
          <Row label="Base model">
            <Select
              value={baseModelId ?? ""}
              // The Select reports `string | null`; the field is a plain id.
              onValueChange={(v) => setField("base_model_id", v ?? "")}
              disabled={disabled || modelsLoading || models.length === 0}
            >
              <SelectTrigger size="sm" className="h-8 w-44">
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

        {!isSft && (
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
              <SelectTrigger size="sm" className="h-8 w-44">
                <SelectValue>
                  {(value: string | null) =>
                    !value || value === LEARN_WITH_RUN
                      ? "Learn one with the run"
                      : (offerable.find((t) => t.id === value)?.name ?? value)
                  }
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {/* A real option, not a placeholder: learning one with the run
                    is what every run did before builds existed, and it stays
                    the default. As a placeholder it would read as "not loaded
                    yet". */}
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
                value={cfg[f.key] as number}
                disabled={disabled}
                maxExp={f.maxExp}
                onChange={(n) => onChange({ ...cfg, [f.key]: n })}
              />
            </Row>
          ) : (
            <Row key={f.key} label={f.label}>
              <Input
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
      </dl>

      {isSft && (
        <div className="space-y-3">
          <DatasetPicker
            available={availableDatasets}
            loading={datasetsLoading}
            value={cfg.datasets ?? []}
            disabled={disabled}
            emptyHint={
              <>No conversation datasets yet. Pull one from the Data view first.</>
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
          {/* Data-order controls: shuffle each epoch + curriculum staging. */}
          <div className="space-y-2 rounded-lg border p-2.5">
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
              available={availableDatasets}
              loading={datasetsLoading}
              value={cfg.curriculum ?? []}
              disabled={disabled}
              onChange={(curriculum) => onChange({ ...cfg, curriculum })}
            />
          ) : (
            <DatasetPicker
              available={availableDatasets}
              loading={datasetsLoading}
              value={cfg.datasets ?? []}
              disabled={disabled}
              onChange={(datasets) => onChange({ ...cfg, datasets })}
            />
          )}
        </div>
      )}

      {isSft ? (
        <p className="text-xs text-muted-foreground">
          Architecture (dimensions, context, heads, layers) is inherited from
          the base model — only the run knobs above apply.
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
