"use client"

import * as React from "react"
import { ChevronDown, ChevronRight, ChevronUp, Plus, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { DatasetPicker, Pow2Input } from "@/components/training/dataset-picker"
import { Input } from "@/components/ui/input"
import { PresetControls } from "@/components/training/preset-controls"
import { DEFAULT_CONFIG } from "@/hooks/use-training"
import { useDatasets } from "@/hooks/use-datasets"
import { useTokenizers } from "@/hooks/use-tokenizers"
import { useModels } from "@/hooks/use-models"
import { cn } from "@/lib/utils"
import { isFineTune } from "@/lib/training"
import type {
  CurriculumStage,
  DatasetInfo,
  TrainConfig,
  TrainMode,
} from "@/lib/training"

// Numeric fields split by which mode they apply to. Architecture + dataset knobs
// only make sense for a fresh pretrain; a fine-tune inherits the base model's
// architecture, so in SFT mode only the run knobs (epochs, learning rate) show.
// `pow2` fields only accept powers of two, so scrolling / arrowing them cycles
// through 16 · 32 · 64 · 128 … rather than stepping by one (see Pow2Input).
// `maxExp` overrides the default exponent ceiling for fields that legitimately go
// larger (e.g. vocab).
type Field = {
  key: keyof TrainConfig
  label: string
  step?: number
  pow2?: boolean
  maxExp?: number
}

// Every field the rail edits as text. Everything else goes through `Number()`,
// so a string field missing from here arrives at the engine as 0 — silently, and
// only for the one field somebody forgot.
const STRING_FIELDS = new Set<keyof TrainConfig>([
  "model_name",
  "base_model_id",
  "tokenizer_id",
])

const ARCH_FIELDS: Field[] = [
  { key: "dimensions", label: "Dimensions", pow2: true },
  { key: "ctx_window", label: "Context window", pow2: true },
  { key: "heads", label: "Heads", pow2: true },
  { key: "layers", label: "Layers" },
  { key: "ff_multiplier", label: "FF multiplier" },
  // Vocab can be far larger than a model dimension — GPT-2 is ~50k (2^16). Allow
  // up to 2^20 (~1M).
  { key: "max_vocab", label: "Max vocab", pow2: true, maxExp: 20 },
]

const RUN_FIELDS: Field[] = [
  { key: "epochs", label: "Epochs" },
  { key: "lr", label: "Learning rate", step: 0.001 },
]

// SFT wants a much lower learning rate than a from-scratch run — the base model
// is already near a good minimum, so a large step would wash out what it learned.
// Now that the pretrain default is 1e-4 (ISSUE-0052/0058 lifted the LR cap), SFT
// sits an order of magnitude below it at 1e-5.
const SFT_DEFAULT_LR = 0.00001

// KTO sits lower still. Its gradient is the cross-entropy one scaled by a factor
// that can exceed 1 on a confidently-wrong example, and it is deliberately allowed
// to push probability *down* — an over-large step there degrades the model far
// faster than an over-large SFT step, which can only ever pull it toward real text.
const KTO_DEFAULT_LR = 0.000005

// RAFT is ordinary SFT under the hood — same loss, same masking — so it wants the
// SFT learning rate, not KTO's much lower one. Named separately because reading
// `SFT_DEFAULT_LR` under a RAFT branch invites someone to "fix" it to the KTO
// value. These two are only reached now when the form is seeded from a cycle's
// job; the cycle's own runs use `raft_lr` / `kto_lr` from its config instead.
const RAFT_DEFAULT_LR = SFT_DEFAULT_LR

// The KL anchor's default. See llm::kto::KtoParams::beta.
const KTO_DEFAULT_BETA = 0.1

// Every learning rate the mode toggle sets on the user's behalf, so switching modes
// can tell "still on a default" from "the user chose this".
const MODE_DEFAULT_LRS: number[] = [
  DEFAULT_CONFIG.lr,
  SFT_DEFAULT_LR,
  KTO_DEFAULT_LR,
]

function defaultLrFor(mode: TrainMode): number {
  if (mode === "sft") return SFT_DEFAULT_LR
  if (mode === "raft") return RAFT_DEFAULT_LR
  if (mode === "kto") return KTO_DEFAULT_LR
  return DEFAULT_CONFIG.lr
}

function isPowerOfTwo(n: number) {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0
}

// The curriculum builder: an ordered list of stages, each its own dataset
// selection + epoch count, trained in sequence. Reuses DatasetPicker per stage so
// a stage reads exactly like the flat picker. Stages can be reordered (the order
// IS the curriculum) and removed; "Add stage" appends an empty one.
function CurriculumBuilder({
  available,
  loading,
  value,
  disabled,
  onChange,
}: {
  available: DatasetInfo[]
  loading: boolean
  value: CurriculumStage[]
  disabled: boolean
  onChange: (next: CurriculumStage[]) => void
}) {
  // Collapsed stages, tracked by index — a purely visual state so the rail can be
  // kept compact. A freshly-added stage starts expanded so it can be configured.
  const [collapsed, setCollapsed] = React.useState<Set<number>>(new Set())

  const toggleCollapsed = (i: number) =>
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })

  const update = (i: number, patch: Partial<CurriculumStage>) =>
    onChange(value.map((s, idx) => (idx === i ? { ...s, ...patch } : s)))

  const remove = (i: number) => onChange(value.filter((_, idx) => idx !== i))

  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir
    if (j < 0 || j >= value.length) return
    const next = [...value]
    ;[next[i], next[j]] = [next[j], next[i]]
    onChange(next)
  }

  const add = () => onChange([...value, { datasets: [], epochs: 1 }])

  // One-line summary of a collapsed stage: its datasets and epoch count.
  const summary = (stage: CurriculumStage) => {
    const names = stage.datasets.map((d) => d.name)
    const datasets = names.length > 0 ? names.join(", ") : "no datasets"
    return `${datasets} · ${stage.epochs} ep`
  }

  return (
    <div className="space-y-3">
      <span className="text-xs text-muted-foreground">Curriculum stages</span>

      {value.map((stage, i) => {
        const isCollapsed = collapsed.has(i)
        const title = stage.name?.trim() || `Stage ${i + 1}`
        return (
          <div key={i} className="rounded-lg border">
            {/* Header: click the title to collapse/expand; reorder + remove stay
                on the right. Kept compact so a long curriculum doesn't dominate. */}
            <div className="flex items-center justify-between gap-2 p-2.5">
              <button
                type="button"
                onClick={() => toggleCollapsed(i)}
                aria-expanded={!isCollapsed}
                className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
              >
                {isCollapsed ? (
                  <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
                ) : (
                  <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
                )}
                <span className="min-w-0 truncate">
                  <span className="text-xs font-medium tabular-nums">
                    {i + 1}.
                  </span>{" "}
                  <span className="text-sm">{title}</span>
                  {isCollapsed && (
                    <span className="ml-1.5 truncate text-xs text-muted-foreground">
                      {summary(stage)}
                    </span>
                  )}
                </span>
              </button>
              <div className="flex shrink-0 items-center gap-0.5">
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className="size-7"
                  disabled={disabled || i === 0}
                  onClick={() => move(i, -1)}
                  aria-label="Move stage up"
                >
                  <ChevronUp className="size-4" />
                </Button>
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className="size-7"
                  disabled={disabled || i === value.length - 1}
                  onClick={() => move(i, 1)}
                  aria-label="Move stage down"
                >
                  <ChevronDown className="size-4" />
                </Button>
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className="size-7 text-muted-foreground hover:text-destructive"
                  disabled={disabled}
                  onClick={() => remove(i)}
                  aria-label="Remove stage"
                >
                  <X className="size-4" />
                </Button>
              </div>
            </div>

            {!isCollapsed && (
              <div className="space-y-2.5 border-t p-2.5">
                <Input
                  value={stage.name ?? ""}
                  onChange={(e) =>
                    update(i, { name: e.target.value || undefined })
                  }
                  placeholder={`Stage ${i + 1} label (optional)`}
                  disabled={disabled}
                  className="h-8 text-sm"
                />

                <DatasetPicker
                  available={available}
                  loading={loading}
                  value={stage.datasets}
                  disabled={disabled}
                  onChange={(datasets) => update(i, { datasets })}
                />

                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm text-muted-foreground">Epochs</span>
                  <Input
                    type="number"
                    min={1}
                    step={1}
                    value={String(stage.epochs)}
                    onChange={(e) =>
                      update(i, { epochs: Number(e.target.value) })
                    }
                    disabled={disabled}
                    className="h-8 w-24 text-right tabular-nums"
                  />
                </div>
              </div>
            )}
          </div>
        )
      })}

      <Button
        type="button"
        size="sm"
        variant="outline"
        className="w-full"
        disabled={disabled}
        onClick={add}
      >
        <Plus className="size-4" /> Add stage
      </Button>
    </div>
  )
}

// Name is a display label only (weights live under the model's UUID), so spaces
// and dots like "Opal 1.1" are allowed — just no surrounding whitespace or
// control characters. Shared by both modes.
function validateName(name: string): string[] {
  if (!name.trim()) return ["Model name is required."]
  if (name !== name.trim())
    return ["Model name may not have leading or trailing whitespace."]
  if (/[\x00-\x1f\x7f]/.test(name))
    return ["Model name may not contain control characters."]
  return []
}

// Mirrors the backend's validate_config / validate_sft so the user sees the
// constraints before submitting. Returns a list of human-readable problems.
function validate(cfg: TrainConfig, inheritsBase = false): string[] {
  const errs = validateName(cfg.model_name)

  if (isFineTune(cfg.mode)) {
    // A follow-up has no base to select: it fine-tunes whatever the job before it
    // produces, which is exactly why it is a follow-up.
    if (!cfg.base_model_id && !inheritsBase)
      errs.push("Select a base model to fine-tune.")
    if (
      cfg.mode === "kto" &&
      !(Number.isFinite(cfg.beta) && (cfg.beta ?? 0) > 0)
    ) {
      errs.push("Beta must be a positive number.")
    }
    if (!Number.isFinite(cfg.epochs) || cfg.epochs <= 0)
      errs.push("Epochs must be positive.")
    if (!Number.isFinite(cfg.lr) || cfg.lr <= 0)
      errs.push("Learning rate must be positive.")
    return errs
  }

  const positives: [keyof TrainConfig, string][] = [
    ["dimensions", "Dimensions"],
    ["ctx_window", "Context window"],
    ["heads", "Heads"],
    ["layers", "Layers"],
    ["epochs", "Epochs"],
    ["ff_multiplier", "FF multiplier"],
  ]
  for (const [k, label] of positives) {
    const v = cfg[k] as number
    if (!Number.isFinite(v) || v <= 0) errs.push(`${label} must be positive.`)
  }

  if (!isPowerOfTwo(cfg.dimensions))
    errs.push("Dimensions must be a power of two (16, 32, 64, 128…).")
  if (!isPowerOfTwo(cfg.ctx_window))
    errs.push("Context window must be a power of two (16, 32, 64, 128…).")
  if (
    Number.isFinite(cfg.dimensions) &&
    cfg.heads > 0 &&
    cfg.dimensions % cfg.heads !== 0
  )
    errs.push(
      `Heads must divide dimensions evenly. ${cfg.dimensions} ÷ ${cfg.heads} isn't whole.`
    )
  if (!Number.isFinite(cfg.lr) || cfg.lr <= 0)
    errs.push("Learning rate must be positive.")
  if (!Number.isFinite(cfg.max_vocab) || cfg.max_vocab < 256)
    errs.push("Max vocab must be at least 256.")

  // Curriculum runs supply their datasets per stage; a plain run uses the flat
  // selection. Validate whichever is active (mirrors the backend).
  const curriculum = cfg.curriculum ?? []
  if (curriculum.length > 0) {
    curriculum.forEach((stage, i) => {
      const ds = stage.datasets ?? []
      if (!Number.isFinite(stage.epochs) || stage.epochs <= 0)
        errs.push(`Stage ${i + 1}: epochs must be positive.`)
      if (ds.length === 0)
        errs.push(`Stage ${i + 1}: select at least one dataset.`)
      if (ds.some((d) => !Number.isInteger(d.count) || d.count < 0))
        errs.push(
          `Stage ${i + 1}: each dataset count must be 0 (all) or positive.`
        )
    })
  } else {
    // Defensive: a config from a stale source (e.g. a preset saved before this
    // field existed) may lack `datasets`; treat missing as an empty selection so
    // it surfaces as a normal validation error rather than a crash.
    const selected = cfg.datasets ?? []
    if (selected.length === 0)
      errs.push("Select at least one dataset to train on.")
    if (selected.some((d) => !Number.isInteger(d.count) || d.count < 0))
      errs.push(
        "Each dataset count must be 0 (all) or a positive whole number."
      )
  }
  return errs
}

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

// "No tokenizer chosen" needs a value of its own: an empty string is how a
// Select says *nothing is selected*, and this option is a real choice — the one
// every run made before vocabulary builds existed. Mapped back to `undefined`
// on the way out, which is what the config means by it.
const LEARN_WITH_RUN = "learn-with-run"

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
  // In curriculum mode, epochs are per-stage, so drop the run-wide Epochs field.
  const runFields = curriculumOn
    ? RUN_FIELDS.filter((f) => f.key !== "epochs")
    : RUN_FIELDS
  const numFields = isKto
    ? [
        ...RUN_FIELDS,
        { key: "beta" as const, label: "Beta (KL anchor)", step: 0.05 },
      ]
    : isSft
      ? RUN_FIELDS
      : // A named vocabulary carries its own size, so the field would be a control
        // wired to nothing — the engine ignores `max_vocab` whenever one is named.
        [
          ...ARCH_FIELDS.filter(
            (f) => !(f.key === "max_vocab" && cfg.tokenizer_id)
          ),
          ...runFields,
        ]

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
  const headDim =
    cfg.heads > 0 && cfg.dimensions % cfg.heads === 0
      ? cfg.dimensions / cfg.heads
      : null
  const dFf = cfg.dimensions * cfg.ff_multiplier

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
                  tokenizer_id:
                    !v || v === LEARN_WITH_RUN ? undefined : v,
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
            showCounts={false}
            emptyHint={
              <>
                No SFT sets found. Fetch conversations into
                <code className="mx-1 rounded bg-muted px-1 py-0.5 font-mono">
                  engine/data/sft/
                </code>
                first.
              </>
            }
            onChange={(datasets) => onChange({ ...cfg, datasets })}
          />
          {(cfg.datasets?.length ?? 0) === 0 &&
            availableDatasets.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Nothing selected trains on <em>every</em> set on disk, including
                any added later. Pick the sets explicitly — a thinking set
                folded into a no-think run is how reasoning traces get stripped
                and trained as answer-only.
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
