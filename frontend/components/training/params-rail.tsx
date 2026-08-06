"use client"

import * as React from "react"
import { ChevronDown, ChevronRight, ChevronUp, Plus, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { SignalGate } from "@/components/training/signal-gate"
import { PresetControls } from "@/components/training/preset-controls"
import { DEFAULT_CONFIG } from "@/hooks/use-training"
import { useDatasets } from "@/hooks/use-datasets"
import { useModels } from "@/hooks/use-models"
import { useSignals } from "@/hooks/use-signals"
import { cn } from "@/lib/utils"
import { isFineTune } from "@/lib/training"
import type {
  CurriculumStage,
  DatasetInfo,
  DatasetSelection,
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

// Exponent bounds: 2^0 (1) up through 2^13 (8192) is the default ceiling; fields
// that go larger (vocab) pass their own `maxExp`.
const EXP_MIN = 0
const EXP_MAX = 13

// A power-of-two field, edited as its exponent: the input holds `k` and the value
// sent upstream is 2^k. This is what makes cycling work — the native ±1 step (from
// the spinner buttons, the wheel, or ↑/↓) moves the exponent by one, which
// doubles/halves the value (… 16 · 32 · 64 …) instead of walking 65 · 66 · 67. A
// muted "= 64" shows the resolved value so the exponent is never cryptic.
function Pow2Input({
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

// The pretrain dataset picker: one checkbox row per uploads/ subfolder. Checking
// a dataset adds it to the run; the count field then dials how many of its files
// to sample (blank/0 = all), with the total shown as "/ <files>". Reads as an
// extension of the spec-sheet rail — a hairline list, tabular figures, muted
// labels — rather than a separate form.
function DatasetPicker({
  available,
  loading,
  value,
  disabled,
  onChange,
}: {
  available: DatasetInfo[]
  loading: boolean
  value: DatasetSelection[]
  disabled: boolean
  onChange: (next: DatasetSelection[]) => void
}) {
  const selected = new Map(value.map((d) => [d.name, d.count]))

  const toggle = (name: string, on: boolean) => {
    if (on) onChange([...value, { name, count: 0 }])
    else onChange(value.filter((d) => d.name !== name))
  }

  const setCount = (name: string, count: number, max: number) => {
    const clamped = Number.isFinite(count)
      ? Math.min(max, Math.max(0, Math.floor(count)))
      : 0
    onChange(value.map((d) => (d.name === name ? { ...d, count: clamped } : d)))
  }

  const totalFiles = value.reduce((sum, d) => {
    const info = available.find((a) => a.name === d.name)
    if (!info) return sum
    return sum + (d.count > 0 ? Math.min(d.count, info.files) : info.files)
  }, 0)

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Datasets</span>
        {value.length > 0 && (
          <span className="text-xs text-muted-foreground tabular-nums">
            {totalFiles.toLocaleString()} files
          </span>
        )}
      </div>

      {loading ? (
        <p className="rounded-lg border border-dashed px-2.5 py-3 text-center text-xs text-muted-foreground">
          Loading datasets…
        </p>
      ) : available.length === 0 ? (
        <p className="rounded-lg border border-dashed px-2.5 py-3 text-center text-xs text-muted-foreground">
          No datasets found. Add a folder of .txt files under
          <code className="mx-1 rounded bg-muted px-1 py-0.5 font-mono">
            data/raw/uploads/
          </code>
          .
        </p>
      ) : (
        <ul className="divide-y rounded-lg border">
          {available.map((d) => {
            const checked = selected.has(d.name)
            const count = selected.get(d.name) ?? 0
            const rowId = `dataset-${d.name}`
            return (
              <li
                key={d.name}
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
                    onCheckedChange={(on) => toggle(d.name, on)}
                  />
                  <span className="truncate text-sm">{d.name}</span>
                </label>

                <span className="flex w-32 shrink-0 items-center justify-end gap-1.5 tabular-nums">
                  {checked ? (
                    <>
                      <Input
                        type="number"
                        min={0}
                        max={d.files}
                        step={1}
                        value={count === 0 ? "" : String(count)}
                        placeholder="all"
                        disabled={disabled}
                        onChange={(e) =>
                          setCount(d.name, Number(e.target.value), d.files)
                        }
                        className="h-8 w-16 text-right tabular-nums"
                      />
                      <span className="w-12 text-xs text-muted-foreground">
                        / {d.files.toLocaleString()}
                      </span>
                    </>
                  ) : (
                    <span className="pr-1 text-xs text-muted-foreground">
                      {d.files.toLocaleString()} files
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
function validate(cfg: TrainConfig): string[] {
  const errs = validateName(cfg.model_name)

  if (isFineTune(cfg.mode)) {
    if (!cfg.base_model_id) errs.push("Select a base model to fine-tune.")
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

export interface ParamsRailProps {
  cfg: TrainConfig
  onChange: (cfg: TrainConfig) => void
  disabled: boolean
  onStart: (cfg: TrainConfig) => void
  // Advances when a new model is registered, so the base-model picker refetches
  // and the just-trained model is selectable without a reload.
  modelsRefreshKey?: unknown
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
}: ParamsRailProps) {
  const { models, isLoading: modelsLoading } = useModels(modelsRefreshKey)
  const { datasets: availableDatasets, isLoading: datasetsLoading } =
    useDatasets()
  // Only meaningful in preference mode, but hooks can't be conditional — and it
  // is one small count, read once per mount.
  const { status: signals } = useSignals(modelsRefreshKey)

  const setField = (key: keyof TrainConfig, value: string) => {
    onChange({
      ...cfg,
      [key]:
        key === "model_name" || key === "base_model_id" ? value : Number(value),
    })
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
      : [...ARCH_FIELDS, ...runFields]

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
  const effectiveCfg: TrainConfig = isSft
    ? { ...cfg, base_model_id: baseModelId }
    : cfg

  const errors = validate(effectiveCfg)
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

  return (
    <form onSubmit={submit} className="space-y-4">
      <PresetControls
        currentConfig={effectiveCfg}
        disabled={disabled}
        onLoad={onChange}
        modelsRefreshKey={modelsRefreshKey}
      />

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
              variant={cfg.mode === mode ? "default" : "ghost"}
              className="flex-1"
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
        {isSft && (
          <Row label="Base model">
            <select
              className="h-8 w-44 truncate rounded-lg border border-input bg-transparent px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50"
              value={baseModelId ?? ""}
              onChange={(e) => setField("base_model_id", e.target.value)}
              disabled={disabled || modelsLoading || models.length === 0}
            >
              {models.length === 0 && (
                <option value="">
                  {modelsLoading ? "Loading…" : "No models"}
                </option>
              )}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
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

      {(isKto || isRaft) && (
        <SignalGate
          status={signals}
          modelId={baseModelId}
          mode={isRaft ? "raft" : "kto"}
          disabled={disabled}
        />
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

      <Button
        type="submit"
        // A preference run with too little rated data is refused at enqueue, so
        // the button matches that rather than letting the form post into a 400.
        // `signals === null` (still loading, or the count failed) leaves it
        // enabled: the server is the authority, and disabling on a failed read
        // would strand the run behind a fetch that may never succeed.
        disabled={
          disabled ||
          errors.length > 0 ||
          (isKto && signals?.ready === false) ||
          (isRaft && signals?.raft_ready === false)
        }
        className="w-full"
      >
        {disabled
          ? isSft
            ? "Fine-tuning…"
            : "Training…"
          : (isKto && signals?.ready === false) ||
              (isRaft && signals?.raft_ready === false)
            ? "Not enough signals yet"
            : isRaft
              ? "Start self-train"
              : isSft
                ? "Start fine-tune"
                : "Start training"}
      </Button>
    </form>
  )
}
