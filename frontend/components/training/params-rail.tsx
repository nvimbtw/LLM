"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { PresetControls } from "@/components/training/preset-controls"
import { DEFAULT_CONFIG } from "@/hooks/use-training"
import { useDatasets } from "@/hooks/use-datasets"
import { useModels } from "@/hooks/use-models"
import { cn } from "@/lib/utils"
import type {
  DatasetInfo,
  DatasetSelection,
  TrainConfig,
  TrainMode,
} from "@/lib/training-socket"

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

  if (cfg.mode === "sft") {
    if (!cfg.base_model_id) errs.push("Select a base model to fine-tune.")
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
  // Defensive: a config from a stale source (e.g. a preset saved before this
  // field existed) may lack `datasets`; treat missing as an empty selection so it
  // surfaces as a normal validation error rather than a crash.
  const selected = cfg.datasets ?? []
  if (selected.length === 0)
    errs.push("Select at least one dataset to train on.")
  if (selected.some((d) => !Number.isInteger(d.count) || d.count < 0))
    errs.push("Each dataset count must be 0 (all) or a positive whole number.")
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

  const setField = (key: keyof TrainConfig, value: string) => {
    onChange({
      ...cfg,
      [key]:
        key === "model_name" || key === "base_model_id" ? value : Number(value),
    })
  }

  // Switching mode nudges the learning rate to that mode's sane default — but only
  // if the user hasn't moved it off the *other* mode's default.
  const setMode = (mode: TrainMode) => {
    if (cfg.mode === mode) return
    const next: TrainConfig = { ...cfg, mode }
    if (mode === "sft" && cfg.lr === DEFAULT_CONFIG.lr) next.lr = SFT_DEFAULT_LR
    else if (mode === "pretrain" && cfg.lr === SFT_DEFAULT_LR)
      next.lr = DEFAULT_CONFIG.lr
    onChange(next)
  }

  const isSft = cfg.mode === "sft"
  const numFields = isSft ? RUN_FIELDS : [...ARCH_FIELDS, ...RUN_FIELDS]

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

      {/* Mode toggle */}
      <div className="inline-flex w-full gap-0.5 rounded-md border p-0.5">
        <Button
          type="button"
          size="sm"
          variant={isSft ? "ghost" : "default"}
          className="flex-1"
          disabled={disabled}
          onClick={() => setMode("pretrain")}
        >
          Pretrain
        </Button>
        <Button
          type="button"
          size="sm"
          variant={isSft ? "default" : "ghost"}
          className="flex-1"
          disabled={disabled}
          onClick={() => setMode("sft")}
        >
          Fine-tune
        </Button>
      </div>

      {/* Model name — the rail's headline / run identity */}
      <div className="space-y-1">
        <label className="text-xs text-muted-foreground">Model name</label>
        <Input
          value={cfg.model_name}
          onChange={(e) => setField("model_name", e.target.value)}
          placeholder="my-model"
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
        <DatasetPicker
          available={availableDatasets}
          loading={datasetsLoading}
          value={cfg.datasets ?? []}
          disabled={disabled}
          onChange={(datasets) => onChange({ ...cfg, datasets })}
        />
      )}

      {isSft ? (
        <p className="text-xs text-muted-foreground">
          Architecture (dimensions, context, heads, layers) is inherited from the
          base model — only the run knobs above apply.
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

      <Button type="submit" disabled={disabled || errors.length > 0} className="w-full">
        {disabled
          ? isSft
            ? "Fine-tuning…"
            : "Training…"
          : isSft
            ? "Start fine-tune"
            : "Start training"}
      </Button>
    </form>
  )
}
