"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { DEFAULT_CONFIG } from "@/hooks/use-training"
import { useModels } from "@/hooks/use-models"
import type { TrainConfig, TrainMode } from "@/lib/training-socket"

// Numeric fields split by which mode they apply to. Architecture + dataset knobs
// only make sense for a fresh pretrain; a fine-tune inherits the base model's
// architecture, so in SFT mode only the run knobs (epochs, learning rate) show.
const ARCH_FIELDS: { key: keyof TrainConfig; label: string; step?: number }[] = [
  { key: "dimensions", label: "Dimensions" },
  { key: "ctx_window", label: "Context window" },
  { key: "heads", label: "Heads" },
  { key: "layers", label: "Layers" },
  { key: "ff_multiplier", label: "FF multiplier" },
  { key: "max_vocab", label: "Max vocab" },
  { key: "num_books", label: "Books (0 = all)" },
  { key: "num_synthetic", label: "Synthetic files (0 = none)" },
]

const RUN_FIELDS: { key: keyof TrainConfig; label: string; step?: number }[] = [
  { key: "epochs", label: "Epochs" },
  { key: "lr", label: "Learning rate", step: 0.001 },
]

// SFT wants a much lower learning rate than a from-scratch run — the base model
// is already near a good minimum, so a large step would wash out what it learned.
const SFT_DEFAULT_LR = 0.0001

function isPowerOfTwo(n: number) {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0
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

  // SFT: the architecture comes from the base model, so only the run knobs and
  // the base selection are validated here.
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
      `Heads must divide dimensions evenly (head size = dimensions ÷ heads). ${cfg.dimensions} ÷ ${cfg.heads} isn't whole.`
    )
  if (!Number.isFinite(cfg.lr) || cfg.lr <= 0)
    errs.push("Learning rate must be positive.")
  if (!Number.isFinite(cfg.max_vocab) || cfg.max_vocab < 256)
    errs.push("Max vocab must be at least 256.")
  if (!Number.isInteger(cfg.num_books) || cfg.num_books < 0)
    errs.push("Books must be 0 (all) or a positive whole number.")
  if (!Number.isInteger(cfg.num_synthetic) || cfg.num_synthetic < 0)
    errs.push("Synthetic files must be 0 (none) or a positive whole number.")
  return errs
}

export function ConfigForm({
  disabled,
  onStart,
}: {
  disabled: boolean
  onStart: (cfg: TrainConfig) => void
}) {
  const [cfg, setCfg] = React.useState<TrainConfig>(DEFAULT_CONFIG)
  const { models, isLoading: modelsLoading } = useModels()

  const setField = (key: keyof TrainConfig, value: string) => {
    setCfg((prev) => ({
      ...prev,
      [key]: key === "model_name" || key === "base_model_id" ? value : Number(value),
    }))
  }

  // Switching mode nudges the learning rate to that mode's sane default — but
  // only if the user hasn't moved it off the *other* mode's default.
  const setMode = (mode: TrainMode) => {
    setCfg((prev) => {
      if (prev.mode === mode) return prev
      const next: TrainConfig = { ...prev, mode }
      if (mode === "sft" && prev.lr === DEFAULT_CONFIG.lr) next.lr = SFT_DEFAULT_LR
      else if (mode === "pretrain" && prev.lr === SFT_DEFAULT_LR)
        next.lr = DEFAULT_CONFIG.lr
      return next
    })
  }

  const isSft = cfg.mode === "sft"
  const numFields = isSft ? RUN_FIELDS : [...ARCH_FIELDS, ...RUN_FIELDS]

  // The base id is *derived*, not stored: if the user hasn't explicitly picked
  // one, default to the first available model. This keeps the select controlled
  // and the submit correct without a setState-in-effect to preselect.
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
    <form onSubmit={submit} className="grid grid-cols-2 gap-4 sm:grid-cols-3">
      {/* Mode toggle: pretrain a fresh model vs. fine-tune (SFT) an existing one. */}
      <div className="col-span-2 flex flex-col gap-1.5 sm:col-span-3">
        <span className="text-sm font-medium">Mode</span>
        <div className="inline-flex w-fit gap-0.5 rounded-md border p-0.5">
          <Button
            type="button"
            size="sm"
            variant={isSft ? "ghost" : "default"}
            onClick={() => setMode("pretrain")}
          >
            Pretrain
          </Button>
          <Button
            type="button"
            size="sm"
            variant={isSft ? "default" : "ghost"}
            onClick={() => setMode("sft")}
          >
            Fine-tune (SFT)
          </Button>
        </div>
      </div>

      <label className="col-span-2 flex flex-col gap-1.5 sm:col-span-3">
        <span className="text-sm font-medium">Model name</span>
        <Input
          value={cfg.model_name}
          onChange={(e) => setField("model_name", e.target.value)}
          placeholder="my-model"
        />
      </label>

      {isSft && (
        <label className="col-span-2 flex flex-col gap-1.5 sm:col-span-3">
          <span className="text-sm font-medium">Base model</span>
          <select
            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50"
            value={baseModelId ?? ""}
            onChange={(e) => setField("base_model_id", e.target.value)}
            disabled={modelsLoading || models.length === 0}
          >
            {models.length === 0 && (
              <option value="">
                {modelsLoading ? "Loading models…" : "No models available"}
              </option>
            )}
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
          <span className="text-xs text-muted-foreground">
            The pretrained model to fine-tune. Its architecture (dimensions,
            context, heads, layers) is inherited — only the run knobs below apply.
          </span>
        </label>
      )}

      {numFields.map((f) => (
        <label key={f.key} className="flex flex-col gap-1.5">
          <span className="text-sm font-medium">{f.label}</span>
          <Input
            type="number"
            step={f.step ?? 1}
            value={String(cfg[f.key])}
            onChange={(e) => setField(f.key, e.target.value)}
          />
        </label>
      ))}

      {!isSft && (
        <div className="col-span-2 text-xs text-muted-foreground tabular-nums sm:col-span-3">
          Head size:{" "}
          <span className="font-medium text-foreground">{headDim ?? "—"}</span>{" "}
          (dimensions ÷ heads) · FF hidden:{" "}
          <span className="font-medium text-foreground">{dFf || "—"}</span>{" "}
          (dimensions × FF multiplier)
        </div>
      )}

      {errors.length > 0 && (
        <ul className="col-span-2 space-y-0.5 text-sm text-destructive sm:col-span-3">
          {errors.map((err) => (
            <li key={err}>{err}</li>
          ))}
        </ul>
      )}

      <div className="col-span-2 sm:col-span-3">
        <Button type="submit" disabled={disabled || errors.length > 0}>
          {disabled
            ? isSft
              ? "Fine-tuning…"
              : "Training…"
            : isSft
              ? "Start fine-tune"
              : "Start training"}
        </Button>
      </div>
    </form>
  )
}
