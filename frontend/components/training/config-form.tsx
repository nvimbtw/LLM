"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { DEFAULT_CONFIG } from "@/hooks/use-training"
import type { TrainConfig } from "@/lib/training-socket"

// Numeric hyperparameters, rendered as a grid of number inputs. model_name is
// handled separately as a text field.
const NUM_FIELDS: { key: keyof TrainConfig; label: string; step?: number }[] = [
  { key: "dimensions", label: "Dimensions" },
  { key: "ctx_window", label: "Context window" },
  { key: "heads", label: "Heads" },
  { key: "layers", label: "Layers" },
  { key: "epochs", label: "Epochs" },
  { key: "ff_multiplier", label: "FF multiplier" },
  { key: "max_vocab", label: "Max vocab" },
  { key: "lr", label: "Learning rate", step: 0.001 },
  { key: "num_books", label: "Books (0 = all)" },
]

function isPowerOfTwo(n: number) {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0
}

// Mirrors the backend's validate_config so the user sees the constraints before
// submitting. Returns a list of human-readable problems (empty = valid).
function validate(cfg: TrainConfig): string[] {
  const errs: string[] = []
  // Name is a display label only (weights live under the model's UUID), so
  // spaces and dots like "Opal 1.1" are allowed — just no surrounding whitespace
  // or control characters.
  const name = cfg.model_name
  if (!name.trim()) errs.push("Model name is required.")
  else if (name !== name.trim())
    errs.push("Model name may not have leading or trailing whitespace.")
  else if (/[\x00-\x1f\x7f]/.test(name))
    errs.push("Model name may not contain control characters.")

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

  const setField = (key: keyof TrainConfig, value: string) => {
    setCfg((prev) => ({
      ...prev,
      [key]: key === "model_name" ? value : Number(value),
    }))
  }

  const errors = validate(cfg)
  const headDim =
    cfg.heads > 0 && cfg.dimensions % cfg.heads === 0
      ? cfg.dimensions / cfg.heads
      : null
  const dFf = cfg.dimensions * cfg.ff_multiplier

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (errors.length > 0) return
    onStart(cfg)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model configuration</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={submit} className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          <label className="col-span-2 flex flex-col gap-1.5 sm:col-span-3">
            <span className="text-sm font-medium">Model name</span>
            <Input
              value={cfg.model_name}
              onChange={(e) => setField("model_name", e.target.value)}
              placeholder="my-model"
            />
          </label>
          {NUM_FIELDS.map((f) => (
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

          <div className="col-span-2 text-xs text-muted-foreground tabular-nums sm:col-span-3">
            Head size:{" "}
            <span className="font-medium text-foreground">
              {headDim ?? "—"}
            </span>{" "}
            (dimensions ÷ heads) · FF hidden:{" "}
            <span className="font-medium text-foreground">{dFf || "—"}</span>{" "}
            (dimensions × FF multiplier)
          </div>

          {errors.length > 0 && (
            <ul className="col-span-2 space-y-0.5 text-sm text-destructive sm:col-span-3">
              {errors.map((err) => (
                <li key={err}>{err}</li>
              ))}
            </ul>
          )}

          <div className="col-span-2 sm:col-span-3">
            <Button type="submit" disabled={disabled || errors.length > 0}>
              {disabled ? "Training…" : "Start training"}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}
