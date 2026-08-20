"use client"

import { Button } from "@/components/ui/button"
import { derived } from "@/components/training/run-config"
import { isFineTune, type DatasetInfo, type TrainConfig } from "@/lib/training"
import { cn } from "@/lib/utils"

/**
 * What is about to be queued, read back as a spec sheet.
 *
 * The board's fields are a dozen separate decisions; this is the one place they
 * are one thing. It states what the run *is* rather than what was typed —
 * dataset counts resolved against the catalogue, head size and feed-forward
 * width worked out — so the numbers that only exist as a consequence are visible
 * before the run rather than after it.
 *
 * Deliberately the same idiom `/models/[id]` uses for a model that already
 * exists: a run being composed and a model being read are the same object at two
 * moments of its life, and they should look it.
 */
export interface BuildSheetProps {
  cfg: TrainConfig
  /** The base model's display name, once the picker has loaded it. */
  baseName?: string
  /** The catalogue, for resolving a selection into a document count. */
  datasets: DatasetInfo[]
  errors: string[]
  disabled: boolean
}

export function BuildSheet({
  cfg,
  baseName,
  datasets,
  errors,
  disabled,
}: BuildSheetProps) {
  const isSft = isFineTune(cfg.mode)
  const { headDim, dFf } = derived(cfg)
  const selection = selected(cfg)
  const documents = selection.reduce((sum, one) => {
    const info = datasets.find((d) => d.id === one.id)
    if (!info) return sum
    return sum + (one.count > 0 ? Math.min(one.count, info.documents) : info.documents)
  }, 0)

  const stages = cfg.curriculum?.length ?? 0

  return (
    <div className="overflow-hidden rounded-(--radius-sheet) border border-border bg-card">
      <header className="border-b border-border px-5 py-4">
        <p className="eyebrow text-muted-foreground">
          {isSft ? "Fine-tune" : "Pretrain"}
        </p>
        <h2
          className={cn(
            "mt-1 font-display text-xl font-semibold tracking-tight",
            !cfg.model_name.trim() && "text-muted-foreground"
          )}
        >
          {cfg.model_name.trim() || "Unnamed"}
        </h2>
      </header>

      <dl className="divide-y divide-border/50">
        {isSft && <Spec label="Base" value={baseName ?? "—"} />}
        {!isSft && (
          <Spec
            label="Tokenizer"
            value={cfg.tokenizer_id ? "Named build" : "Learned with the run"}
          />
        )}

        <Spec
          label="Datasets"
          value={selection.length === 0 ? "None picked" : String(selection.length)}
          muted={selection.length === 0}
        />
        {documents > 0 && (
          <Spec label="Documents" value={documents.toLocaleString()} />
        )}
        {stages > 0 && <Spec label="Stages" value={String(stages)} />}

        {!isSft && (
          <>
            <Spec
              label="Shape"
              value={`${cfg.layers}L · ${cfg.heads}H · ${cfg.dimensions}d`}
            />
            <Spec label="Context" value={`${cfg.ctx_window} tokens`} />
            <Spec
              label="Head size"
              value={headDim ? String(headDim) : "Uneven"}
              muted={!headDim}
            />
            <Spec label="FF hidden" value={dFf ? dFf.toLocaleString() : "—"} />
          </>
        )}

        {stages === 0 && <Spec label="Epochs" value={String(cfg.epochs)} />}
        <Spec label="Learning rate" value={format(cfg.lr)} />
        {cfg.mode === "kto" && cfg.beta !== undefined && (
          <Spec label="Beta" value={format(cfg.beta)} />
        )}
      </dl>

      <div className="space-y-3 border-t border-border p-5">
        {errors.length > 0 && !disabled && (
          <ul className="space-y-1 text-xs text-destructive">
            {errors.map((err) => (
              <li key={err}>{err}</li>
            ))}
          </ul>
        )}

        <Button
          type="submit"
          disabled={disabled || errors.length > 0}
          className="w-full"
        >
          {disabled
            ? isSft
              ? "Starting fine-tune…"
              : "Starting training…"
            : isSft
              ? "Start fine-tune"
              : "Start training"}
        </Button>
        <p className="text-center text-xs text-muted-foreground">
          Queued behind whatever is running.
        </p>
      </div>
    </div>
  )
}

function Spec({
  label,
  value,
  muted,
}: {
  label: string
  value: string
  muted?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 px-5 py-2.5">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd
        className={cn(
          "text-right text-xs tabular-nums",
          muted ? "text-muted-foreground" : "font-medium"
        )}
      >
        {value}
      </dd>
    </div>
  )
}

/** The selection a run actually trains on: the stages', or the flat one. */
function selected(cfg: TrainConfig) {
  const stages = cfg.curriculum ?? []
  if (stages.length > 0) return stages.flatMap((stage) => stage.datasets ?? [])
  return cfg.datasets ?? []
}

/**
 * A learning rate reads as 1e-4, not 0.0001 — six leading zeros is a number
 * nobody can compare at a glance, and these span four orders of magnitude
 * between a pretrain and a preference tune.
 */
function format(n: number): string {
  if (!Number.isFinite(n)) return "—"
  if (n !== 0 && Math.abs(n) < 0.001) return n.toExponential(0).replace("e-", "e−")
  return String(n)
}
