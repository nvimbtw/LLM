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
 * The admin form keeps this recap as an open ruled ledger at the end of its fields.
 */
export interface BuildSheetProps {
  cfg: TrainConfig
  /** The base model's display name, once the picker has loaded it. */
  baseName?: string
  /**
   * The display name of what a pretrain is continuing from, when that is a
   * finished model. Absent for a checkpoint: those are not in the model list,
   * by design, and the sheet says which run position it is instead.
   */
  resumeName?: string
  /** The catalogue, for resolving a selection into a document count. */
  datasets: DatasetInfo[]
  errors: string[]
  disabled: boolean
}

/** Summarizes the effective run after its editable fields and disables queueing while validation fails or submission is pending. Inherited shape and vocabulary remain explicit. */
export function BuildSheet({
  cfg,
  baseName,
  resumeName,
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
    return (
      sum +
      (one.count > 0 ? Math.min(one.count, info.documents) : info.documents)
    )
  }, 0)

  const stages = cfg.curriculum?.length ?? 0

  return (
    <div className="min-w-0 border-t border-border/60">
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 pt-6 pb-4">
        <h2 className="font-display text-xl">Review</h2>
        <p className="min-w-0 text-xs break-words text-muted-foreground">
          {cfg.model_name.trim() || "Unnamed"} ·{" "}
          {isSft ? "Fine-tune" : "Pretrain"}
        </p>
      </header>

      <dl className="grid gap-x-8 sm:grid-cols-2 lg:grid-cols-3">
        {isSft && <Spec label="Base" value={baseName ?? "—"} />}
        {!isSft && (
          <>
            <Spec
              label="Starts from"
              value={
                !cfg.resume_from
                  ? "Random weights"
                  : (resumeName ?? "A checkpoint")
              }
              muted={!cfg.resume_from}
            />
            <Spec
              label="Tokenizer"
              value={
                cfg.resume_from
                  ? "From the checkpoint"
                  : cfg.tokenizer_id
                    ? "Named build"
                    : "Learned with the run"
              }
            />
          </>
        )}

        <Spec
          label="Datasets"
          value={
            selection.length === 0 ? "None picked" : String(selection.length)
          }
          muted={selection.length === 0}
        />
        {documents > 0 && (
          <Spec label="Documents" value={documents.toLocaleString()} />
        )}
        {stages > 0 && <Spec label="Stages" value={String(stages)} />}

        {!isSft &&
          (cfg.resume_from ? (
            <Spec label="Shape" value="From the checkpoint" muted />
          ) : cfg.architecture ? (
            <>
              <Spec
                label="Recipe revision"
                value={cfg.architecture.revision_id}
              />
              <Spec label="Context" value="From the selected recipe" />
              <Spec
                label="Parameter bindings"
                value={String(Object.keys(cfg.architecture.parameters).length)}
              />
              <Spec
                label="Numerical preflight"
                value="Required in the training job"
              />
            </>
          ) : (
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
              <Spec
                label="FF hidden"
                value={dFf ? dFf.toLocaleString() : "—"}
              />
            </>
          ))}

        {stages === 0 && <Spec label="Epochs" value={String(cfg.epochs)} />}
        <Spec label="Learning rate" value={format(cfg.lr)} />
        {cfg.mode === "kto" && cfg.beta !== undefined && (
          <Spec label="Beta" value={format(cfg.beta)} />
        )}
      </dl>

      <div className="flex flex-wrap items-center justify-between gap-4 py-4">
        {errors.length > 0 && !disabled && (
          <ul className="w-full space-y-1 text-xs text-destructive">
            {errors.map((err) => (
              <li key={err}>{err}</li>
            ))}
          </ul>
        )}

        <Button
          type="submit"
          disabled={disabled || errors.length > 0}
          className="w-full sm:order-last sm:w-auto"
        >
          {disabled
            ? isSft
              ? "Starting fine-tune…"
              : "Starting training…"
            : isSft
              ? "Start fine-tune"
              : "Start training"}
        </Button>
        <p className="text-xs text-muted-foreground">
          Queued behind whatever is running.
        </p>
      </div>
    </div>
  )
}

/** Displays one derived run fact, muting values that are absent or inherited. */
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
    <div className="flex min-w-0 items-baseline justify-between gap-3 border-b border-border/40 py-2.5">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd
        className={cn(
          "min-w-0 text-right text-xs break-words tabular-nums",
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
  if (n !== 0 && Math.abs(n) < 0.001)
    return n.toExponential(0).replace("e-", "e−")
  return String(n)
}
