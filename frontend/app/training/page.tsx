"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import { ConfigForm } from "@/components/training/config-form"
import { Section } from "@/components/training/section"
import { TokenizePanel } from "@/components/training/tokenize-panel"
import { TrainPanel } from "@/components/training/train-panel"
import { useTraining } from "@/hooks/use-training"
import type { TrainConfig } from "@/lib/training-socket"

// One-line recap of a submitted config, shown in the collapsed configuration
// header once a run is under way.
function configSummary(cfg: TrainConfig): string {
  return `${cfg.model_name} · ${cfg.dimensions}d · ${cfg.layers}L · ${cfg.heads}H · ${cfg.epochs} epochs`
}

export default function TrainingPage() {
  const t = useTraining()
  // Remember the config the run was started with so we can recap it once the
  // configuration section collapses.
  const [startedConfig, setStartedConfig] = React.useState<TrainConfig | null>(
    null
  )

  const start = React.useCallback(
    (cfg: TrainConfig) => {
      setStartedConfig(cfg)
      t.start(cfg)
    },
    [t]
  )

  const hasRun = t.status != null
  const tokenizing = t.status === "tokenizing"
  const training = t.status === "training"
  // The training section appears once tokenization is done (any non-tokenizing
  // status), matching the two-phase flow.
  const trainPhase = hasRun && t.status !== "tokenizing"

  return (
    <div className="mx-auto w-full max-w-4xl flex-1 space-y-4 px-4 py-8 sm:px-6">
      <header>
        <h1 className="font-heading text-2xl font-semibold tracking-tight">
          Train a model
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Tokenize your uploaded files, then train a transformer — streamed
          live.
        </p>
      </header>

      <Section
        title="Model configuration"
        active={!hasRun}
        done={hasRun}
        description={
          hasRun && startedConfig ? configSummary(startedConfig) : undefined
        }
      >
        <ConfigForm disabled={t.isActive} onStart={start} />
      </Section>

      {t.error && (
        <p className="text-sm text-destructive" role="alert">
          {t.error}
        </p>
      )}

      {hasRun && (
        <Section
          title="Tokenization"
          active={tokenizing}
          done={trainPhase}
          throbberMode="loading"
          description={
            tokenizing
              ? "Building the BPE tokenizer and corpus from your uploads…"
              : "Tokenizer and corpus built."
          }
        >
          <TokenizePanel
            tokenize={t.tokenize}
            vocabSeries={t.vocabSeries}
            log={t.log}
          />
        </Section>
      )}

      {trainPhase && (
        <Section
          title="Training"
          active={training}
          done={t.status === "done"}
          throbberMode="training"
          description={
            training
              ? "Optimizing the transformer — loss streaming live."
              : `Run ${t.status}.`
          }
          action={
            training ? (
              <Button variant="outline" size="sm" onClick={t.cancel}>
                Cancel
              </Button>
            ) : undefined
          }
        >
          <TrainPanel
            train={t.train}
            loss={t.loss}
            batchSeries={t.batchSeries}
            checkpoints={t.checkpoints}
          />
        </Section>
      )}
    </div>
  )
}
