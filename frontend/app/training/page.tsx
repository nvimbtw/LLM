"use client"

import { ConfigForm } from "@/components/training/config-form"
import { TokenizePanel } from "@/components/training/tokenize-panel"
import { TrainPanel } from "@/components/training/train-panel"
import { useTraining } from "@/hooks/use-training"

export default function TrainingPage() {
  const t = useTraining()

  return (
    <div className="mx-auto w-full max-w-4xl flex-1 space-y-6 px-4 py-8 sm:px-6">
      <header>
        <h1 className="font-heading text-2xl font-semibold tracking-tight">
          Train a model
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Tokenize your uploaded files, then train a transformer — streamed live.
        </p>
      </header>

      <ConfigForm disabled={t.isActive} onStart={t.start} />

      {t.error && (
        <p className="text-sm text-destructive" role="alert">
          {t.error}
        </p>
      )}

      {t.status != null && (
        <TokenizePanel
          tokenize={t.tokenize}
          vocabSeries={t.vocabSeries}
          log={t.log}
          active={t.status === "tokenizing"}
        />
      )}

      {/* The training section appears once tokenization is done (any
          non-tokenizing status), matching the two-phase flow. */}
      {t.status != null && t.status !== "tokenizing" && (
        <TrainPanel
          train={t.train}
          loss={t.loss}
          checkpoints={t.checkpoints}
          status={t.status}
          onCancel={t.cancel}
        />
      )}
    </div>
  )
}
