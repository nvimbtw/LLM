"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import { Appendix } from "@/components/training/appendix"
import { ParamsRail } from "@/components/training/params-rail"
import { Telemetry } from "@/components/training/telemetry"
import { TrainingShell } from "@/components/training/training-shell"
import { DEFAULT_CONFIG, useTraining } from "@/hooks/use-training"
import type { TrainConfig } from "@/lib/training-socket"

// SFT wants a much lower learning rate than a from-scratch run (see params-rail).
const SFT_DEFAULT_LR = 0.0001

export default function TrainingPage() {
  const t = useTraining()
  // Config state lives here (not in the rail) so the continuation flow can seed it.
  const [cfg, setCfg] = React.useState<TrainConfig>(DEFAULT_CONFIG)

  // Which mode the telemetry should label itself as: the *running* run's mode when
  // one exists (so a reattached run reads right), else what's staged in the form.
  const runIsSft =
    t.runConfig?.mode === "sft" || t.tokenize.conversations != null
  // The phase stepper follows the staged form when idle and the run when active.
  const displayMode = t.isActive ? (t.runConfig?.mode ?? cfg.mode) : cfg.mode
  const displayName = t.isActive
    ? (t.runConfig?.model_name ?? cfg.model_name)
    : cfg.model_name

  // Seed a fine-tune whose base is the just-finished pretrain — no manual base
  // pick. Switches the rail to SFT mode; the user tweaks epochs/LR then Starts.
  const continueToSft = React.useCallback(() => {
    if (!t.finishedModel) return
    setCfg((prev) => ({
      ...prev,
      mode: "sft",
      base_model_id: t.finishedModel!.id,
      model_name: `${t.finishedModel!.name}-sft`,
      lr: prev.lr === DEFAULT_CONFIG.lr ? SFT_DEFAULT_LR : prev.lr,
    }))
  }, [t.finishedModel])

  const showContinue =
    t.pretrainDone && t.finishedModel != null && !t.isActive && cfg.mode !== "sft"

  const primaryAction = t.isActive ? (
    <Button variant="outline" size="sm" onClick={t.cancel}>
      Cancel
    </Button>
  ) : showContinue ? (
    <Button size="sm" onClick={continueToSft}>
      Continue to SFT
    </Button>
  ) : undefined

  return (
    <TrainingShell
      modelName={displayName}
      status={t.status}
      mode={displayMode}
      pretrainDone={t.pretrainDone}
      primaryAction={primaryAction}
      rail={
        <ParamsRail
          cfg={cfg}
          onChange={setCfg}
          disabled={t.isActive}
          onStart={t.start}
          modelsRefreshKey={t.finishedModel?.id}
        />
      }
      main={
        t.status == null ? (
          <p className="text-sm text-muted-foreground">
            Configure the run in the panel to the left, then start training —
            tokenization and loss stream live here.
          </p>
        ) : (
          <>
            {t.error && (
              <p className="text-sm text-destructive" role="alert">
                {t.error}
              </p>
            )}
            <Telemetry
              status={t.status}
              isSft={runIsSft}
              tokenize={t.tokenize}
              train={t.train}
              loss={t.loss}
              vocabSeries={t.vocabSeries}
              batchSeries={t.batchSeries}
            />
          </>
        )
      }
      appendix={<Appendix log={t.log} checkpoints={t.checkpoints} />}
    />
  )
}
