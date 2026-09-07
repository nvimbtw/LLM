"use client"

import { Appendix } from "@/components/training/appendix"
import { DiagnosticsPanel } from "@/components/training/diagnostics-panel"
import { Telemetry } from "@/components/training/telemetry"
import { TrainingHero } from "@/components/training/training-hero"
import type { useJob } from "@/hooks/use-job"
import type { Job } from "@/lib/jobs"
import { isFineTune, type RunPhase, type TrainConfig } from "@/lib/training"

/**
 * A training job's body: the same telemetry the training page always had, now
 * rendered against a job rather than against a page-local run.
 *
 * The figures come straight from `components/training` — supplying them is this
 * panel's whole job.
 */
export interface JobTrainPanelProps {
  job: Job
  run: ReturnType<typeof useJob>
}

export function JobTrainPanel({ job, run }: JobTrainPanelProps) {
  // The config the run is actually executing. Live frames carry it once attached;
  // before that (queued, or reloaded after it finished) the row's stored config is
  // the same thing, and is always present.
  const config = run.runConfig ?? (job.config as TrainConfig | null)
  // Every fine-tune renders conversations in phase 1 rather than training a BPE
  // tokenizer, which is the whole of what the figures switch on here.
  const isSft = config ? isFineTune(config.mode) : false

  const status = runPhase(job, {
    training: run.train.step != null,
    // Phase 1 has to *say* it is happening. Inferring it from the absence of
    // training frames instead would put "building the corpus" on the screen of
    // any run reporting nothing at all — an engine that predates the phase-1
    // frames, or one that has stopped talking — which is a claim about a run
    // rather than a fact about it.
    building:
      Object.keys(run.tokenize).length > 0 && run.tokenize.done !== true,
  })

  return (
    <div className="space-y-8">
      {/* Everything here spans the width. There is no rail beside it: the config
          recap that used to hold that column renders nothing for a training job,
          because the engine's run row carries no `config` field for it to
          describe — and half a page of empty gutter beside the hero is worse
          than no rail at all. */}
      <TrainingHero
        status={status}
        isSft={isSft}
        tokenize={run.tokenize}
        train={run.train}
        curriculum={config?.curriculum ?? []}
      />

      <Telemetry
        status={status}
        isSft={isSft}
        tokenize={run.tokenize}
        train={run.train}
        loss={run.loss}
        batchSeries={run.batchSeries}
        buildSeries={run.buildSeries}
        curriculum={config?.curriculum ?? []}
        ctxWindow={config?.ctx_window ?? 0}
      />

      <DiagnosticsPanel diag={run.diag} />
      <Appendix log={run.log} checkpoints={run.checkpoints} />
    </div>
  )
}

/**
 * Maps the job's status onto the phase the training figures switch on.
 *
 * The two are genuinely different questions — the job's status says whether the
 * queue is running this row, the phase says which half of the run it is in — so
 * the translation is explicit rather than a shared enum that would have to mean
 * both. A queued job has no phase at all, which is what `null` renders as.
 *
 * A running job is training unless phase 1 is actively saying otherwise. Both
 * halves report, so both are read from what arrived rather than from what did
 * not: the first training frame ends phase 1 whatever the build said, and the
 * build has to be mid-flight for its own view to take the screen. A run that
 * says nothing gets the training view — the one with the figures on it.
 */
function runPhase(
  job: Job,
  { training, building }: { training: boolean; building: boolean }
): RunPhase | null {
  switch (job.status) {
    // Neither has begun, so neither has a phase. `waiting` also never reaches this
    // panel — the detail page renders the queued panel for it — but the switch is
    // total on purpose, so adding a status is a compile error rather than a
    // silently missing case.
    case "queued":
    case "waiting":
      return null
    case "running":
    case "cancelling":
      return !training && building ? "tokenizing" : "training"
    case "done":
      return "done"
    case "cancelled":
      return "cancelled"
    case "failed":
      return "failed"
  }
}
