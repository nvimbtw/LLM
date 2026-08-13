"use client"

import { ConfigRecap } from "@/components/jobs/config-recap"
import { Appendix } from "@/components/training/appendix"
import { DiagnosticsPanel } from "@/components/training/diagnostics-panel"
import { Telemetry } from "@/components/training/telemetry"
import type { useJob } from "@/hooks/use-job"
import type { Job } from "@/lib/jobs"
import { isFineTune, type RunPhase, type TrainConfig } from "@/lib/training"

/**
 * A training job's body: the same telemetry the training page always had, now
 * rendered against a job rather than against a page-local run.
 *
 * The figures come straight from `components/training` — this panel's whole job is
 * to supply them, plus a read-only recap of the config the run was queued with.
 * The recap matters more here than it did on the old page: a job can sit in the
 * queue for hours, and by the time you open it you no longer remember what you
 * asked for.
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

  return (
    <div className="space-y-8">
      <div className="grid gap-6 lg:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]">
        <ConfigRecap job={job} config={config} />
        <Telemetry
          status={runPhase(job)}
          isSft={isSft}
          tokenize={run.tokenize}
          train={run.train}
          loss={run.loss}
          vocabSeries={run.vocabSeries}
          batchSeries={run.batchSeries}
          curriculum={config?.curriculum ?? []}
        />
      </div>

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
 */
function runPhase(job: Job): RunPhase | null {
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
      // A running train job is training. There is no separate phase 1 any more:
      // the engine builds the dataset as part of the run and does not report it,
      // so there is no `tokenize.done` to wait for — keying off it left every
      // running job reading as "tokenizing" for its whole life, which showed the
      // pre-training placeholder instead of the loss chart.
      return "training"
    case "done":
      return "done"
    case "cancelled":
      return "cancelled"
    case "failed":
      return "failed"
  }
}
