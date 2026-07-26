"use client"

import { Appendix } from "@/components/training/appendix"
import { DiagnosticsPanel } from "@/components/training/diagnostics-panel"
import { Telemetry } from "@/components/training/telemetry"
import { Panel } from "@/components/ui/panel"
import type { useJob } from "@/hooks/use-job"
import type { Job } from "@/lib/jobs"
import type { RunPhase, TrainConfig } from "@/lib/training"

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
  const isSft = config?.mode === "sft"

  return (
    <div className="space-y-8">
      <div className="grid gap-6 lg:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]">
        <ConfigRecap config={config} />
        <Telemetry
          status={runPhase(job, run)}
          isSft={isSft}
          tokenize={run.tokenize}
          train={run.train}
          loss={run.loss}
          vocabSeries={run.vocabSeries}
          batchSeries={run.batchSeries}
          curriculum={config?.curriculum ?? []}
          // The globe's frames are the per-report 3-D projections, in order.
          emb3Frames={run.diag.map((d) => d.emb_pca3)}
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
function runPhase(job: Job, run: ReturnType<typeof useJob>): RunPhase | null {
  switch (job.status) {
    case "queued":
      return null
    case "running":
    case "cancelling":
      // Phase 1 finishing is what flips this, before the first loss frame — so a
      // slow GPU start reads as "training, nothing yet" rather than as a
      // tokenizer that has stalled.
      return run.tokenize.done ? "training" : "tokenizing"
    case "done":
      return "done"
    case "cancelled":
      return "cancelled"
    case "failed":
      return "failed"
  }
}

function ConfigRecap({ config }: { config: TrainConfig | null }) {
  if (!config) return <div />

  const rows: [string, string][] = [
    ["Mode", config.mode === "sft" ? "Fine-tune" : "Pretrain"],
    ["Dimensions", String(config.dimensions)],
    ["Context", String(config.ctx_window)],
    ["Heads", String(config.heads)],
    ["Layers", String(config.layers)],
    ["Epochs", String(config.epochs)],
    ["Learning rate", config.lr.toExponential(0)],
    ["FF multiplier", `${config.ff_multiplier}×`],
    ["Max vocab", String(config.max_vocab)],
    ["Shuffle", config.shuffle ? "On" : "Off"],
  ]

  const datasets =
    config.curriculum.length > 0
      ? config.curriculum
          .map((stage, i) => stage.name ?? `Stage ${i + 1}`)
          .join(" → ")
      : config.datasets.map((d) => d.name).join(", ")

  return (
    <Panel title="Config" description="As queued" bodyClassName="p-0">
      <dl className="divide-y divide-border/40">
        {rows.map(([label, value]) => (
          <div key={label} className="flex items-baseline justify-between gap-3 px-5 py-2">
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="text-xs tabular-nums">{value}</dd>
          </div>
        ))}
        {datasets && (
          <div className="px-5 py-2">
            <dt className="text-xs text-muted-foreground">
              {config.curriculum.length > 0 ? "Curriculum" : "Datasets"}
            </dt>
            <dd className="mt-1 text-xs leading-relaxed">{datasets}</dd>
          </div>
        )}
      </dl>
    </Panel>
  )
}
