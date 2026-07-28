"use client"

import { Appendix } from "@/components/training/appendix"
import { DiagnosticsPanel } from "@/components/training/diagnostics-panel"
import { Telemetry } from "@/components/training/telemetry"
import { Panel } from "@/components/ui/panel"
import type { useJob } from "@/hooks/use-job"
import type { Job } from "@/lib/jobs"
import { isFineTune, type RunPhase, type TrainConfig, type TrainMode } from "@/lib/training"

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

const MODE_LABEL: Record<TrainMode, string> = {
  pretrain: "Pretrain",
  sft: "Fine-tune",
  kto: "Preference tune",
  raft: "Rejection tune",
}

function ConfigRecap({ config }: { config: TrainConfig | null }) {
  if (!config) return <div />

  const fineTune = isFineTune(config.mode)

  // A fine-tune's architecture is the base model's — it loads those weights — and
  // the two knobs below it are pretraining's alone: there is no BPE train to cap a
  // vocabulary for, and conversation order is always reshuffled. Listing them for a
  // fine-tune was the whole of the "why does it say 64 dimensions" problem: the
  // numbers came from a form the fine-tune composers never show.
  const rows: [string, string][] = [
    ["Mode", MODE_LABEL[config.mode]],
    ["Dimensions", String(config.dimensions)],
    ["Context", String(config.ctx_window)],
    ["Heads", String(config.heads)],
    ["Layers", String(config.layers)],
    ["Epochs", String(config.epochs)],
    ["Learning rate", config.lr.toExponential(0)],
    ["FF multiplier", `${config.ff_multiplier}×`],
  ]
  if (!fineTune) {
    rows.push(["Max vocab", String(config.max_vocab)])
    rows.push(["Shuffle", config.shuffle ? "On" : "Off"])
  }
  if (config.mode === "kto" && config.beta !== undefined) {
    rows.push(["Beta", String(config.beta)])
  }

  // Pretrain builds its corpus from the selected uploads; a fine-tune's data is
  // the chat corpus or the rated catalogue, neither of which is a dataset pick.
  const datasets = fineTune
    ? ""
    : config.curriculum.length > 0
      ? config.curriculum
          .map((stage, i) => stage.name ?? `Stage ${i + 1}`)
          .join(" → ")
      : config.datasets.map((d) => d.name).join(", ")

  return (
    <Panel
      title="Config"
      description={fineTune ? "Inherited from the base" : "As queued"}
      bodyClassName="p-0"
    >
      <dl className="divide-y divide-border/40">
        {rows.map(([label, value]) => (
          <div
            key={label}
            className="flex items-baseline justify-between gap-3 px-5 py-2"
          >
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
