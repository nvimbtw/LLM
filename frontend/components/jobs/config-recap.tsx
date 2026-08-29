"use client"

import { Panel } from "@/components/ui/panel"
import { benchmarkBudgetOf, benchmarkSuiteOf, TIER_LABEL } from "@/lib/jobs"
import type { Job } from "@/lib/jobs"
import type { TokenizerBuild } from "@/lib/tokenizers"
import { isFineTune, type TrainConfig, type TrainMode } from "@/lib/training"

/**
 * What a job was asked to do, as queued.
 *
 * Every kind gets one, because the recap is worth most exactly where there is
 * nothing else to show: a job can sit in the queue for hours, and by the time you
 * open it you no longer remember what you asked for. It used to exist only for
 * training, which is why a queued benchmark read as an empty page — the run had
 * plenty to say about itself, and nobody was saying it.
 *
 * `config` overrides the row's stored config, for a live training run whose frames
 * carry the config it is actually executing.
 */
export interface ConfigRecapProps {
  job: Job
  config?: TrainConfig | null
  className?: string
}

export function ConfigRecap({ job, config, className }: ConfigRecapProps) {
  const recap = describe(job, config)
  if (!recap) return null

  return (
    <Panel
      title="Config"
      description={recap.description}
      bodyClassName="p-0"
      className={className}
    >
      <dl className="divide-y divide-border/40">
        {recap.rows.map(([label, value]) => (
          <div
            key={label}
            className="flex items-baseline justify-between gap-3 px-5 py-2"
          >
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="text-right text-xs tabular-nums">{value}</dd>
          </div>
        ))}
        {recap.blocks.map(([label, text]) => (
          <div key={label} className="px-5 py-2">
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="mt-1 text-xs leading-relaxed">{text}</dd>
          </div>
        ))}
      </dl>
    </Panel>
  )
}

/** A recap: short rows, then anything too long to sit on one line. */
interface Recap {
  description?: string
  rows: [string, string][]
  blocks: [string, string][]
}

const MODE_LABEL: Record<TrainMode, string> = {
  pretrain: "Pretrain",
  sft: "Fine-tune",
  kto: "Preference tune",
  raft: "Rejection tune",
}

function describe(job: Job, override?: TrainConfig | null): Recap | null {
  switch (job.kind) {
    case "train":
      return train((override ?? job.config) as TrainConfig | null)
    case "tokenize":
      return tokenize(job.config as TokenizerBuild | null)
    case "benchmark": {
      // The assistant suite was missing here and fell through to the language
      // description, so a queued assistant run described itself as scoring grammar.
      const suite = benchmarkSuiteOf(job)
      const suiteRow =
        suite === "markup"
          ? "Markup — special-token use"
          : suite === "assistant"
            ? "Assistant — held-out reply bits"
            : "Language — text, grammar, generation"
      const budget = benchmarkBudgetOf(job)
      return {
        description: "As queued",
        rows: [
          ["Model", job.label],
          ["Suite", suiteRow],
          // Only when one was forced. An unbudgeted run renders each conversation's
          // own marker, which is the default and not a setting worth a row.
          ...(budget
            ? ([
                [
                  "Thinking budget",
                  `${TIER_LABEL[budget]} — every held-out conversation forced to ${budget}`,
                ],
              ] as [string, string][])
            : []),
        ],
        blocks: [],
      }
    }
    case "chat":
      return null
  }
}

function tokenize(config: TokenizerBuild | null): Recap | null {
  if (!config) return null
  return {
    description: "As queued",
    rows: [["Max vocab", String(config.max_vocab)]],
    // Empty is not "none": the engine resolves it to every corpus on disk, and a
    // blank here would read as a build that learned from nothing.
    blocks: [
      [
        "Corpora",
        config.datasets.map((d) => d.name).join(", ") || "Every corpus on disk",
      ],
    ],
  }
}

function train(config: TrainConfig | null): Recap | null {
  if (!config) return null

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
    // Which vocabulary, before how large: a run that names one ignores
    // `max_vocab` entirely, since the size belongs to the build.
    // What it started from, before what it started with: a run that continued
    // from a state took its architecture from there, so the rows above are the
    // request rather than the last word on it.
    rows.push([
      "Starts from",
      config.resume_from ? "A saved state" : "Random weights",
    ])
    rows.push([
      "Tokenizer",
      config.tokenizer_id ? "A built vocabulary" : "Learned with the run",
    ])
    if (!config.tokenizer_id) rows.push(["Max vocab", String(config.max_vocab)])
    rows.push(["Shuffle", config.shuffle ? "On" : "Off"])
  }
  // SFT only: pretrain always strips, and RAFT/KTO have no thinking to strip, so
  // the row would be noise in both cases.
  if (config.mode === "sft") {
    rows.push(["Strip thinking", config.strip_thinking ? "On" : "Off"])
  }
  if (config.mode === "kto" && config.beta !== undefined) {
    rows.push(["Beta", String(config.beta)])
  }

  // Pretrain builds its corpus from the selected uploads and SFT from the
  // selected conversation sets — both are a dataset pick and both are recapped,
  // because which sets went in is the thing that tells two runs from the same
  // base apart. RAFT and KTO are not: their data is the model's own generations
  // and the rated catalogue, neither of which is chosen here.
  const picksDatasets = !fineTune || config.mode === "sft"
  const datasets = !picksDatasets
    ? ""
    : config.curriculum.length > 0
      ? config.curriculum
          .map((stage, i) => stage.name ?? `Stage ${i + 1}`)
          .join(" → ")
      : // Empty is not "none" — the engine resolves it to every set on disk, so
        // say that rather than showing a blank where a selection should be.
        config.datasets.map((d) => d.name).join(", ") ||
        (config.mode === "sft" ? "Every set on disk" : "")

  return {
    description: fineTune ? "Inherited from the base" : "As queued",
    rows,
    blocks: datasets
      ? [[config.curriculum.length > 0 ? "Curriculum" : "Datasets", datasets]]
      : [],
  }
}
