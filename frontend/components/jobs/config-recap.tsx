"use client"

import { Panel } from "@/components/ui/panel"
import {
  benchmarkBudgetOf,
  benchmarkSuiteOf,
  validationConfig,
  trainConfig,
  TIER_LABEL,
} from "@/lib/jobs"
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

/** Renders the queued family's actual configuration, including exact validation snapshot/profile identity. */
export function ConfigRecap({ job, config, className }: ConfigRecapProps) {
  const recap = describe(job, config)
  if (!recap) return null

  return (
    <Panel
      variant="open"
      title="Config"
      description={recap.description}
      bodyClassName="p-0"
      className={className}
    >
      <dl className="divide-y divide-border/40">
        {recap.rows.map(([label, value]) => (
          <div
            key={label}
            className="flex items-baseline justify-between gap-3 py-2"
          >
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="min-w-0 text-right text-xs break-all tabular-nums">
              {value}
            </dd>
          </div>
        ))}
        {recap.blocks.map(([label, text]) => (
          <div key={label} className="py-2">
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="mt-1 text-xs leading-relaxed break-words">{text}</dd>
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

/** Describes the actual queued family configuration, keeping validation identity/profile separate from model training settings. */
function describe(job: Job, override?: TrainConfig | null): Recap | null {
  switch (job.kind) {
    case "train":
      return train(override ?? trainConfig(job))
    case "tokenize":
      return tokenize(job.config as TokenizerBuild | null)
    case "benchmark": {
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
    case "rl": {
      const cfg = (job.config ?? {}) as Record<string, unknown>
      return {
        description: "As queued",
        rows: [
          ["Gym", String(cfg.gym_id ?? "bash")],
          [
            "Progression",
            cfg.curriculum
              ? "Automatic, with validation and fixed tests"
              : "Selected tasks",
          ],
          ["Iterations", String(cfg.iterations ?? 50)],
          ["Tasks per iteration", String(cfg.tasks_per_batch ?? 4)],
          ["Attempts per task", String(cfg.group_size ?? 8)],
        ],
        blocks: [],
      }
    }
    case "model_validation": {
      const request = validationConfig(job)
      if (!request) return null
      const subject = request.subject
      const profile = request.profile
      return {
        description: "Saved validation snapshot",
        rows: [
          [
            "Mode",
            request.mode === "inspect"
              ? "Declaration inspection"
              : "Full validation",
          ],
          [
            "Subject",
            subject.kind === "draft"
              ? `Draft ${subject.draft_version}`
              : "Accepted revision",
          ],
          [
            "Source identity",
            subject.kind === "draft"
              ? subject.content_hash
              : subject.reference.content_hash,
          ],
          ["Parameters", String(Object.keys(request.parameters ?? {}).length)],
          ...(profile
            ? ([
                ["Device", profile.device],
                ["Precision", profile.precision],
                ["Modes", profile.modes.join(", ")],
              ] as [string, string][])
            : []),
        ],
        blocks: [],
      }
    }
  }
}

function tokenize(config: TokenizerBuild | null): Recap | null {
  if (!config) return null
  return {
    description: "As queued",
    rows: [["Max vocab", String(config.max_vocab)]],
    // Empty is not "none": the LLM engine resolves it to every corpus on disk, and a
    // blank here would read as a build that learned from nothing.
    blocks: [
      [
        "Corpora",
        config.datasets.map((d) => d.name).join(", ") || "Every corpus on disk",
      ],
    ],
  }
}

/** Summarizes actual queued bindings and inherited definitions, never filling missing modular/SFT structure from legacy form defaults. */
function train(config: TrainConfig | null): Recap | null {
  if (!config) return null
  const fineTune = isFineTune(config.mode)
  const inherited = fineTune || Boolean(config.resume_from)
  const rows: [string, string][] = [
    ["Mode", MODE_LABEL[config.mode] ?? "Training"],
  ]
  if (config.resume_from)
    rows.push(["Architecture", "Inherited from the checkpoint"])
  else if (fineTune)
    rows.push(["Architecture", "Inherited from the saved base model"])
  else if (config.architecture)
    rows.push(
      ["Recipe revision", config.architecture.revision_id],
      ["Numerical preflight", "Required in the training job"]
    )
  else
    for (const [label, value] of [
      ["Dimensions", config.dimensions],
      ["Context", config.ctx_window],
      ["Heads", config.heads],
      ["Layers", config.layers],
      ["FF multiplier", config.ff_multiplier],
    ] as const)
      rows.push([label, Number.isFinite(value) ? String(value) : "Unavailable"])
  rows.push(
    [
      "Epochs",
      Number.isFinite(config.epochs) ? String(config.epochs) : "Unavailable",
    ],
    [
      "Learning rate",
      Number.isFinite(config.lr) ? config.lr.toExponential(0) : "Unavailable",
    ],
    [
      "Batch size",
      Number.isFinite(config.batch_size)
        ? String(config.batch_size)
        : "Unavailable",
    ]
  )
  if (inherited)
    rows.push([
      "Tokenizer",
      config.resume_from
        ? "Inherited from the checkpoint"
        : "Inherited from the saved base model",
    ])
  else {
    rows.push(
      ["Starts from", "Random weights"],
      [
        "Tokenizer",
        config.tokenizer_id ? "A built vocabulary" : "Learned with the run",
      ]
    )
    if (!config.tokenizer_id)
      rows.push([
        "Max vocab",
        Number.isFinite(config.max_vocab)
          ? String(config.max_vocab)
          : "Unavailable",
      ])
  }
  if (!fineTune) rows.push(["Shuffle", config.shuffle ? "On" : "Off"])
  if (config.mode === "sft")
    rows.push(["Strip thinking", config.strip_thinking ? "On" : "Off"])
  if (config.mode === "kto" && config.beta !== undefined)
    rows.push(["Beta", String(config.beta)])
  const curriculum = config.curriculum ?? []
  const picksDatasets = !fineTune || config.mode === "sft"
  const datasets = !picksDatasets
    ? ""
    : curriculum.length
      ? curriculum
          .map((stage, index) => stage.name ?? `Stage ${index + 1}`)
          .join(" → ")
      : (config.datasets ?? [])
          .map((dataset) => dataset.name ?? dataset.id)
          .join(", ")
  const blocks: [string, string][] = datasets
    ? [[curriculum.length ? "Curriculum" : "Datasets", datasets]]
    : []
  if (!inherited && config.architecture)
    blocks.unshift([
      "Recipe parameters",
      Object.entries(config.architecture.parameters)
        .map(([name, value]) => `${name}=${String(value)}`)
        .join(", ") || "Declared defaults",
    ])
  return {
    description: inherited ? "Inherited saved definition" : "As queued",
    rows,
    blocks,
  }
}
