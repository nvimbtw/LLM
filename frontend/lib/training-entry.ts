import { configForMode } from "@/components/training/run-config"
import type { TrainConfig } from "./training"

export type TrainingEntryMode = "pretrain" | "sft"

/** Interprets only the two supported Workbench mode hints; cycle modes come from retained drafts or rerun seeds. */
export function trainingEntryMode(
  value: string | null
): TrainingEntryMode | null {
  return value === "pretrain" || value === "sft" ? value : null
}

/** Chooses initial editor settings without changing a retained draft; explicit resume inherits its checkpoint and fresh entries apply the requested mode. */
export function trainingEntryConfig({
  defaults,
  draft,
  resume,
  mode,
}: {
  defaults: TrainConfig
  draft: TrainConfig | null
  resume: string | null
  mode: TrainingEntryMode | null
}): TrainConfig {
  if (resume) return { ...defaults, resume_from: resume }
  if (draft) return draft
  return mode ? configForMode(defaults, mode) : defaults
}
