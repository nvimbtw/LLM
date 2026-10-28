"use client"

import * as React from "react"
import { Check } from "lucide-react"

import { cn } from "@/lib/utils"
import { PageHeader } from "@/components/ui/page-header"
import type { RunPhase, TrainMode } from "@/lib/training"

type StepState = "todo" | "active" | "done"

/** Names a training phase and identifies completed work with an explicit check rather than an activity dot. */
function Step({ label, state }: { label: string; state: StepState }) {
  return (
    <span className="flex items-center gap-2">
      {state === "done" && (
        <Check aria-label="Complete" className="size-3 text-status-success" />
      )}
      <span
        className={cn(
          "text-sm",
          state === "todo" ? "text-muted-foreground" : "text-foreground",
          state === "active" && "font-medium"
        )}
      >
        {label}
      </span>
    </span>
  )
}

/** Resolves pre-training and fine-tuning phase states from the actual run and completed predecessor. */
function PhaseStepper({
  status,
  mode,
  pretrainDone,
}: {
  status: RunPhase | null
  mode: TrainMode
  pretrainDone: boolean
}) {
  const running = status === "tokenizing" || status === "training"
  const sft = mode === "sft"

  const pretrainState: StepState = pretrainDone
    ? "done"
    : !sft && running
      ? "active"
      : !sft && status === "done"
        ? "done"
        : "todo"

  const sftState: StepState = sft
    ? running
      ? "active"
      : status === "done"
        ? "done"
        : "active"
    : "todo"

  return (
    <div className="flex items-center gap-3">
      <Step label="Pretrain" state={pretrainState} />
      <span aria-hidden className="text-muted-foreground">
        →
      </span>
      <Step label="SFT" state={sftState} />
    </div>
  )
}

const STATUS_LABEL: Record<RunPhase, string> = {
  tokenizing: "tokenizing",
  training: "training",
  done: "done",
  cancelled: "cancelled",
  failed: "failed",
}

/** Displays the current run state as text, including a truthful idle state before execution. */
function StatusLabel({ status }: { status: RunPhase | null }) {
  return (
    <span className="text-sm text-muted-foreground">
      {status ? STATUS_LABEL[status] : "idle"}
    </span>
  )
}

export interface TrainingShellProps {
  modelName: string
  status: RunPhase | null
  mode: TrainMode
  pretrainDone: boolean
  primaryAction?: React.ReactNode
  rail: React.ReactNode
  main: React.ReactNode
  diagnostics?: React.ReactNode
  appendix: React.ReactNode
}

/** Arranges training identity, phase controls, parameters and telemetry without decorative section rules. */
export function TrainingShell({
  modelName,
  status,
  mode,
  pretrainDone,
  primaryAction,
  rail,
  main,
  diagnostics,
  appendix,
}: TrainingShellProps) {
  return (
    <div className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 sm:px-8">
      <PageHeader
        title={modelName || "Train a model"}
        meta={<StatusLabel status={status} />}
        action={
          <div className="flex items-center gap-6">
            <PhaseStepper
              status={status}
              mode={mode}
              pretrainDone={pretrainDone}
            />
            {primaryAction}
          </div>
        }
      />
      <div className="grid grid-cols-1 gap-8 pt-6 lg:grid-cols-[340px_minmax(0,1fr)] lg:gap-12">
        <aside className="lg:sticky lg:top-6 lg:self-start">{rail}</aside>
        <main className="min-w-0 space-y-8">{main}</main>
      </div>

      {(diagnostics || appendix) && (
        <div className="mt-8 space-y-8">
          {diagnostics}
          {appendix}
        </div>
      )}
    </div>
  )
}
