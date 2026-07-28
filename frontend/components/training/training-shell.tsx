"use client"

import * as React from "react"

import { cn } from "@/lib/utils"
import type { RunPhase, TrainMode } from "@/lib/training"

// A step in the Pretrain › SFT masthead stepper. `state` drives the editorial
// rendering: a filled rule/dot for done, a solid dot for active, muted otherwise.
type StepState = "todo" | "active" | "done"

function Step({ label, state }: { label: string; state: StepState }) {
  return (
    <span className="flex items-center gap-2">
      <span
        className={cn(
          "size-1.5 rounded-full",
          state === "active" && "bg-foreground",
          state === "done" && "bg-chart-2",
          state === "todo" && "bg-muted-foreground/40"
        )}
      />
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

// Two-step Pretrain › SFT indicator. Makes the two activities lateral instead of
// stacked. Reflects the current run's mode + status and the sticky "pretrain
// succeeded" marker, so it reads Pretrain ✓ → SFT active across the continuation.
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
        : "active" // SFT config staged but not yet started
    : "todo"

  return (
    <div className="flex items-center gap-3">
      <Step label="Pretrain" state={pretrainState} />
      <span aria-hidden className="h-px w-8 bg-border" />
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

function StatusDot({ status }: { status: RunPhase | null }) {
  const running = status === "tokenizing" || status === "training"
  return (
    <span className="flex items-center gap-2 text-sm text-muted-foreground">
      <span
        className={cn(
          "size-2 rounded-full",
          running && "animate-pulse bg-foreground",
          status === "done" && "bg-chart-2",
          (status === "failed" || status === "cancelled") && "bg-destructive",
          status == null && "bg-muted-foreground/40"
        )}
      />
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
  // Full-width deep-dive section rendered below the rail+main grid, spanning the
  // whole content width (the diagnostics charts stack here).
  diagnostics?: React.ReactNode
  appendix: React.ReactNode
}

// The editorial spread: a masthead (identity · status · phase stepper · primary
// action) over a wide two-column body — a fixed params rail beside a fluid main
// column. Every region is always present and fills in; nothing mounts/unmounts as
// the run progresses, so the layout never pops or jumps.
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
      {/* Masthead */}
      <header className="flex flex-wrap items-center justify-between gap-4 pb-3">
        <div className="flex min-w-0 items-center gap-3">
          <h1 className="font-heading text-2xl font-semibold tracking-tight">
            {modelName || "Train a model"}
          </h1>
          <StatusDot status={status} />
        </div>
        <div className="flex items-center gap-6">
          <PhaseStepper
            status={status}
            mode={mode}
            pretrainDone={pretrainDone}
          />
          {primaryAction}
        </div>
      </header>
      {/* The masthead keeps its own hairline. It used to be the throbber itself —
          a full-width rule doubling as the border — which is exactly what made the
          indicator too wide to read as one mark. */}
      <div className="border-b border-border" />

      {/* Body: fixed rail + fluid main */}
      <div className="grid grid-cols-1 gap-8 pt-6 lg:grid-cols-[340px_minmax(0,1fr)] lg:gap-12">
        <aside className="lg:sticky lg:top-6 lg:self-start">{rail}</aside>
        <main className="min-w-0 space-y-8">{main}</main>
      </div>

      {/* Full-width deep-dive: the diagnostics stack, then the appendix, both
          spanning the whole content width below the rail+hero section. */}
      {(diagnostics || appendix) && (
        <div className="mt-8 space-y-8">
          {diagnostics}
          {appendix}
        </div>
      )}
    </div>
  )
}
