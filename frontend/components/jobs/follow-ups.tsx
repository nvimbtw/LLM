"use client"

import * as React from "react"
import { Plus, Trash2 } from "lucide-react"

import { ParamsRail } from "@/components/training/params-rail"
import { Button } from "@/components/ui/button"
import { Panel } from "@/components/ui/panel"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ACTIVE_BENCHMARK_SUITES, type BenchmarkSuite } from "@/lib/benchmark"
import type { FollowUp } from "@/lib/jobs"
import type { TrainConfig } from "@/lib/training"
import { DEFAULT_CONFIG } from "@/hooks/use-training"

/**
 * The "Then run" editor: what to do with the model this run produces, once it has
 * one.
 *
 * The GPU is one device and a training run is hours, so the expensive thing is not
 * the compute — it is the gap between a run finishing at 3am and somebody being
 * awake to queue what comes next. A step here is filed at the same moment as the
 * run it follows and starts on its own when that run lands.
 *
 * **A flat list, deliberately.** Each step follows the one above it — step 2
 * fine-tunes what step 1 produced, not what the root produced — which is what
 * "then" reads as. `lib/jobs.ts::nest` is where that becomes the nested shape the
 * server wants; see its comment for why the two differ.
 *
 * A step names no model. That is the whole point of it being a step: the model is
 * whatever the job before it builds, and it is filled in server-side at the moment
 * that becomes true.
 */

/** One step, as this editor holds it: the kind plus its own config state. */
export interface Step {
  // Stable across re-renders and reorders, so React keys a step by identity rather
  // than by index — an index key would carry one step's input state onto another
  // when one above it is removed.
  id: string
  kind: "train" | "benchmark"
  train: TrainConfig
  suite: BenchmarkSuite
  // Validation problems reported up by the step's own editor. Held here so the
  // composer's submit can refuse a chain that would 400.
  errors: string[]
}

/** A fresh step. Fine-tune rather than pretrain: it is what follows a run. */
export function newStep(kind: "train" | "benchmark" = "train"): Step {
  return {
    id: crypto.randomUUID(),
    kind,
    train: {
      ...DEFAULT_CONFIG,
      mode: "sft",
      model_name: "",
      // A fine-tune inherits the base's architecture, so these are placeholders the
      // server replaces at promotion. They still have to pass validation, and the
      // defaults do.
      lr: 0.00001,
      epochs: 2,
    },
    suite: "language",
    errors: [],
  }
}

/** The wire shape of the whole list, in order. */
export function toFollowUps(steps: Step[]): FollowUp[] {
  return steps.map((step) =>
    step.kind === "train"
      ? {
          kind: "train" as const,
          // Stripped for the same reason the rail strips it: the server refuses a
          // follow-up that names its own base.
          config: { ...step.train, base_model_id: undefined },
        }
      : { kind: "benchmark" as const, config: { suite: step.suite } }
  )
}

/**
 * Which model a step works on, in words.
 *
 * Every step takes its model from the step directly above it — but "above" supplies
 * it two different ways, and saying so is the difference between a list that reads
 * correctly and one that quietly lies. A training step *produces* a model; a
 * benchmark step produces a score and passes on the model it *scored*. So a
 * benchmark in the middle of a chain is transparent: the step after it works on the
 * same model the benchmark did. See the backend's `jobs::chain::model_of`.
 */
function modelSource(steps: Step[], index: number, rootLabel: string): string {
  if (index === 0) return `the model ${rootLabel} produces`
  return steps[index - 1].kind === "train"
    ? `the model step ${index} produces`
    : `the model step ${index} scores`
}

/** Every problem across every step, for the composer's submit to block on. */
export function stepErrors(steps: Step[]): string[] {
  return steps.flatMap((step, i) =>
    step.errors.map((err) => `Step ${i + 1}: ${err}`)
  )
}

export interface FollowUpsProps {
  steps: Step[]
  onChange: (steps: Step[]) => void
  disabled: boolean
  /**
   * What the chain hangs off, in words — "this run", "this benchmark". Used in the
   * prose only; the first step always follows the job being composed.
   */
  rootLabel?: string
}

export function FollowUps({
  steps,
  onChange,
  disabled,
  rootLabel = "this run",
}: FollowUpsProps) {
  const update = (id: string, patch: Partial<Step>) =>
    onChange(steps.map((s) => (s.id === id ? { ...s, ...patch } : s)))

  return (
    <Panel
      variant="open"
      title="Then run"
      description={`Queued now, started when ${rootLabel} finishes`}
    >
      <div className="space-y-3">
        {steps.length === 0 ? (
          <p className="text-xs leading-relaxed text-muted-foreground">
            Nothing follows this one. Add a step to fine-tune or score the model
            it produces without waiting up for it — each step runs against what
            the step before it built, and the whole chain is cancelled if any of
            them fails.
          </p>
        ) : (
          steps.map((step, i) => (
            <div key={step.id} className="border-y border-border/60 py-4">
              <div className="flex items-center gap-2">
                <span className="w-6 shrink-0 text-sm text-muted-foreground tabular-nums">
                  {i + 1}.
                </span>
                <Select
                  value={step.kind}
                  onValueChange={(value) =>
                    update(step.id, {
                      kind: (value as "train" | "benchmark" | null) ?? "train",
                      // The other kind's problems are not this kind's problems.
                      errors: [],
                    })
                  }
                  disabled={disabled}
                >
                  <SelectTrigger className="w-44">
                    <SelectValue>
                      {(value: string | null) =>
                        value === "benchmark" ? "Benchmark" : "Train"
                      }
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="train">Train</SelectItem>
                    <SelectItem value="benchmark">Benchmark</SelectItem>
                  </SelectContent>
                </Select>

                {step.kind === "benchmark" && (
                  <Select
                    value={step.suite}
                    onValueChange={(value) =>
                      update(step.id, {
                        suite: (value as BenchmarkSuite | null) ?? "language",
                      })
                    }
                    disabled={disabled}
                  >
                    <SelectTrigger className="w-40">
                      <SelectValue>
                        {(value: string | null) =>
                          ACTIVE_BENCHMARK_SUITES.find((s) => s.id === value)?.label ??
                          "Assistant"
                        }
                      </SelectValue>
                    </SelectTrigger>
                    <SelectContent>
                      {ACTIVE_BENCHMARK_SUITES.map((option) => (
                        <SelectItem key={option.id} value={option.id}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}

                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className="ml-auto"
                  disabled={disabled}
                  aria-label={`Remove step ${i + 1}`}
                  onClick={() =>
                    onChange(steps.filter((s) => s.id !== step.id))
                  }
                >
                  <Trash2 className="size-4" />
                </Button>
              </div>

              <p className="mt-2 pl-8 text-xs text-muted-foreground">
                {step.kind === "train" ? "Trains on " : "Scores "}
                {modelSource(steps, i, rootLabel)}.
              </p>

              {step.kind === "train" && (
                <div className="mt-3 border-t pt-3">
                  <ParamsRail
                    cfg={step.train}
                    onChange={(train) => update(step.id, { train })}
                    disabled={disabled}
                    // Never called: a follow-up rail renders no submit button.
                    onStart={() => {}}
                    followUp
                    onValidity={(errors) => update(step.id, { errors })}
                  />
                </div>
              )}
            </div>
          ))
        )}

        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={disabled}
          onClick={() => onChange([...steps, newStep()])}
          className="gap-2"
        >
          <Plus className="size-4" /> Add step
        </Button>
      </div>
    </Panel>
  )
}
