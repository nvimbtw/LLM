"use client"

import * as React from "react"
import { Play } from "lucide-react"

import { Throbber } from "@/components/ui/throbber"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useModels } from "@/hooks/use-models"
import type { BenchmarkRun, SuiteInfo } from "@/lib/benchmark"

export interface RunBarProps {
  suite: SuiteInfo | null
  active: BenchmarkRun | null
  isStarting: boolean
  onStart: (modelId: string) => void
}

export function RunBar({ suite, active, isStarting, onStart }: RunBarProps) {
  const { models, isLoading } = useModels()
  // Only the *explicit* choice is stored; the default is derived from the list.
  // Storing the default instead would mean writing state from an effect the moment
  // models load, for a value that is already computable from what we have.
  const [chosenId, setChosenId] = React.useState("")
  const modelId = chosenId || models[0]?.id || ""

  const busy = Boolean(active) || isStarting
  const percent = active?.progress ? Math.round(active.progress.fraction * 100) : 0

  return (
    <section className="rounded-xl border border-border/60 bg-background">
      <div className="flex flex-wrap items-center gap-3 px-5 py-4">
        {/* The primitive can emit null on clear; the bar always wants a concrete
            id, so a clear falls back to the empty string (which disables Run). */}
        <Select
          value={modelId}
          onValueChange={(value) => setChosenId(value ?? "")}
          disabled={busy || isLoading}
        >
          <SelectTrigger className="w-full min-w-0 sm:w-72">
            {/* Base UI's Select.Value renders the raw `value` — here a UUID — so
                the label has to be looked up explicitly. The render function also
                supersedes `placeholder`, so the empty case is handled here too. */}
            <SelectValue>
              {(value: string | null) =>
                models.find((model) => model.id === value)?.name ??
                (isLoading ? "Loading models…" : "Select a model")
              }
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {models.map((model) => (
              <SelectItem key={model.id} value={model.id}>
                {model.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button onClick={() => onStart(modelId)} disabled={busy || !modelId} className="gap-2">
          <Play className="size-3.5" />
          {busy ? "Running…" : "Run benchmark"}
        </Button>

        <div className="ml-auto text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground/60">
          {suite ? <SuiteStatus suite={suite} /> : null}
        </div>
      </div>

      {/* The bar keeps its own divider. It used to *be* the throbber — a
          full-width rule — which is what made the indicator read as a bar rather
          than as one mark. */}
      <div className="border-b border-border/60" />

      {active && (
        <div className="px-5 pb-3 pt-1">
          <div className="flex items-baseline gap-3">
            {/* Always mounted while a run is live, so stage changes morph the mark
                rather than remounting it. */}
            <Throbber mode="benchmarking" className="h-5 shrink-0" />
            <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
              {active.progress?.stage ?? "Starting…"}
            </span>
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground/70">
              {percent}%
            </span>
          </div>
          <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-secondary">
            <div
              className="h-full rounded-full bg-primary transition-[width] duration-500"
              style={{ width: `${Math.max(percent, 2)}%` }}
            />
          </div>
        </div>
      )}
    </section>
  )
}

// Says whether the reference corpus is installed. Without it a run still produces
// grammar and generation numbers but no perplexity — worth knowing *before*
// spending a run rather than reading an empty panel afterwards.
function SuiteStatus({ suite }: { suite: SuiteInfo }) {
  if (suite.corpus_files === 0) {
    return (
      <span
        className="text-destructive"
        title={`Put held-out .txt files in ${suite.corpus_dir} — see scripts/prepare_eval_corpus.py`}
      >
        No reference corpus
      </span>
    )
  }
  return (
    <span className="tabular-nums" title={suite.corpus_dir}>
      {suite.corpus_files} corpus {suite.corpus_files === 1 ? "file" : "files"}
      {suite.blimp_files > 0 && ` · ${suite.blimp_files} BLiMP`}
    </span>
  )
}
