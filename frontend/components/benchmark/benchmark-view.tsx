"use client"

import { Throbber } from "@/components/ui/throbber"
import { Skeleton } from "@/components/ui/skeleton"
import { useBenchmark } from "@/hooks/use-benchmark"
import { formatDuration } from "@/lib/benchmark"

import { CorpusPanel } from "./corpus-panel"
import { GenerationPanel } from "./generation-panel"
import { GrammarPanel } from "./grammar-panel"
import { HistoryPanel } from "./history-panel"
import { RunBar } from "./run-bar"
import { ScoreCard } from "./score-card"

export function BenchmarkView() {
  const {
    runs,
    suite,
    active,
    detail,
    selectedId,
    select,
    start,
    remove,
    isLoading,
    isStarting,
    error,
  } = useBenchmark()

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <Skeleton className="h-16 rounded-xl" />
        <Skeleton className="h-40 rounded-xl" />
        <Skeleton className="h-72 rounded-xl" />
      </div>
    )
  }

  const report = detail?.report ?? null

  return (
    <div className="flex flex-col gap-6">
      <RunBar suite={suite} active={active} isStarting={isStarting} onStart={start} />

      {error && <p className="text-sm text-destructive">{error}</p>}

      {report && detail ? (
        <>
          <ScoreCard report={report} modelName={detail.model_name} />
          <GrammarPanel grammar={report.grammar} />
          <CorpusPanel corpus={report.corpus} />
          <GenerationPanel generation={report.generation} />
          <p className="text-[11px] text-muted-foreground/60 tabular-nums">
            Completed in {formatDuration(report.duration_secs)}.
          </p>
        </>
      ) : active ? (
        // The first run of all has no earlier report to sit under, which would
        // otherwise leave the page empty while the GPU works. The mark fills that
        // space and reads as "measuring" rather than "stalled".
        <div className="flex flex-col gap-5 rounded-xl border border-dashed border-border/60 px-6 py-12">
          <Throbber mode="benchmarking" className="h-6 w-full" />
          <p className="text-center text-sm text-muted-foreground">
            {active.progress?.stage ?? "Starting…"}
          </p>
          <p className="text-center text-[11px] text-muted-foreground/60">
            Scoring {active.model_name}. This takes a minute or two on the GPU.
          </p>
        </div>
      ) : (
        // Failed runs carry their error in the history row, so this only has to
        // cover "nothing selected yet".
        <p className="rounded-xl border border-dashed border-border/60 px-5 py-10 text-center text-sm text-muted-foreground">
          {detail?.status === "failed"
            ? (detail.error ?? "This run failed.")
            : "Select a run below, or benchmark a model to get started."}
        </p>
      )}

      <HistoryPanel runs={runs} selectedId={selectedId} onSelect={select} onDelete={remove} />
    </div>
  )
}
