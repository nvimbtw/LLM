"use client"

import { Throbber } from "@/components/ui/throbber"
import { Skeleton } from "@/components/ui/skeleton"
import { useBenchmark } from "@/hooks/use-benchmark"
import { formatDuration } from "@/lib/benchmark"

import { CorpusPanel } from "./corpus-panel"
import { DurationPanel } from "./duration-panel"
import { GenerationPanel } from "./generation-panel"
import { GrammarPanel } from "./grammar-panel"
import { HistoryPanel } from "./history-panel"
import { ScalingPanel } from "./scaling-panel"
import { ScoreCard } from "./score-card"

export function BenchmarkView() {
  const { runs, active, detail, selectedId, select, remove, isLoading, error } =
    useBenchmark()

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
        <div className="flex flex-col items-center gap-5 rounded-xl border border-dashed border-border/60 px-6 py-12">
          <Throbber mode="benchmarking" className="h-7" />
          <p className="text-center text-sm text-muted-foreground">
            Scoring in progress.
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
            : "Select a run below, or queue a benchmark from the jobs page."}
        </p>
      )}

      {/* Cross-run views: outside the report branch above, since they read the
          whole history rather than the selected run. */}
      <ScalingPanel runs={runs} />
      <DurationPanel runs={runs} />

      <HistoryPanel runs={runs} selectedId={selectedId} onSelect={select} onDelete={remove} />
    </div>
  )
}
