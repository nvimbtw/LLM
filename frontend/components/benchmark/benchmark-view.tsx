"use client"

import * as React from "react"

import { Throbber } from "@/components/ui/throbber"
import { Skeleton } from "@/components/ui/skeleton"
import { useBenchmark } from "@/hooks/use-benchmark"
import { BENCHMARK_SUITES, formatDuration, isMarkupReport } from "@/lib/benchmark"
import { cn } from "@/lib/utils"

import { CorpusPanel } from "./corpus-panel"
import { MarkupPanel } from "./markup-panel"
import { DurationPanel } from "./duration-panel"
import { GenerationPanel } from "./generation-panel"
import { GrammarPanel } from "./grammar-panel"
import { HistoryPanel } from "./history-panel"
import { ScalingPanel } from "./scaling-panel"
import { ScoreCard } from "./score-card"

/**
 * The benchmark page, in the order the questions get asked.
 *
 * Cross-run first — the curves and the table are how you find the run worth
 * reading — then that run's report underneath. It used to open on one model's
 * numbers with the history buried at the bottom, which answered "how did this one
 * do" before "which one should I look at".
 *
 * Picking a row therefore has to take you to the report, since it is now below
 * the fold: `select` remembers that a scroll is owed, and the effect below spends
 * it once the fetched detail actually belongs to the row you clicked. Scrolling
 * on click instead would aim at the previous model's report and land wrong when
 * the new one swapped in at a different height.
 */
export function BenchmarkView() {
  const {
    runs,
    active,
    detail,
    selectedId,
    select,
    remove,
    isLoading,
    error,
    selectedSuite,
    selectSuite,
  } = useBenchmark()

  const reportRef = React.useRef<HTMLDivElement>(null)
  const scrollOwedTo = React.useRef<string | null>(null)

  const selectAndReveal = React.useCallback(
    (id: string) => {
      scrollOwedTo.current = id
      select(id)
    },
    [select]
  )

  React.useEffect(() => {
    if (!detail || scrollOwedTo.current !== detail.id_run) return
    scrollOwedTo.current = null
    reportRef.current?.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
      block: "start",
    })
  }, [detail])

  const switcher = (
    // Two benchmarks, not two views of one. The blurb is under the tabs rather than
    // in a tooltip because "which of these am I looking at" is the first question
    // anyone arriving here has, and the scores are not comparable across them.
    <div className="flex flex-col gap-2">
      <div
        role="tablist"
        aria-label="Benchmark"
        className="inline-flex self-start rounded-md border border-border p-0.5"
      >
        {BENCHMARK_SUITES.map((suite) => (
          <button
            key={suite.id}
            role="tab"
            type="button"
            aria-selected={selectedSuite === suite.id}
            onClick={() => selectSuite(suite.id)}
            className={cn(
              "rounded px-3 py-1.5 text-sm transition-colors",
              selectedSuite === suite.id
                ? "bg-muted font-medium text-foreground"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {suite.label}
          </button>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        {BENCHMARK_SUITES.find((s) => s.id === selectedSuite)?.blurb}
      </p>
    </div>
  )

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        {switcher}
        <Skeleton className="h-72 rounded-lg" />
        <Skeleton className="h-40 rounded-lg" />
        <Skeleton className="h-16 rounded-lg" />
      </div>
    )
  }

  const report = detail?.report ?? null

  return (
    <div className="flex flex-col gap-6">
      {switcher}
      {error && <p className="text-sm text-destructive">{error}</p>}

      {/* Cross-run views: they read the whole history rather than the selected
          run, and they are what the page opens on. Side by side once there is
          room, because they are two readings of the same set of runs. */}
      <div className="grid gap-6 xl:grid-cols-2">
        <ScalingPanel runs={runs} />
        <DurationPanel runs={runs} />
      </div>

      <HistoryPanel
        runs={runs}
        selectedId={selectedId}
        onSelect={selectAndReveal}
        onDelete={remove}
      />

      {/* scroll-mt keeps the heading clear of the sticky app header when a row
          jumps here, rather than tucking it just underneath. */}
      <div ref={reportRef} className="flex scroll-mt-6 flex-col gap-6">
        {report && detail ? (
          <>
            <ScoreCard report={report} modelName={detail.model_name} />
            {isMarkupReport(report) ? (
              <MarkupPanel report={report} />
            ) : (
              <>
                <div className="grid gap-6 xl:grid-cols-2">
                  <GrammarPanel grammar={report.grammar} />
                  <CorpusPanel corpus={report.corpus} />
                </div>
                <GenerationPanel generation={report.generation} />
              </>
            )}
            <p className="text-xs text-muted-foreground tabular-nums">
              Completed in {formatDuration(report.duration_secs)}.
            </p>
          </>
        ) : active ? (
          // The first run of all has no earlier report to sit under, which would
          // otherwise leave the page empty while the GPU works. The mark fills that
          // space and reads as "measuring" rather than "stalled".
          <div className="flex flex-col items-center gap-5 rounded-lg border border-dashed border-border px-6 py-12">
            <Throbber mode="benchmarking" />
            <p className="text-center text-sm text-muted-foreground">
              Scoring in progress.
            </p>
            <p className="text-center text-xs text-muted-foreground">
              Scoring {active.model_name}. This takes a minute or two on the
              GPU.
            </p>
          </div>
        ) : (
          // Failed runs carry their error in the history row, so this only has to
          // cover "nothing selected yet".
          <p className="rounded-lg border border-dashed border-border px-5 py-10 text-center text-sm text-muted-foreground">
            {detail?.status === "failed"
              ? (detail.error ?? "This run failed.")
              : "Select a run above, or queue a benchmark from the jobs page."}
          </p>
        )}
      </div>
    </div>
  )
}
