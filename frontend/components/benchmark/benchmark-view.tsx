"use client"

import * as React from "react"

import { Throbber } from "@/components/ui/throbber"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useBenchmark } from "@/hooks/use-benchmark"
import {
  BENCHMARK_SUITES,
  formatDuration,
  isLanguageReport,
  isMarkupReport,
} from "@/lib/benchmark"

import { CorpusPanel } from "./corpus-panel"
import { MarkupPanel } from "./markup-panel"
import { DurationPanel } from "./duration-panel"
import { GenerationPanel } from "./generation-panel"
import { AblationPanel } from "./ablation-panel"
import { HistoryPanel } from "./history-panel"
import { ModelFilter } from "./model-filter"
import { TokensPanel } from "./tokens-panel"
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

  // Models kept *off* the two figures. Stored as the hidden set rather than the
  // visible one so a model that appears later — a benchmark finishing while the
  // page is open, or the whole cast changing when you switch suite — is plotted by
  // default instead of being filtered out by a selection made before it existed.
  const [hidden, setHidden] = React.useState<Set<string>>(new Set())

  const toggleModel = React.useCallback((model: string) => {
    setHidden((current) => {
      const next = new Set(current)
      if (!next.delete(model)) next.add(model)
      return next
    })
  }, [])
  const showAllModels = React.useCallback(() => setHidden(new Set()), [])
  const showOnlyModel = React.useCallback(
    (model: string) => {
      setHidden(
        new Set(
          runs.map((run) => run.model_name).filter((name) => name !== model)
        )
      )
    },
    [runs]
  )

  // What the two figures plot. The history table below is deliberately never
  // filtered: hiding a model is about making a crowded plot readable, and a run
  // missing from the table as well would read as a run that no longer exists.
  const plotted = React.useMemo(
    () => runs.filter((run) => !hidden.has(run.model_name)),
    [runs, hidden]
  )
  // Derived from what is actually on this suite, not from `hidden.size`: the set
  // survives a suite switch, and a name left in it that no run here carries is not a
  // filter this page is under.
  const filtered = plotted.length < runs.length

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

  const switchSuite = (value: unknown) => {
    const suite = BENCHMARK_SUITES.find((one) => one.id === value)
    if (suite) selectSuite(suite.id)
  }

  const switcher = (
    <TabsList variant="line" aria-label="Benchmark">
      {BENCHMARK_SUITES.map((suite) => (
        <TabsTrigger key={suite.id} value={suite.id}>
          {suite.label}
        </TabsTrigger>
      ))}
    </TabsList>
  )

  if (isLoading) {
    return (
      <Tabs value={selectedSuite} onValueChange={switchSuite} className="gap-6">
        {switcher}
        <TabsContent value={selectedSuite} className="space-y-6">
          <Skeleton className="h-72 rounded-(--radius-sheet)" />
          <Skeleton className="h-40 rounded-(--radius-sheet)" />
          <Skeleton className="h-16 rounded-(--radius-sheet)" />
        </TabsContent>
      </Tabs>
    )
  }

  const report = detail?.report ?? null

  return (
    <Tabs value={selectedSuite} onValueChange={switchSuite} className="gap-6">
      {switcher}
      <TabsContent value={selectedSuite} className="space-y-6">
        {error && <p className="text-sm text-destructive">{error}</p>}

        {/* Cross-run views: they read the whole history rather than the selected
          run, and they are what the page opens on.

          Full width, one above the other. They used to sit side by side, which
          halved the plot each figure got: a scatter's whole job is separating
          points that are close together, and at 480px with a name beside every
          dot there was no room left to separate anything. Two tall figures read
          better than four small quadrants, and neither is compared to the other
          mark-for-mark — they share a y axis and nothing else. */}
        <ModelFilter
          runs={runs}
          hidden={hidden}
          onToggle={toggleModel}
          onShowAll={showAllModels}
          onOnly={showOnlyModel}
        />
        <TokensPanel runs={plotted} filtered={filtered} />
        <DurationPanel runs={plotted} filtered={filtered} />

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
              <ScoreCard report={report} />
              {isMarkupReport(report) ? (
                <MarkupPanel report={report} />
              ) : isLanguageReport(report) ? (
                <>
                  <div className="grid gap-6 xl:grid-cols-2">
                    <CorpusPanel corpus={report.corpus} />
                    {report.ablation && (
                      <AblationPanel ablation={report.ablation} />
                    )}
                  </div>
                  {report.surface && (
                    <GenerationPanel
                      surface={report.surface}
                      samples={report.samples}
                    />
                  )}
                </>
              ) : null}
              <p className="text-xs text-muted-foreground tabular-nums">
                Completed in {formatDuration(report.total_secs)}.
              </p>
            </>
          ) : active ? (
            // The first run of all has no earlier report to sit under, which would
            // otherwise leave the page empty while the GPU works. The mark fills that
            // space and reads as "measuring" rather than "stalled".
            <div className="flex items-center gap-4 border-y border-border/60 py-6">
              <Throbber mode="benchmarking" />
              <p className="min-w-0 text-sm text-muted-foreground">
                Scoring {active.model_name}.
              </p>
            </div>
          ) : (
            // Failed runs carry their error in the history row, so this only has to
            // cover "nothing selected yet".
            <p className="border-y border-border/60 py-6 text-sm text-muted-foreground">
              {detail?.status === "failed"
                ? (detail.error ?? "This run failed.")
                : "Select a run above, or queue a benchmark from the jobs page."}
            </p>
          )}
        </div>
      </TabsContent>
    </Tabs>
  )
}
