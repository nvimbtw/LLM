"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { Throbber } from "@/components/ui/throbber"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useBenchmark } from "@/hooks/use-benchmark"
import { BENCHMARK_SUITES } from "@/lib/benchmark"

import { BenchmarkReport } from "./benchmark-report-view"
import { DurationPanel } from "./duration-panel"
import { HistoryPanel } from "./history-panel"
import { ModelFilter } from "./model-filter"
import { TokensPanel } from "./tokens-panel"

/** Compares recorded benchmark runs and opens a selected report at its canonical detail URL. */
export function BenchmarkView() {
  const router = useRouter()
  const {
    runs,
    active,
    detail,
    selectedId,
    remove,
    isLoading,
    error,
    selectedSuite,
    selectSuite,
  } = useBenchmark()

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

  const plotted = React.useMemo(
    () => runs.filter((run) => !hidden.has(run.model_name)),
    [runs, hidden]
  )

  const filtered = plotted.length < runs.length

  /** Opens the selected benchmark's own detail route. */
  const openReport = (id: string) => router.push(`/bench/${id}`)

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
          onSelect={openReport}
          onDelete={remove}
        />
        <div className="flex scroll-mt-6 flex-col gap-6">
          {report && detail ? (
            <BenchmarkReport run={detail} />
          ) : active ? (
            <div className="flex items-center gap-4 border-y border-border/60 py-6">
              <Throbber mode="benchmarking" />
              <p className="min-w-0 text-sm text-muted-foreground">
                Scoring {active.model_name}.
              </p>
            </div>
          ) : (
            <p className="border-y border-border/60 py-6 text-sm text-muted-foreground">
              {detail?.status === "failed"
                ? (detail.error ?? "This run failed.")
                : "Select a run above, or queue a benchmark from Admin."}
            </p>
          )}
        </div>
      </TabsContent>
    </Tabs>
  )
}
