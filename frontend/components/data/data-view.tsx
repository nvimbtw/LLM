"use client"

import * as React from "react"

import { LoadError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import { RunningDot } from "@/components/ui/running-dot"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DatasetList } from "@/components/data/dataset-list"
import { HubSearch } from "@/components/data/hub-search"
import { useDataRuns, useStoredDatasets } from "@/hooks/use-data"
import type { DataRun } from "@/lib/data"

/**
 * The data page: what is stored, what can be pulled, and what is in flight.
 *
 * Three tabs rather than three stacked panels — browsing the Hub and reviewing
 * what is already here are separate errands, and stacking them would put a
 * search box above the list a returning user actually came for.
 */
export function DataView() {
  const [refreshKey, setRefreshKey] = React.useState(0)
  const datasets = useStoredDatasets(refreshKey)
  const runs = useDataRuns(refreshKey)

  const active = runs.data.filter(
    (run) => run.status === "running" || run.status === "queued"
  ).length

  return (
    <Tabs defaultValue="datasets">
      <TabsList>
        <TabsTrigger value="datasets">Datasets</TabsTrigger>
        <TabsTrigger value="browse">Browse</TabsTrigger>
        <TabsTrigger value="runs">
          Runs{active > 0 ? ` (${active})` : ""}
        </TabsTrigger>
      </TabsList>

      <TabsContent value="datasets" className="mt-6">
        <DatasetList
          datasets={datasets.data}
          isLoading={datasets.isLoading}
          error={datasets.error}
          onChanged={() => setRefreshKey((k) => k + 1)}
        />
      </TabsContent>

      <TabsContent value="browse" className="mt-6">
        <HubSearch onQueued={() => setRefreshKey((k) => k + 1)} />
      </TabsContent>

      <TabsContent value="runs" className="mt-6">
        <RunList
          runs={runs.data}
          isLoading={runs.isLoading}
          error={runs.error}
        />
      </TabsContent>
    </Tabs>
  )
}

function RunList({
  runs,
  isLoading,
  error,
}: {
  runs: DataRun[]
  isLoading: boolean
  error: string | null
}) {
  if (error) {
    return (
      <Panel title="Runs">
        <LoadError title="The run list didn't load." hint={error} />
      </Panel>
    )
  }

  return (
    <Panel title="Runs" description="newest first" bodyClassName="p-0">
      {isLoading ? (
        <div className="space-y-3 p-5">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      ) : runs.length === 0 ? (
        <p className="px-5 py-10 text-center text-sm text-muted-foreground">
          Nothing has run yet.
        </p>
      ) : (
        <ul className="divide-y divide-border">
          {runs.map((run) => (
            <li key={run.id} className="flex items-center gap-4 px-5 py-3">
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm">{run.label}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {run.status === "running" && run.stage
                    ? `${run.stage} · ${Math.round(run.progress * 100)}%`
                    : run.status}
                  {run.error ? ` — ${run.error}` : ""}
                </p>
              </div>
              {run.status === "running" && <RunningDot />}
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}
