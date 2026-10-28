"use client"

import * as React from "react"

import { LoadError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import { ResourceList, ResourceRow } from "@/components/ui/resource-list"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DatasetList } from "@/components/data/dataset-list"
import { HubSearch } from "@/components/data/hub-search"
import { useDataRuns, useStoredDatasets } from "@/hooks/use-data"
import type { DataRun } from "@/lib/data"
import { useSessionToken } from "@/hooks/use-auth"

/**
 * The data page: what is stored, what can be pulled, and what is in flight.
 *
 * Three tabs rather than three stacked panels — browsing the Hub and reviewing
 * what is already here are separate errands, and stacking them would put a
 * search box above the list a returning user actually came for.
 */
export function DataView() {
  const token = useSessionToken()
  return token ? <DataSession key={token} /> : null
}

/** Owns one account's dataset tab, selected deletion and catalogue reads, clearing them on session replacement. */
function DataSession() {
  const [refreshKey, setRefreshKey] = React.useState(0)
  const datasets = useStoredDatasets(refreshKey)
  const runs = useDataRuns(refreshKey)

  const active = runs.data.filter(
    (run) => run.status === "running" || run.status === "queued"
  ).length

  return (
    <Tabs defaultValue="datasets">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <TabsList variant="line">
          <TabsTrigger value="datasets">Datasets</TabsTrigger>
          <TabsTrigger value="browse">Browse</TabsTrigger>
          <TabsTrigger value="runs">
            Runs{active > 0 ? ` (${active})` : ""}
          </TabsTrigger>
        </TabsList>
      </div>

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

export interface RunListProps {
  runs: DataRun[]
  isLoading: boolean
  error: string | null
}

/** Presents stored run progress or the load failure supplied by the data hook. */
function RunList({ runs, isLoading, error }: RunListProps) {
  if (error) {
    return (
      <Panel variant="open" title="Runs">
        <LoadError title="The run list didn't load." hint={error} />
      </Panel>
    )
  }

  return (
    <Panel variant="open" title="Runs" bodyClassName="p-0">
      {isLoading ? (
        <div className="space-y-3 py-4">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      ) : runs.length === 0 ? (
        <p className="py-6 text-sm text-muted-foreground">
          Nothing has run yet.
        </p>
      ) : (
        <ResourceList density="compact">
          {runs.map((run) => (
            <ResourceRow
              density="compact"
              key={run.id}
              className="flex items-center gap-4"
            >
              <div className="min-w-0 flex-1">
                <p className="text-sm break-words">{run.label}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {run.status === "running" && run.stage
                    ? `${run.stage} · ${Math.round(run.progress * 100)}%`
                    : run.status}
                  {run.error ? ` — ${run.error}` : ""}
                </p>
              </div>
            </ResourceRow>
          ))}
        </ResourceList>
      )}
    </Panel>
  )
}
