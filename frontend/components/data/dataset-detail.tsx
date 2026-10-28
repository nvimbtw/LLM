"use client"

import Link from "next/link"
import * as React from "react"

import { LoadError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import { RunningDot } from "@/components/ui/running-dot"
import { Skeleton } from "@/components/ui/skeleton"
import { PageHeader } from "@/components/ui/page-header"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ProcessingWorkbench } from "@/components/data/processing-workbench"
import { useDataset, useDocuments } from "@/hooks/use-data"
import { documentText, formatBytes, type Dataset } from "@/lib/data"

/** One page of documents. Enough to judge a corpus by, few enough to scroll. */
const PAGE = 10

export interface DatasetDetailProps {
  datasetId: string
}

/** Loads a dataset and presents its documents and cleaning workbench, polling active pulls. */
export function DatasetDetail({ datasetId }: DatasetDetailProps) {
  const [refreshKey, setRefreshKey] = React.useState(0)
  const { data: dataset, isLoading, error } = useDataset(datasetId, refreshKey)

  React.useEffect(() => {
    if (dataset?.status !== "pulling") return
    const timer = setInterval(() => setRefreshKey((k) => k + 1), 2000)
    return () => clearInterval(timer)
  }, [dataset?.status])

  if (error) {
    return (
      <>
        <PageHeader title="Dataset" back={{ href: "/data", label: "Data" }} />
        <LoadError
          title="That dataset didn't load."
          hint={error}
          onRetry={() => setRefreshKey((key) => key + 1)}
        />
      </>
    )
  }
  if (isLoading || !dataset) {
    return (
      <>
        <PageHeader title="Dataset" back={{ href: "/data", label: "Data" }} />
        <Skeleton className="h-64 w-full" />
      </>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title={dataset.name}
        back={{ href: "/data", label: "Data" }}
        meta={
          <span className="flex items-center gap-2 text-xs text-muted-foreground">
            {dataset.kind === "sft" ? "Conversations" : "Prose"} ·{" "}
            {dataset.status === "pulling" && <RunningDot />}
            {dataset.status}
          </span>
        }
      />
      <Summary dataset={dataset} />

      <Tabs defaultValue="documents">
        <TabsList variant="line">
          <TabsTrigger value="documents">Documents</TabsTrigger>
          <TabsTrigger value="clean">Clean</TabsTrigger>
        </TabsList>

        <TabsContent value="documents" className="mt-6">
          <Documents datasetId={datasetId} total={dataset.document_count} />
        </TabsContent>

        <TabsContent value="clean" className="mt-6">
          {dataset.status === "ready" ? (
            <ProcessingWorkbench dataset={dataset} />
          ) : (
            <Panel variant="open" title="Clean">
              <p className="py-6 text-sm text-muted-foreground">
                {dataset.status === "pulling"
                  ? "Still arriving. Cleaning can start once the pull finishes."
                  : "This dataset failed to pull, so there is nothing to clean."}
              </p>
            </Panel>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

/** Presents the dataset’s recorded source, counts and cleaning provenance. */
function Summary({ dataset }: { dataset: Dataset }) {
  return (
    <Panel variant="open" title="Details">
      <dl className="grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2">
        <Row
          label="Documents"
          value={dataset.document_count.toLocaleString()}
        />
        <Row label="Size" value={formatBytes(dataset.byte_size)} />
        <Row
          label="Status"
          value={
            dataset.status === "pulling" ? (
              <span className="flex items-center gap-1.5">
                <RunningDot /> pulling
              </span>
            ) : (
              dataset.status
            )
          }
        />
        <Row
          label="Source"
          value={
            dataset.source_repo ? (
              <a
                href={`https://huggingface.co/datasets/${dataset.source_repo}`}
                target="_blank"
                rel="noreferrer"
                className="hover:underline"
              >
                {dataset.source_repo}
              </a>
            ) : (
              "—"
            )
          }
        />
        {dataset.parent_id && (
          <Row
            label="Cleaned from"
            value={
              <Link
                href={`/data/${dataset.parent_id}`}
                className="hover:underline"
              >
                its source
              </Link>
            }
          />
        )}
      </dl>

      {dataset.steps && dataset.steps.length > 0 && (
        <div className="mt-5 border-t border-border pt-4">
          <p className="mb-2 eyebrow text-muted-foreground">How it was made</p>
          <ol className="space-y-1 text-xs text-muted-foreground">
            {dataset.steps.map((step, index) => (
              <li key={`${step.processor}-${index}`}>
                {index + 1}. {step.processor}
                {step.params && Object.keys(step.params).length > 0
                  ? ` (${Object.entries(step.params)
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(", ")})`
                  : ""}
              </li>
            ))}
          </ol>
        </div>
      )}
    </Panel>
  )
}

/** Aligns one dataset fact with its label, keeping measured figures tabular. */
function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 sm:block">
      <dt className="eyebrow text-muted-foreground">{label}</dt>
      <dd className="tabular-nums sm:mt-0.5">{value}</dd>
    </div>
  )
}

/** Loads one bounded document page and provides navigation through the dataset. */
function Documents({ datasetId, total }: { datasetId: string; total: number }) {
  const [page, setPage] = React.useState(0)
  const { data, isLoading, error } = useDocuments(datasetId, page * PAGE, PAGE)

  if (error) {
    return <LoadError title="Those documents didn't load." hint={error} />
  }

  const pages = Math.max(1, Math.ceil(total / PAGE))

  return (
    <Panel
      variant="open"
      title="Documents"
      description={`${total.toLocaleString()} in total`}
      className="tabular-nums"
      bodyClassName="p-0"
    >
      {isLoading ? (
        <div className="space-y-3 py-4">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-full" />
        </div>
      ) : data.length === 0 ? (
        <p className="py-6 text-sm text-muted-foreground">
          Nothing here yet.
        </p>
      ) : (
        <ul className="divide-y divide-border">
          {data.map((document) => (
            <li key={document.id} className="py-4">
              <p className="mb-1 text-xs text-muted-foreground tabular-nums">
                #{document.ordinal}
              </p>
              <pre className="max-h-40 overflow-auto text-xs whitespace-pre-wrap">
                {documentText(document)}
              </pre>
            </li>
          ))}
        </ul>
      )}

      {pages > 1 && (
        <div className="flex items-center justify-between border-t border-border py-3 text-xs">
          <button
            className="text-muted-foreground hover:text-foreground disabled:opacity-40"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            Previous
          </button>
          <span className="text-muted-foreground tabular-nums">
            {page + 1} of {pages}
          </span>
          <button
            className="text-muted-foreground hover:text-foreground disabled:opacity-40"
            disabled={page + 1 >= pages}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
          </button>
        </div>
      )}
    </Panel>
  )
}
