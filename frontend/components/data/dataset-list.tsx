"use client"

import Link from "next/link"
import * as React from "react"
import { GitBranch, Trash2 } from "lucide-react"

import { DeleteConfirmation } from "@/components/ui/delete-confirmation"
import { Button } from "@/components/ui/button"
import { LoadError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import { RunningDot } from "@/components/ui/running-dot"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/toast"
import { deleteDataset, formatBytes, type Dataset } from "@/lib/data"
import { useAuth } from "@/hooks/use-auth"

export interface DatasetListProps {
  datasets: Dataset[]
  isLoading: boolean
  error: string | null
  onChanged: () => void
}

/**
 * Everything stored here, newest first.
 *
 * A bordered `ul` rather than a table: there is no table primitive in
 * `components/ui/`, and every other list in this app — the dataset picker, the
 * job history — is this shape.
 */
export function DatasetList({
  datasets,
  isLoading,
  error,
  onChanged,
}: DatasetListProps) {
  const [pending, setPending] = React.useState<Dataset | null>(null)
  const { isAdmin } = useAuth()

  if (error) {
    return (
      <Panel variant="open" title="Datasets">
        <LoadError
          title="The dataset list didn't load."
          hint={error}
          onRetry={onChanged}
        />
      </Panel>
    )
  }

  return (
    <Panel
      variant="open"
      title="Datasets"
      description={
        datasets.length === 1 ? "1 stored" : `${datasets.length} stored`
      }
      bodyClassName="p-0"
    >
      {isLoading ? (
        <div className="space-y-3 py-4">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
        </div>
      ) : datasets.length === 0 ? (
        <p className="py-6 text-sm text-muted-foreground">
          Nothing pulled yet. Find a dataset on the Hub and pull a slice of it.
        </p>
      ) : (
        <ul className="divide-y divide-border">
          {datasets.map((dataset) => (
            <li key={dataset.id} className="group flex items-center gap-4">
              <Link
                href={`/data/${dataset.id}`}
                className="min-w-0 flex-1 self-stretch py-4 outline-none"
              >
                <div className="flex items-center gap-2">
                  <span className="truncate font-display text-xl leading-snug group-hover:underline">
                    {dataset.name}
                  </span>
                  {dataset.origin === "derived" && (
                    <GitBranch
                      className="size-3.5 shrink-0 text-muted-foreground"
                      aria-label="derived from another dataset"
                    />
                  )}
                  <StatusBadge dataset={dataset} />
                </div>
                <p className="mt-0.5 truncate text-xs text-muted-foreground tabular-nums">
                  {dataset.kind === "sft" ? "Conversations" : "Prose"}
                  {dataset.source_repo ? ` · ${dataset.source_repo}` : ""}
                  {` · ${dataset.document_count.toLocaleString()} documents`}
                  {dataset.byte_size > 0
                    ? ` · ${formatBytes(dataset.byte_size)}`
                    : ""}
                </p>
              </Link>

              {isAdmin && (
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={`Delete ${dataset.name}`}
                  onClick={() => setPending(dataset)}
                >
                  <Trash2 className="size-4" />
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}

      {pending && (
        <DeleteConfirmation
          key={pending.id}
          open
          onOpenChange={(open) => {
            if (!open) setPending(null)
          }}
          title={`Delete ${pending.name}?`}
          description={`${pending.document_count.toLocaleString()} documents will be removed. Datasets cleaned from this one are kept and lose their link to it. This cannot be undone.`}
          label="Delete dataset"
          onDelete={() => deleteDataset(pending.id)}
          onDeleted={() => {
            toast.success(`Deleted ${pending.name}.`)
            onChanged()
          }}
        />
      )}
    </Panel>
  )
}

/** Shows an in-progress or failed acquisition state; ready datasets need no badge. */
function StatusBadge({ dataset }: { dataset: Dataset }) {
  if (dataset.status === "pulling") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <RunningDot />
        pulling
      </span>
    )
  }
  if (dataset.status === "failed") {
    return <span className="text-xs text-destructive">failed</span>
  }
  return null
}
