"use client"

import { EmptyState } from "@/components/ui/empty-state"

import Link from "next/link"
import * as React from "react"
import { GitBranch, Trash2, MoreHorizontal } from "lucide-react"

import { DeleteConfirmation } from "@/components/ui/delete-confirmation"
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"
import { LoadError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import {
  ResourceList,
  ResourceRow,
  ResourceMetadata,
} from "@/components/ui/resource-list"
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
        <EmptyState title="No datasets yet">
          Find a dataset in Browse and pull a slice of it.
        </EmptyState>
      ) : (
        <ResourceList>
          {datasets.map((dataset) => (
            <ResourceRow key={dataset.id} className="flex items-center gap-4">
              <Link
                href={`/data/${dataset.id}`}
                className="min-w-0 flex-1 self-stretch outline-none"
              >
                <div className="flex items-center gap-2">
                  <span className="font-display text-xl leading-snug break-words group-hover:underline">
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
                <ResourceMetadata className="tabular-nums">
                  {dataset.kind === "sft" ? "Conversations" : "Prose"}
                  {dataset.source_repo ? ` · ${dataset.source_repo}` : ""}
                  {` · ${dataset.document_count.toLocaleString()} documents`}
                  {dataset.byte_size > 0
                    ? ` · ${formatBytes(dataset.byte_size)}`
                    : ""}
                </ResourceMetadata>
              </Link>

              {isAdmin && (
                <DropdownMenu>
                  <DropdownMenuTrigger
                    render={
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={`Options for ${dataset.name}`}
                      >
                        <MoreHorizontal />
                      </Button>
                    }
                  />
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      variant="destructive"
                      onClick={() => setPending(dataset)}
                    >
                      <Trash2 />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </ResourceRow>
          ))}
        </ResourceList>
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
        pulling
      </span>
    )
  }
  if (dataset.status === "failed") {
    return <span className="text-xs text-destructive">failed</span>
  }
  return null
}
