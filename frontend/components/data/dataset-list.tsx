"use client"

import Link from "next/link"
import * as React from "react"
import { ArrowRight, GitBranch, Trash2 } from "lucide-react"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { LoadError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import { RunningDot } from "@/components/ui/running-dot"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/toast"
import { deleteDataset, formatBytes, type Dataset } from "@/lib/data"

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

  async function confirmDelete() {
    if (!pending) return
    const name = pending.name
    setPending(null)
    try {
      await deleteDataset(pending.id)
      toast.success(`Deleted ${name}.`)
      onChanged()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : `Could not delete ${name}.`)
    }
  }

  if (error) {
    return (
      <Panel title="Datasets">
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
      title="Datasets"
      description={
        datasets.length === 1 ? "1 stored" : `${datasets.length} stored`
      }
      bodyClassName="p-0"
    >
      {isLoading ? (
        <div className="space-y-3 p-5">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
        </div>
      ) : datasets.length === 0 ? (
        <p className="px-5 py-10 text-center text-sm text-muted-foreground">
          Nothing pulled yet. Find a dataset on the Hub and pull a slice of it.
        </p>
      ) : (
        <ul className="divide-y divide-border">
          {datasets.map((dataset) => (
            <li key={dataset.id} className="flex items-center gap-4 px-5 py-4">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <Link
                    href={`/data/${dataset.id}`}
                    className="truncate text-sm font-medium hover:underline"
                  >
                    {dataset.name}
                  </Link>
                  {dataset.origin === "derived" && (
                    <GitBranch
                      className="size-3.5 shrink-0 text-muted-foreground"
                      aria-label="derived from another dataset"
                    />
                  )}
                  <StatusBadge dataset={dataset} />
                </div>
                <p className="mt-0.5 truncate text-xs text-muted-foreground">
                  {dataset.kind === "sft" ? "Conversations" : "Prose"}
                  {dataset.source_repo ? ` · ${dataset.source_repo}` : ""}
                  {` · ${dataset.document_count.toLocaleString()} documents`}
                  {dataset.byte_size > 0
                    ? ` · ${formatBytes(dataset.byte_size)}`
                    : ""}
                </p>
              </div>

              <Button
                variant="ghost"
                size="icon"
                aria-label={`Delete ${dataset.name}`}
                onClick={() => setPending(dataset)}
              >
                <Trash2 className="size-4" />
              </Button>
              <Link
                href={`/data/${dataset.id}`}
                className="text-muted-foreground hover:text-foreground"
                aria-label={`Open ${dataset.name}`}
              >
                <ArrowRight className="size-4" />
              </Link>
            </li>
          ))}
        </ul>
      )}

      <AlertDialog
        open={pending !== null}
        onOpenChange={(open) => !open && setPending(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {pending?.name}?</AlertDialogTitle>
            <AlertDialogDescription>
              {pending?.document_count.toLocaleString()} documents go with it.
              Datasets cleaned from this one are kept, and lose their link back
              to it.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep it</AlertDialogCancel>
            <AlertDialogAction onClick={confirmDelete}>
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Panel>
  )
}

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
