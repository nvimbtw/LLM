"use client"

import Link from "next/link"
import { ResourceList, ResourceRow } from "@/components/ui/resource-list"
import { CornerDownRight, MoreHorizontal, Trash2, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { KIND_LABEL, jobHref, type Job } from "@/lib/jobs"

export interface ChainBandProps {
  jobs: Job[]
  /** Every job on the board, for resolving a `depends_on` into a label. */
  all: Job[]
  onCancel: (id: string) => void
  onDelete: (id: string) => void
}

/** Renders unnumbered follow-ups with their dependency labels and explicit cancellation/removal actions. */
export function ChainBand({ jobs, all, onCancel, onDelete }: ChainBandProps) {
  if (jobs.length === 0) return null

  const labelOf = (id: string | null) =>
    all.find((job) => job.id_job === id)?.label

  return (
    <ResourceList density="compact">
      {jobs.map((job) => {
        const after = labelOf(job.depends_on)
        return (
          <ResourceRow density="compact" key={job.id_job} className="group">
            <div className="flex items-center gap-3 sm:gap-4">
              <CornerDownRight
                className="size-3.5 shrink-0 text-muted-foreground/40"
                aria-hidden
              />

              <div className="min-w-0 flex-1">
                <Link
                  href={jobHref(job)}
                  className="block text-sm break-words hover:underline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring"
                >
                  {job.label}
                </Link>
                <span className="block text-xs break-words text-muted-foreground">
                  {after ? `after ${after}` : "after the job it follows"}
                </span>
              </div>

              <span className="shrink-0 text-xs text-muted-foreground">
                {KIND_LABEL[job.kind]}
              </span>

              <DropdownMenu>
                <DropdownMenuTrigger
                  render={
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Options for ${job.label}`}
                      className="shrink-0 text-muted-foreground transition-opacity focus-visible:opacity-100 data-[popup-open]:opacity-100 sm:opacity-70 sm:group-hover:opacity-100"
                    >
                      <MoreHorizontal />
                    </Button>
                  }
                />
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => onCancel(job.id_job)}>
                    <X />
                    Cancel
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    variant="destructive"
                    onClick={() => onDelete(job.id_job)}
                  >
                    <Trash2 />
                    Remove
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </ResourceRow>
        )
      })}
    </ResourceList>
  )
}
