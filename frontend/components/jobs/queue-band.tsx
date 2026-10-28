"use client"

import * as React from "react"
import Link from "next/link"
import { ResourceList, ResourceRow } from "@/components/ui/resource-list"
import { ArrowUp, GripVertical, MoreHorizontal, Trash2, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { KIND_LABEL, formatClock, jobHref, type Job } from "@/lib/jobs"
import { cn } from "@/lib/utils"

export interface QueueBandProps {
  jobs: Job[]
  paused: boolean
  onReorder: (id: string, afterId: string | null) => void
  onCancel: (id: string) => void
  onDelete: (id: string) => void
}

/** Renders ordered waiting jobs with drag reordering and equivalent Run next actions; dispatch pause does not stop running work. */
export function QueueBand({
  jobs,
  paused,
  onReorder,
  onCancel,
  onDelete,
}: QueueBandProps) {
  const [draggingId, setDraggingId] = React.useState<string | null>(null)

  const [dropAfter, setDropAfter] = React.useState<string | null | undefined>(
    undefined
  )

  if (jobs.length === 0) return null

  /** Commits a changed drag position and clears the transient drop indicator. */
  const commit = () => {
    if (draggingId && dropAfter !== undefined) {
      const index = jobs.findIndex((j) => j.id_job === draggingId)
      const currentAfter = index > 0 ? jobs[index - 1].id_job : null
      if (dropAfter !== currentAfter) onReorder(draggingId, dropAfter)
    }
    setDraggingId(null)
    setDropAfter(undefined)
  }

  return (
    <ResourceList
      density="compact"
      className={cn("transition-opacity", paused && "opacity-60")}
    >
      {jobs.map((job, index) => {
        const isDragging = job.id_job === draggingId

        const showMarkerAbove =
          dropAfter !== undefined &&
          (index === 0
            ? dropAfter === null
            : dropAfter === jobs[index - 1].id_job)

        return (
          <ResourceRow
            density="compact"
            key={job.id_job}
            draggable
            onDragStart={() => setDraggingId(job.id_job)}
            onDragEnd={() => {
              setDraggingId(null)
              setDropAfter(undefined)
            }}
            onDragOver={(event) => {
              event.preventDefault()
              if (!draggingId) return

              const box = event.currentTarget.getBoundingClientRect()
              const above = event.clientY < box.top + box.height / 2
              setDropAfter(
                above
                  ? index === 0
                    ? null
                    : jobs[index - 1].id_job
                  : job.id_job
              )
            }}
            onDrop={(event) => {
              event.preventDefault()
              commit()
            }}
            className={cn("group relative", isDragging && "opacity-40")}
          >
            {showMarkerAbove && (
              <span className="pointer-events-none absolute inset-x-0 top-0 h-px bg-primary" />
            )}

            <div className="flex items-center gap-2 sm:gap-4">
              <GripVertical
                className="size-3.5 shrink-0 cursor-grab text-muted-foreground/0 transition-colors group-hover:text-muted-foreground/40"
                aria-hidden
              />

              <span className="w-6 shrink-0 text-right text-xs text-muted-foreground tabular-nums">
                {String(index + 1).padStart(2, "0")}
              </span>

              <div className="min-w-0 flex-1">
                <Link
                  href={jobHref(job)}
                  className="block text-sm break-words hover:underline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring"
                >
                  {job.label}
                </Link>
              </div>

              <span className="shrink-0 text-xs text-muted-foreground">
                {KIND_LABEL[job.kind]}
              </span>

              <span className="hidden w-14 shrink-0 text-right text-xs text-muted-foreground tabular-nums sm:block">
                {formatClock(job.created_at)}
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
                  <DropdownMenuItem
                    disabled={index === 0}
                    onClick={() => onReorder(job.id_job, null)}
                  >
                    <ArrowUp />
                    Run next
                  </DropdownMenuItem>
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
            {index === jobs.length - 1 && dropAfter === job.id_job && (
              <span className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-primary" />
            )}
          </ResourceRow>
        )
      })}
    </ResourceList>
  )
}
