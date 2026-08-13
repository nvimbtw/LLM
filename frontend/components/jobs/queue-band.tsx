"use client"

import * as React from "react"
import Link from "next/link"
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

/**
 * The waiting line, in the order it will execute.
 *
 * Rows are hairline-ruled and carry a tabular-figure gutter, so the queue reads as
 * a numbered ledger rather than a stack of cards — the position *is* the
 * information, and a number in a fixed-width column is the shortest way to say it.
 *
 * Reordering is a drag, using the native HTML5 drag API rather than a library: the
 * whole interaction is one list of one row type, and a dependency for that would
 * cost more than it saves. A drop sends "put this one after that one" rather than
 * an index, so it lands where it was dropped even if the queue moved mid-drag.
 * "Run next" in each row's menu is the same move for anyone not dragging.
 */
export interface QueueBandProps {
  jobs: Job[]
  paused: boolean
  onReorder: (id: string, afterId: string | null) => void
  onCancel: (id: string) => void
  onDelete: (id: string) => void
}

export function QueueBand({
  jobs,
  paused,
  onReorder,
  onCancel,
  onDelete,
}: QueueBandProps) {
  const [draggingId, setDraggingId] = React.useState<string | null>(null)
  // Where the dragged row would land: the id it would sit after, or null for the
  // front of the queue. Held separately from `draggingId` so the marker can show
  // a drop above the first row, which has no preceding id to name.
  const [dropAfter, setDropAfter] = React.useState<string | null | undefined>(
    undefined
  )

  if (jobs.length === 0) return null

  const commit = () => {
    if (draggingId && dropAfter !== undefined) {
      // Dropping a row back where it already was is a no-op worth skipping: it
      // would otherwise rewrite a rank and nudge every watching client.
      const index = jobs.findIndex((j) => j.id_job === draggingId)
      const currentAfter = index > 0 ? jobs[index - 1].id_job : null
      if (dropAfter !== currentAfter) onReorder(draggingId, dropAfter)
    }
    setDraggingId(null)
    setDropAfter(undefined)
  }

  return (
    <ul className={cn("transition-opacity", paused && "opacity-50")}>
      {jobs.map((job, index) => {
        const isDragging = job.id_job === draggingId
        // The line lands above this row when the drop target is the row before it.
        const showMarkerAbove =
          dropAfter !== undefined &&
          (index === 0
            ? dropAfter === null
            : dropAfter === jobs[index - 1].id_job)

        return (
          <li
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
              // Above the midpoint means "before this row", i.e. after its
              // predecessor; below means "after this row".
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
            className={cn(
              "group relative border-b border-border/40 last:border-0",
              isDragging && "opacity-40"
            )}
          >
            {showMarkerAbove && (
              <span className="pointer-events-none absolute inset-x-5 top-0 h-px bg-primary" />
            )}

            <div className="flex items-center gap-3 px-5 py-3">
              {/* The grip is the drag affordance; the whole row is draggable, but
                  nothing would say so without it. */}
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
                  className="block truncate text-sm font-medium hover:underline"
                >
                  {job.label}
                </Link>
              </div>

              <span className="shrink-0 text-xs font-medium tracking-[0.14em] text-muted-foreground uppercase">
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
                      className="shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100 data-[popup-open]:opacity-100"
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

            {/* A drop past the last row has no following row to draw above, so the
                final row carries the marker on its own bottom edge. */}
            {index === jobs.length - 1 && dropAfter === job.id_job && (
              <span className="pointer-events-none absolute inset-x-5 bottom-0 h-px bg-primary" />
            )}
          </li>
        )
      })}
    </ul>
  )
}
