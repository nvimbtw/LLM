"use client"

import Link from "next/link"
import { CornerDownRight, MoreHorizontal, Trash2, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { KIND_LABEL, jobHref, type Job } from "@/lib/jobs"

/**
 * Jobs queued behind another job: what runs when that one lands.
 *
 * **Separate from the queue band, and not merely for looks.** A queued job has a
 * position — it is third, and third is a fact you can drag. A waiting job has no
 * position at all: it runs when the job it follows finishes, which may be in four
 * hours or never. Numbering it 04 would be a lie, and offering a drag handle would
 * offer a reorder the server has no way to honour. So the ledger's numbered gutter
 * is replaced by the one thing that is true of these rows — what each one is after.
 */
export interface ChainBandProps {
  jobs: Job[]
  /** Every job on the board, for resolving a `depends_on` into a label. */
  all: Job[]
  onCancel: (id: string) => void
  onDelete: (id: string) => void
}

export function ChainBand({ jobs, all, onCancel, onDelete }: ChainBandProps) {
  if (jobs.length === 0) return null

  const labelOf = (id: string | null) =>
    all.find((job) => job.id_job === id)?.label

  return (
    <ul className="border-t border-border/60">
      {jobs.map((job) => {
        const after = labelOf(job.depends_on)
        return (
          <li
            key={job.id_job}
            className="group border-b border-border/40 last:border-0"
          >
            <div className="flex items-center gap-3 px-5 py-3">
              <CornerDownRight
                className="size-3.5 shrink-0 text-muted-foreground/40"
                aria-hidden
              />

              <div className="min-w-0 flex-1">
                <Link
                  href={jobHref(job)}
                  className="block truncate text-sm font-medium hover:underline"
                >
                  {job.label}
                </Link>
                <span className="block truncate text-xs text-muted-foreground">
                  {/* The parent may have scrolled off the board into history, or
                      been deleted. Either way the row still means something on its
                      own, so it says the general thing rather than nothing. */}
                  {after ? `after ${after}` : "after the job it follows"}
                </span>
              </div>

              <span className="shrink-0 text-xs font-medium tracking-[0.14em] text-muted-foreground uppercase">
                {KIND_LABEL[job.kind]}
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
                  {/* Cancelling one takes everything chained behind it too — the
                      server does that, since a job whose base model will never be
                      built can only fail later and more expensively. */}
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
          </li>
        )
      })}
    </ul>
  )
}
