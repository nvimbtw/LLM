"use client"

import { ChevronLeft, ChevronRight } from "lucide-react"

import { Button } from "@/components/ui/button"

/**
 * The history's page control: a position, and a step either way.
 *
 * Deliberately not a numbered pager. Page 7 of the job history is not a place
 * anyone navigates to — the questions asked of this list are "what ran today" and
 * "what ran before that", which is a direction, not a coordinate. A row of numbers
 * would spend more of the frame than the thing it indexes.
 *
 * Renders nothing at all when there is one page, so a short history has no chrome
 * under it saying so.
 */
export interface HistoryPagerProps {
  /** Zero-based. */
  page: number
  pages: number
  onGo: (page: number) => void
}

export function HistoryPager({ page, pages, onGo }: HistoryPagerProps) {
  if (pages <= 1) return null

  return (
    <div className="flex items-center justify-between gap-3 border-t border-border py-3">
      <span className="text-xs text-muted-foreground tabular-nums">
        Page {page + 1} of {pages}
      </span>

      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Newer jobs"
          disabled={page === 0}
          onClick={() => onGo(page - 1)}
        >
          <ChevronLeft />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Older jobs"
          disabled={page >= pages - 1}
          onClick={() => onGo(page + 1)}
        >
          <ChevronRight />
        </Button>
      </div>
    </div>
  )
}
