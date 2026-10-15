"use client"

import type { ReactNode } from "react"
import { Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

export type Visibility = "all" | "published" | "private"
export interface ModelsFilterProps {
  visibility: Visibility
  onVisibility: (visibility: Visibility) => void
  counts: Record<Visibility, number>
  query: string
  onQuery: (query: string) => void
  showVisibility: boolean
  action?: ReactNode
}

const VISIBILITY_LABEL: Record<Visibility, string> = {
  all: "All",
  published: "Published",
  private: "Private",
}

/** Narrows the registry by visibility and text, keeping optional admin actions on the same quiet control row. */
export function ModelsFilter({
  visibility,
  onVisibility,
  counts,
  query,
  onQuery,
  showVisibility,
  action,
}: ModelsFilterProps) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
      {showVisibility && (
        <div className="flex items-center gap-4">
          {(["all", "published", "private"] as const).map((option) => (
            <Button
              key={option}
              type="button"
              variant="text"
              size="sm"
              aria-pressed={visibility === option}
              onClick={() => onVisibility(option)}
              className={cn(
                "gap-1.5 rounded-none border-b px-0 pb-1",
                visibility === option
                  ? "border-foreground text-foreground"
                  : "border-transparent text-muted-foreground"
              )}
            >
              {VISIBILITY_LABEL[option]}
              <span className="text-xs text-muted-foreground tabular-nums">
                {counts[option]}
              </span>
            </Button>
          ))}
        </div>
      )}
      <div className="relative ml-auto w-full sm:w-52">
        <Search className="pointer-events-none absolute top-1/2 left-0 size-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          variant="inline"
          value={query}
          onChange={(event) => onQuery(event.target.value)}
          placeholder="Search models"
          aria-label="Search models"
          className="h-8 pl-6 text-sm"
        />
      </div>
      {action}
    </div>
  )
}
