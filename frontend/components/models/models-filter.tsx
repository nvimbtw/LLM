"use client"

import { Search } from "lucide-react"

import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

/**
 * The registry's one control row: which visibility, and a search. Chips left,
 * search right, on one line.
 *
 * These were previously headings — the page nested a visibility section inside
 * which sat a family section inside which sat the grid, so finding one model meant
 * reading two levels of scaffolding that existed for a registry of a dozen rows.
 * As filters the same facts cost one line and answer a question instead of
 * imposing a structure.
 *
 * **Family is not a chip.** It was, and a second chip group beside the first
 * pushed the search onto a line of its own, so the row read as two rows of
 * unrelated controls under a stat ledger. Family narrowing lives in the search
 * instead — `ModelsGrid` matches the query against the family name — which costs
 * a word typed and buys the row back. How many families exist is still answered,
 * once, by the ledger above.
 *
 * Each remaining filter carries its own count, so choosing one is never a step
 * into an empty result.
 */
export interface ModelsFilterProps {
  visibility: Visibility
  onVisibility: (visibility: Visibility) => void
  counts: Record<Visibility, number>
  query: string
  onQuery: (query: string) => void
  /** Visibility is admin-only: a normal session sees published models and no toggle. */
  showVisibility: boolean
}

export type Visibility = "all" | "published" | "private"

const VISIBILITY_LABEL: Record<Visibility, string> = {
  all: "All",
  published: "Published",
  private: "Private",
}

export function ModelsFilter({
  visibility,
  onVisibility,
  counts,
  query,
  onQuery,
  showVisibility,
}: ModelsFilterProps) {
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-border pb-4">
      {showVisibility && (
        <div className="flex items-center gap-1">
          {(["all", "published", "private"] as const).map((option) => (
            <Chip
              key={option}
              active={visibility === option}
              onClick={() => onVisibility(option)}
            >
              {VISIBILITY_LABEL[option]}
              <Count>{counts[option]}</Count>
            </Chip>
          ))}
        </div>
      )}

      {/* `sm:w-56` and `ml-auto`: on one line the search is the right-hand end of
          the row, and on a narrow viewport it takes the full width below the
          chips rather than shrinking into a slot nothing fits in. */}
      <div className="relative ml-auto w-full sm:w-56">
        <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={query}
          onChange={(e) => onQuery(e.target.value)}
          placeholder="Search models"
          aria-label="Search models"
          className="h-8 pl-8 text-sm"
        />
      </div>
    </div>
  )
}

function Chip({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        // `rounded-(--radius-marker)`, not a capsule: `--radius` exists because this app
        // acknowledges a corner rather than rounding it away, and a row of
        // capsules is the one shape that opts out of that decision.
        "flex items-center gap-1.5 rounded-(--radius-marker) px-2.5 py-1 text-xs font-medium tracking-[0.06em] transition-colors",
        active
          ? "bg-foreground text-background"
          : "text-muted-foreground hover:bg-secondary hover:text-foreground"
      )}
    >
      {children}
    </button>
  )
}

function Count({ children }: { children: React.ReactNode }) {
  return <span className="tabular-nums opacity-50">{children}</span>
}
