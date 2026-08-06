import * as React from "react"

import { cn } from "@/lib/utils"

// A single figure in the stat run: a big tabular value over a muted label. No
// box, no fill — the hairline rules between items (owned by StatRun) do the
// separating, so the numbers read like a figure's data line, not a grid of cards.
export interface Stat {
  label: string
  value: string
}

// A horizontal run of stats separated by thin vertical hairline rules. Wraps on
// narrow screens (each item keeps its left rule). Pin it at the top of the main
// column so the headline numbers are legible at a glance without scrolling.
export function StatRun({ stats }: { stats: Stat[] }) {
  return (
    <dl className="flex flex-wrap gap-y-3">
      {stats.map((s) => (
        <div
          key={s.label}
          className="flex min-w-[7rem] flex-col gap-0.5 border-l pr-6 pl-4 first:border-l-0 first:pl-0"
        >
          <dt className="text-xs text-muted-foreground">{s.label}</dt>
          {/* The display face, not the UI one: these are the figures the page
              exists to show, and Fraunces' numerals are where the editorial
              character actually lands. Tabular so columns of them still align. */}
          <dd className="font-display text-2xl font-semibold tabular-nums">
            {s.value}
          </dd>
        </div>
      ))}
    </dl>
  )
}

// A captioned figure with a reserved height, so a figure slot occupies its space
// from first paint and fills in rather than popping. `caption` reads as an
// article's figure caption; `aside` is an optional right-aligned note (e.g. the
// early→late legend). Pass the chart (or a placeholder) as children.
export function Figure({
  caption,
  aside,
  minHeight = "16rem",
  children,
}: {
  caption: React.ReactNode
  aside?: React.ReactNode
  minHeight?: string
  children: React.ReactNode
}) {
  return (
    <figure className="space-y-2">
      <figcaption className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{caption}</span>
        {aside}
      </figcaption>
      <div style={{ minHeight }}>{children}</div>
    </figure>
  )
}

// The quiet placeholder a figure shows before its data exists — a hairline dashed
// frame with a muted caption, filling the figure's reserved height so neighbours
// never move when the real chart arrives.
export function FigurePlaceholder({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        "flex h-full min-h-[inherit] items-center justify-center rounded-md border border-dashed text-sm text-muted-foreground",
        className
      )}
    >
      {children}
    </div>
  )
}
