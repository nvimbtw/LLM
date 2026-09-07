import * as React from "react"

import { cn } from "@/lib/utils"

// A figure's number: compact, or an em dash for one nothing measured. Shared by
// every training figure so a count reads the same in all of them — and so "not
// measured" is one spelling rather than each caller's own.
export function compact(n: number | undefined | null): string {
  if (n == null) return "—"
  return Intl.NumberFormat("en-US", { notation: "compact" }).format(n)
}

// A wall clock in words: seconds under a minute, then minutes, then hours. Here
// beside `compact` and for the same reason — an ETA, an elapsed clock and a
// finished run's duration are the same reading, and three spellings of it across
// the training figures is three things to keep in step.
export function duration(secs: number | undefined | null): string {
  if (secs == null || !Number.isFinite(secs)) return "—"
  const s = Math.max(0, Math.round(secs))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

// A single figure in the stat run: a big tabular value over a muted label. No
// box, no fill of its own — the hairline rules between cells do the separating,
// so the numbers read like a figure's data line, not a grid of cards.
export interface Stat {
  label: string
  value: string
}

// The headline figures, as a divided strip: two columns on a phone, four across
// from `sm` up.
//
// A fixed grid rather than a wrapping row. Wrapping is what made this ragged —
// the items are different widths, so a row that broke left a short last item and
// a dangling rule where the new row started, and the break moved as the numbers
// changed. A grid keeps the columns still, and the rules can then be drawn only
// where two cells actually meet: never down a row's first cell, never above the
// first row.
export function StatRun({ stats }: { stats: Stat[] }) {
  return (
    <dl className="grid grid-cols-2 sm:grid-cols-4">
      {stats.map((s, i) => (
        <div
          key={s.label}
          className={cn(
            "flex min-w-0 flex-col gap-1 px-5 py-4",
            i % 2 === 1 && "border-l border-border",
            i >= 2 && "border-t border-border",
            "sm:border-t-0",
            i % 4 === 0 ? "sm:border-l-0" : "sm:border-l sm:border-border"
          )}
        >
          <dt className="truncate text-xs text-muted-foreground">{s.label}</dt>
          {/* The display face, not the UI one: these are the figures the page
              exists to show, and Fraunces' numerals are where the editorial
              character actually lands. Tabular so columns of them still align. */}
          <dd className="figure truncate text-2xl">
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
  minHeight = "20rem",
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
        "flex h-full min-h-[inherit] items-center justify-center rounded-(--radius-control-sm) border border-dashed text-sm text-muted-foreground",
        className
      )}
    >
      {children}
    </div>
  )
}
