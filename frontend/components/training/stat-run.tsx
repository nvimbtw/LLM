import * as React from "react"

import { cn } from "@/lib/utils"

/** Formats a measured count compactly, preserving unavailable values as an em dash. */
export function compact(n: number | undefined | null): string {
  if (n == null) return "—"
  return Intl.NumberFormat("en-US", { notation: "compact" }).format(n)
}

/** Formats elapsed or remaining seconds as a compact duration; unavailable readings stay explicit. */
export function duration(secs: number | undefined | null): string {
  if (secs == null || !Number.isFinite(secs)) return "—"
  const s = Math.max(0, Math.round(secs))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

export interface Stat {
  label: string
  value: string
}

/** Aligns headline statistics in a responsive open grid; spacing groups the figures without cell dividers. */
export function StatRun({ stats }: { stats: Stat[] }) {
  return (
    <dl className="grid grid-cols-2 gap-x-8 gap-y-5 px-5 py-4 sm:grid-cols-4">
      {stats.map((s) => (
        <div key={s.label} className="flex min-w-0 flex-col gap-1">
          <dt className="truncate text-xs text-muted-foreground">{s.label}</dt>
          <dd className="truncate figure text-2xl">{s.value}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Presents a chart and caption at a reserved height, with optional right-aligned measurement context. */
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

/** Reserves a chart's footprint with quiet unavailable/loading text, without adding a decorative frame. */
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
        "flex h-full min-h-[inherit] items-center justify-center text-sm text-muted-foreground",
        className
      )}
    >
      {children}
    </div>
  )
}
