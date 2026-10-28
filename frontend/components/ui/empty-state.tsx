import type { ReactNode } from "react"

import { cn } from "@/lib/utils"
import { SectionTexture } from "./section-texture"

export interface EmptyStateProps {
  title: string
  children?: ReactNode
  action?: ReactNode
  className?: string
}

/** Presents an empty collection with bounded warm texture and an optional useful action; explanatory copy is supplied only when the caller needs it. */
export function EmptyState({
  title,
  children,
  action,
  className,
}: EmptyStateProps) {
  return (
    <section
      className={cn(
        "relative isolate flex min-h-44 flex-col items-center justify-center gap-4 py-12 text-center",
        className
      )}
    >
      <SectionTexture tone="warm" className="-z-10" />
      <h2 className="font-display text-xl">{title}</h2>
      {children && (
        <div className="max-w-sm text-sm text-muted-foreground">{children}</div>
      )}
      {action}
    </section>
  )
}
