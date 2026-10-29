import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { cn } from "@/lib/utils"
import { SectionTexture } from "./section-texture"

/** Page identity includes authored resource content, functional metadata and actions; generic introductory prose does not belong here. */
export interface PageHeaderProps {
  title: React.ReactNode
  description?: React.ReactNode
  /** Functional figures displayed beside page actions, with no introductory prose. */
  meta?: React.ReactNode
  /** Page actions, right-aligned against the title. */
  action?: React.ReactNode
  /**
   * Renders the "← Parent" link above the title. Pages that are reachable from
   * more than one parent should pass a resolved value rather than a constant —
   * see `useParentTrail`.
   */
  back?: { href: string; label: string }
  className?: string
  titleClassName?: string
  /** Allows a page to extend its identity texture beyond the content column. */
  textureClassName?: string
  /** Allows editable titles to occupy the space remaining beside page actions. */
  contentClassName?: string
}

/** Renders the page title, optional metadata, back link and actions with shared typography and spacing. */
export function PageHeader({
  title,
  description,
  meta,
  action,
  back,
  className,
  titleClassName,
  textureClassName,
  contentClassName,
}: PageHeaderProps) {
  return (
    <header className={cn("relative isolate mb-8 py-3 pb-6", className)}>
      <SectionTexture className={cn("-z-10", textureClassName)} />
      {back && (
        <Link
          href={back.href}
          className="mb-4 inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="size-3" />
          {back.label}
        </Link>
      )}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className={cn("min-w-0", contentClassName)}>
          <h1
            className={cn(
              "font-display text-3xl font-semibold tracking-tight",
              titleClassName
            )}
          >
            {title}
          </h1>
          {description && (
            <div className="mt-2 max-w-2xl space-y-3 text-sm leading-relaxed text-muted-foreground">
              {description}
            </div>
          )}
        </div>
        {(meta || action) && (
          <div className="flex max-w-full flex-wrap items-center justify-end gap-x-8 gap-y-4">
            {meta}
            {action && <div className="max-w-full shrink-0">{action}</div>}
          </div>
        )}
      </div>
    </header>
  )
}
