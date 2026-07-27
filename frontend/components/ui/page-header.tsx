import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { cn } from "@/lib/utils"

/**
 * The masthead every page opens with — title, a sentence on what the page is for,
 * and optionally one action and one way back up.
 *
 * Pulled out of the pages because all fifteen of them had spelled the same header
 * by hand, and they had drifted: two type scales, two fonts, three paddings. One
 * component is what makes the rule (display face, hairline rule under, action on
 * the right) hold as pages get added.
 */
export interface PageHeaderProps {
  title: React.ReactNode
  description?: React.ReactNode
  /** A single primary action, right-aligned against the title. */
  action?: React.ReactNode
  /**
   * Renders the "← Parent" link above the title. Pages that are reachable from
   * more than one parent should pass a resolved value rather than a constant —
   * see `useParentTrail`.
   */
  back?: { href: string; label: string }
  className?: string
}

export function PageHeader({
  title,
  description,
  action,
  back,
  className,
}: PageHeaderProps) {
  return (
    <header className={cn("mb-8 border-b border-border pb-6", className)}>
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
        <div className="min-w-0">
          <h1 className="font-display text-3xl font-semibold tracking-tight">
            {title}
          </h1>
          {/* A div, not a <p>: the longer composers explain themselves in three
              paragraphs, and nesting those inside a <p> is invalid markup that
              React silently reparents. `space-y` styles them when they come. */}
          {description && (
            <div className="mt-2 max-w-2xl space-y-3 text-sm leading-relaxed text-muted-foreground">
              {description}
            </div>
          )}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </div>
    </header>
  )
}
