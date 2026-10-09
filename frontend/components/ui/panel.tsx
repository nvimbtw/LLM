import { cn } from "@/lib/utils"

export interface PanelProps {
  title: string
  description?: string
  /** A short fact about the section; actions belong with their content. */
  aside?: React.ReactNode
  /** Admin workspaces use open sections; other callers retain the sheet. */
  variant?: "sheet" | "open"
  className?: string
  bodyClassName?: string
  children: React.ReactNode
}

export function Panel({
  title,
  description,
  aside,
  variant = "sheet",
  className,
  bodyClassName,
  children,
}: PanelProps) {
  const open = variant === "open"
  return (
    <section
      className={cn(
        "min-w-0",
        !open &&
          "rounded-(--radius-sheet) border border-border/45 bg-card shadow-sheet",
        className
      )}
    >
      <header
        className={cn(
          open
            ? "mb-4 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1"
            : "flex items-baseline justify-between gap-3 border-b border-border px-5 py-4"
        )}
      >
        <div
          className={cn(
            "min-w-0",
            open && "flex flex-wrap items-baseline gap-x-3 gap-y-1"
          )}
        >
          <h2
            className={
              open
                ? "font-display text-xl"
                : "text-sm font-semibold tracking-tight"
            }
          >
            {title}
          </h2>
          {description && (
            <p
              className={cn(
                "text-muted-foreground",
                open ? "text-xs" : "mt-0.5 eyebrow"
              )}
            >
              {description}
            </p>
          )}
        </div>
        {aside && (
          <div className={open ? "text-xs text-muted-foreground" : "shrink-0"}>
            {aside}
          </div>
        )}
      </header>
      <div className={cn(open ? "min-w-0" : "p-5", bodyClassName)}>
        {children}
      </div>
    </section>
  )
}
