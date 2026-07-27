import { cn } from "@/lib/utils"

// A hairline-framed section with a titled header — the editorial replacement for
// the heavy shadow Card, so panels read as quiet ledgers on the warm paper.
export function Panel({
  title,
  description,
  className,
  bodyClassName,
  children,
}: {
  title: string
  description?: string
  className?: string
  bodyClassName?: string
  children: React.ReactNode
}) {
  return (
    // `bg-card`, not `bg-background`: the card token is now a step above the page
    // in both themes, so a panel is a sheet lying on it rather than an outline
    // drawn on it. That one step is most of what the flatness was.
    <section
      className={cn("rounded-lg border border-border bg-card", className)}
    >
      <header className="border-b border-border px-5 py-4">
        <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
        {description && (
          <p className="mt-0.5 text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase">
            {description}
          </p>
        )}
      </header>
      <div className={cn("p-5", bodyClassName)}>{children}</div>
    </section>
  )
}
