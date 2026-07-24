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
    <section className={cn("rounded-xl border border-border/60 bg-background", className)}>
      <header className="border-b border-border/60 px-5 py-4">
        <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
        {description && (
          <p className="mt-0.5 text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground/60">
            {description}
          </p>
        )}
      </header>
      <div className={cn("p-5", bodyClassName)}>{children}</div>
    </section>
  )
}
