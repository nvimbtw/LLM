import { cn } from "@/lib/utils"

// A titled section that holds content you can act on — so it is a *sheet* under
// the elevation rule in globals.css: the card surface, an edge dropped to a
// whisper, and `shadow-sheet`. Furniture (the stat ledgers, filter rows) takes
// rules instead and never a box; see the ledger idiom in docs/frontend.
//
// The shadow is not a return of the heavy Material card this replaced: in light
// it is the ink at 5%/22% falling ten pixels, in dark it is a 6% highlight along
// the top edge, because a shadow on near-black has nothing to fall on.
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
      className={cn(
        "rounded-(--radius-sheet) border border-border/45 bg-card shadow-sheet",
        className
      )}
    >
      <header className="border-b border-border px-5 py-4">
        <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
        {description && (
          <p className="eyebrow mt-0.5 text-muted-foreground">
            {description}
          </p>
        )}
      </header>
      <div className={cn("p-5", bodyClassName)}>{children}</div>
    </section>
  )
}
