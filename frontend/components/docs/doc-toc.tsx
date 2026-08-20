import { cn } from "@/lib/utils"
import type { Heading } from "@/lib/docs"

/**
 * The contents rail. Static links, no scroll-spy: a highlight that chases the
 * scroll is a second moving thing on a page whose whole job is to be read.
 */
export function DocToc({ headings }: { headings: Heading[] }) {
  if (headings.length < 2) return null

  return (
    <nav className="sticky top-10 flex flex-col gap-2 border-l border-border pl-5">
      <span className="eyebrow text-muted-foreground">
        On this page
      </span>
      <ul className="flex flex-col gap-1.5 text-sm">
        {headings.map((heading) => (
          <li key={heading.id}>
            <a
              href={`#${heading.id}`}
              className={cn(
                "block text-muted-foreground transition-colors hover:text-foreground",
                heading.depth === 3 && "pl-3 text-xs"
              )}
            >
              {heading.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  )
}
