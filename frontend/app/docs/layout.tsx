import Link from "next/link"
import { ArrowUpRight } from "lucide-react"
import {
  DocNav,
  type NavLink,
  type NavSection,
} from "@/components/docs/doc-nav"
import { navigation, type NavEntry } from "@/lib/docs"

const routeFor = (slug: string[]) => ["/docs", ...slug].join("/")

/**
 * Public frame for the docs. Deliberately outside `(app)` — the docs are
 * readable without an account and must not depend on the signed-in chrome, so
 * this builds its own masthead and rail rather than borrowing the sidebar.
 */
export default function DocsLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const toLink = (entry: NavEntry): NavLink => ({
    title: entry.label,
    href: routeFor(entry.slug),
    children: entry.children?.map(toLink),
  })

  const sections: NavSection[] = navigation()
    .filter((section) => section.entries.length > 0)
    .map((section) => ({
      title: section.title,
      links: section.entries.map(toLink),
    }))

  return (
    <div className="min-h-svh">
      <header className="sticky top-0 z-20 border-b border-border bg-background/85 backdrop-blur-sm">
        <div className="mx-auto flex h-14 max-w-[88rem] items-center gap-4 px-5 sm:px-8">
          <Link href="/docs" className="flex items-baseline gap-2.5">
            <span className="font-display text-lg font-semibold tracking-tight">
              Slate
            </span>
            <span className="text-xs font-medium tracking-[0.22em] text-muted-foreground uppercase">
              Docs
            </span>
          </Link>
          <Link
            href="/"
            className="ml-auto inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground"
          >
            Open the app
            <ArrowUpRight className="size-3.5" />
          </Link>
        </div>
      </header>

      <div className="mx-auto flex max-w-[88rem] gap-12 px-5 py-12 sm:px-8">
        {/* Sticky under the 3.5rem header, and its own scroll container so a
            long tree never traps the page's scroll. */}
        <aside className="sticky top-[3.5rem] hidden h-[calc(100svh-3.5rem)] w-52 shrink-0 overflow-y-auto py-2 md:block">
          <DocNav sections={sections} />
        </aside>
        <div className="min-w-0 flex-1">
          {/* Below `md` the rail is gone, so the tree folds into a disclosure
              above the page. A <details> rather than a drawer: no state, no
              JS, and it closes itself on navigation because the page remounts. */}
          <details className="mb-10 border-b border-border pb-6 md:hidden">
            <summary className="cursor-pointer text-xs font-medium tracking-[0.18em] text-muted-foreground uppercase marker:text-border">
              Contents
            </summary>
            <div className="mt-5">
              <DocNav sections={sections} />
            </div>
          </details>
          {children}
        </div>
      </div>
    </div>
  )
}
