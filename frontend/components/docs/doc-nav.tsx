"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"

export interface NavLink {
  title: string
  href: string
  children?: NavLink[]
}

export interface NavSection {
  title: string
  links: NavLink[]
}

/**
 * The docs sidebar. Client-only for `usePathname` — marking the current page is
 * the one thing this cannot do on the server, and a sidebar that does not say
 * where you are is most of the way to no sidebar at all.
 */
export function DocNav({ sections }: { sections: NavSection[] }) {
  const pathname = usePathname()

  return (
    <nav className="flex flex-col gap-7">
      {sections.map((section) => (
        <div key={section.title}>
          <span className="text-xs font-medium tracking-[0.18em] text-muted-foreground uppercase">
            {section.title}
          </span>
          <ul className="mt-2.5 flex flex-col border-l border-border">
            {section.links.map((link) => (
              <li key={link.href}>
                <NavItem link={link} active={pathname === link.href} />
                {link.children && (
                  <ul className="flex flex-col">
                    {link.children.map((child) => (
                      <li key={child.href}>
                        <NavItem
                          link={child}
                          active={pathname === child.href}
                          nested
                        />
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  )
}

/**
 * The active mark is a hairline sitting on top of the list's own rule, not a
 * filled pill — the rest of the app marks state with weight and a rule, and a
 * block of accent here would be the loudest thing on a page of prose.
 */
function NavItem({
  link,
  active,
  nested = false,
}: {
  link: NavLink
  active: boolean
  nested?: boolean
}) {
  return (
    <Link
      href={link.href}
      className={cn(
        "-ml-px block border-l py-1 pl-3 text-sm transition-colors",
        nested && "pl-6 text-[0.8125rem]",
        active
          ? "border-primary font-medium text-foreground"
          : "border-transparent text-muted-foreground hover:border-border hover:text-foreground"
      )}
    >
      {link.title}
    </Link>
  )
}
