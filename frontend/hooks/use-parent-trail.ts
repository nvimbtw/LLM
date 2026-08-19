"use client"

import { useSearchParams } from "next/navigation"

/**
 * Where "back" goes, for pages reachable from more than one place.
 *
 * A job detail page hardcoded its way back to the queue, so drilling
 * cycle → job → back dropped you in /jobs having never been there. The parent is
 * therefore carried in the URL by whoever linked, which keeps it right across a
 * reload and a shared link — unlike `router.back()`, which would also send you
 * backwards through a redirect or a form submit.
 *
 * `parent`, not `from`: composers already use `?from=<job id>` to seed themselves
 * from a previous run, and the two must not collide.
 */
export interface ParentTrail {
  href: string
  label: string
}

/** Adds the parent to a link's href. Use at the linking site, not the target. */
export function withParent(href: string, parent: string): string {
  const sep = href.includes("?") ? "&" : "?"
  return `${href}${sep}parent=${encodeURIComponent(parent)}`
}

// What each parent route is called in a back link. Matched against the path only,
// so a parent carrying its own query still resolves.
const LABELS: ReadonlyArray<readonly [RegExp, string]> = [
  [/^\/cycles\/[^/]+$/, "Cycle"],
  [/^\/cycles$/, "Cycles"],
  [/^\/admin$/, "Admin"],
  [/^\/benchmark$/, "Benchmark"],
  [/^\/models\/[^/]+$/, "Model"],
  [/^\/models$/, "Models"],
  [/^\/conversations$/, "Conversations"],
  [/^\/data\/[^/]+$/, "Dataset"],
  [/^\/data$/, "Data"],
]

export function useParentTrail(fallback: ParentTrail): ParentTrail {
  const params = useSearchParams()
  const parent = params.get("parent")

  // Relative-only, and no protocol-relative `//host`: this value lands in an
  // href, so an attacker-supplied query must not be able to point it offsite.
  if (!parent || !parent.startsWith("/") || parent.startsWith("//"))
    return fallback

  const label = LABELS.find(([pattern]) =>
    pattern.test(parent.split("?")[0])
  )?.[1]
  return label ? { href: parent, label } : fallback
}
