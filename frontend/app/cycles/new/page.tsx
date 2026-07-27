import Link from "next/link"
import { ArrowRight } from "lucide-react"

import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"
import { CYCLE_KINDS } from "@/lib/cycles"

/**
 * Pick a kind of cycle, then configure it.
 *
 * There is exactly one card here today, which is the point: the step exists so
 * that the second kind is a new entry in `CYCLE_KINDS` and a new composer route,
 * rather than a redesign of a page that had assumed post-training was the only
 * thing a cycle could be.
 */
export default function NewCyclePage() {
  return (
    <PageShell width="reading">
      <PageHeader
        back={{ href: "/cycles", label: "Cycles" }}
        title="Start a cycle"
        description="A cycle runs the queue for you and stops on its own. Pick what it should loop on."
      />

      <ul className="space-y-3">
        {CYCLE_KINDS.map((kind) => (
          <li key={kind.kind}>
            <Link
              href={kind.href}
              className="group flex items-start gap-4 rounded-lg border border-border px-5 py-4 transition-colors hover:border-border hover:bg-muted/40"
            >
              <div className="min-w-0 flex-1">
                <h2 className="font-display text-lg font-semibold tracking-tight">
                  {kind.label}
                </h2>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  {kind.description}
                </p>
              </div>
              <ArrowRight className="mt-1 size-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
            </Link>
          </li>
        ))}
      </ul>
    </PageShell>
  )
}
