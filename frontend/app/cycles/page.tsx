import Link from "next/link"

import { CyclesView } from "@/components/cycles/cycles-view"

export default function CyclesPage() {
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-5 py-12 sm:px-8">
      <header className="mb-8 border-b border-border/60 pb-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="font-display text-3xl font-semibold tracking-tight">Cycles</h1>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
              Post-training, unattended. A cycle repeats one round — build the
              catalogue, self-train on what cleared the bar, preference-tune away
              what didn&rsquo;t, benchmark — and stops when a round no longer
              improves, keeping the best model rather than the last one.
            </p>
          </div>
          <Link
            href="/cycles/new"
            className="shrink-0 text-sm underline-offset-4 hover:underline"
          >
            Start a cycle →
          </Link>
        </div>
      </header>
      <CyclesView />
    </div>
  )
}
