import Link from "next/link"

import { CyclesView } from "@/components/cycles/cycles-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function CyclesPage() {
  return (
    <PageShell>
      <PageHeader
        title="Cycles"
        description="Loops that run themselves. A cycle drives the job queue on your behalf — repeating a round, judging whether it improved, and stopping when it stops paying — so a night of post-training is one decision instead of thirty."
        action={
          <Link
            href="/cycles/new"
            className="text-sm underline-offset-4 hover:underline"
          >
            Start a cycle →
          </Link>
        }
      />
      <CyclesView />
    </PageShell>
  )
}
