import { Suspense } from "react"

import { TrainComposer } from "@/components/jobs/train-composer"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function NewTrainJobPage() {
  return (
    <PageShell width="board" className="max-w-4xl">
      <PageHeader
        titleClassName="text-2xl font-normal"
        className="mb-6 border-b-0 pb-0"
        back={{ href: "/admin", label: "Admin" }}
        title="Train a model"
      />

      {/* The form reads `?from=` to seed itself from a previous run, which makes it
          a client boundary that can't be prerendered. Suspense is what lets the
          static half of the page (this header) still be. */}
      <Suspense fallback={null}>
        <TrainComposer />
      </Suspense>
    </PageShell>
  )
}
