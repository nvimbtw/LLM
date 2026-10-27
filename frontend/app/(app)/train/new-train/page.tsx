import { Suspense } from "react"

import { TrainComposer } from "@/components/jobs/train-composer"
import { PageShell } from "@/components/ui/page-shell"
import { AdminBoundary } from "@/components/admin/admin-boundary"

/** Hosts the query-seeded training editor inside the bounded application page. */
export default function NewTrainJobPage() {
  return (
    <PageShell width="board" className="max-w-4xl">
      <Suspense fallback={null}>
        <AdminBoundary>
          <TrainComposer />
        </AdminBoundary>
      </Suspense>
    </PageShell>
  )
}
