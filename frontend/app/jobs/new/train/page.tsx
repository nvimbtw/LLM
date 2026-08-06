import { Suspense } from "react"
import Link from "next/link"

import { TrainComposer } from "@/components/jobs/train-composer"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function NewTrainJobPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        back={{ href: "/jobs", label: "Queue" }}
        title="Train a model"
        description={
          <>
            <p>
              Pretrain from scratch or fine-tune an existing model. The run is
              queued — it starts when the GPU is free, and you can close the
              tab.
            </p>
            <p>
              Self-training and preference tuning are not here: they are stages
              a{" "}
              <Link href="/cycles" className="underline underline-offset-4">
                cycle
              </Link>{" "}
              runs for you, once it has built a catalogue worth training on.
            </p>
          </>
        }
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
