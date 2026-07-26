import { Suspense } from "react"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { TrainComposer } from "@/components/jobs/train-composer"

export default function NewTrainJobPage() {
  return (
    <div className="mx-auto w-full max-w-3xl flex-1 px-5 py-12 sm:px-8">
      <header className="mb-8 border-b border-border/60 pb-6">
        <Link
          href="/jobs"
          className="mb-4 inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="size-3" />
          Queue
        </Link>
        <h1 className="font-display text-3xl font-semibold tracking-tight">
          Train a model
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Pretrain from scratch or fine-tune an existing model. The run is queued —
          it starts when the GPU is free, and you can close the tab.
        </p>
      </header>

      {/* The form reads `?from=` to seed itself from a previous run, which makes it
          a client boundary that can't be prerendered. Suspense is what lets the
          static half of the page (this header) still be. */}
      <Suspense fallback={null}>
        <TrainComposer />
      </Suspense>
    </div>
  )
}
