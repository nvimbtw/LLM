import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { BenchmarkComposer } from "@/components/jobs/benchmark-composer"

export default function NewBenchmarkJobPage() {
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
          Benchmark a model
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Held-out perplexity, minimal-pair grammar and surface metrics. The run
          takes a minute or two on the GPU and waits its turn in the queue.
        </p>
      </header>
      <BenchmarkComposer />
    </div>
  )
}
