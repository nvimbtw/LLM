import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { AutolabelComposer } from "@/components/jobs/autolabel-composer"

export default function NewAutolabelJobPage() {
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
          Auto-label replies
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Generates several replies to each past prompt and ranks them with rules —
          n-gram fluency against the reference corpus, real-word rate, repetition,
          and whether the turn was closed properly. The best and worst of each set
          become preference training data, which is how a KTO run gets a dataset
          without waiting for hundreds of hand-clicked thumbs.
        </p>
        <p className="mt-3 text-sm text-muted-foreground">
          These labels only ever reach the training set. The benchmark&rsquo;s
          preference score stays computed from human thumbs alone, so it remains an
          independent check on whether a tune learned to write better replies or
          just to satisfy the rules.
        </p>
      </header>
      <AutolabelComposer />
    </div>
  )
}
