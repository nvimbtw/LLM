import { AutolabelComposer } from "@/components/jobs/autolabel-composer"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function NewAutolabelJobPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        back={{ href: "/jobs", label: "Queue" }}
        title="Auto-label replies"
        description={
          <>
            <p>
              Generates several replies to each past prompt and ranks them with
              rules — n-gram fluency against the reference corpus, real-word
              rate, repetition, and whether the turn was closed properly. The
              best and worst of each set become preference training data, which
              is how a KTO run gets a dataset without waiting for hundreds of
              hand-clicked thumbs.
            </p>
            <p>
              These labels only ever reach the training set. The
              benchmark&rsquo;s preference score stays computed from human
              thumbs alone, so it remains an independent check on whether a tune
              learned to write better replies or just to satisfy the rules.
            </p>
          </>
        }
      />
      <AutolabelComposer />
    </PageShell>
  )
}
