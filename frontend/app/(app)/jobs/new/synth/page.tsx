import { SynthComposer } from "@/components/jobs/synth-composer"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function NewSynthJobPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        back={{ href: "/jobs", label: "Queue" }}
        title="Generate SFT conversations"
        description={
          <>
            <p>
              Asks a cheap hosted model for conversations in the voice of the
              hand-written ones — real gold conversations are embedded in every
              call as examples, because telling a small model to write like a
              senior engineer does not work and showing it does. What generation
              adds over the curated set is long reasoning blocks, which the
              hand-written conversations almost never have.
            </p>
            <p>
              Every conversation is validated before it is written: the markup
              vocabulary comes from the tokenizer itself, and the length is
              measured by rendering it through the real chat template and
              counting the tokens the training run will chunk. Nothing that
              would not fit gets written.
            </p>
            <p>
              This runs on the network lane rather than the GPU queue — it waits
              on somebody else&rsquo;s API, so training, benchmarks and chat
              carry on beside it.
            </p>
          </>
        }
      />
      <SynthComposer />
    </PageShell>
  )
}
