import { TeachComposer } from "@/components/jobs/teach-composer"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function NewTeachJobPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        back={{ href: "/jobs", label: "Queue" }}
        title="Teach from a local model"
        description={
          <>
            <p>
              The same job as generating SFT data from a hosted model, with the
              model running here instead. That changes what is affordable more
              than what is possible: a local run costs nothing per call, so it
              can produce thousands of conversations rather than the couple of
              hundred a paid API is worth spending on. Volume is the actual
              remedy for a model that repeats itself.
            </p>
            <p>
              Each call asks one question and then answers it{" "}
              <em>three times</em>, with a longer thinking block each time. That
              is deliberate. Drawing budgets independently across separate
              conversations would give the same spread of reasoning lengths, but
              the corpus would hold no two records differing only in how much
              thinking they show — and that contrast is what makes a thinking
              budget learnable rather than noise. The three replies are required
              to be equally correct, so the tiers teach length, not effort.
            </p>
            <p>
              Answers are filed by <em>measuring</em> the thinking block against
              the target model&rsquo;s tokenizer, not by which tier was
              requested. A teacher asked for forty words routinely writes
              seventy, and a label that describes the request rather than the
              content is not something a student can learn from.
            </p>
            <p>
              This takes the GPU line, unlike hosted generation: the model is
              loaded onto the device, so training, benchmarks and chat wait
              behind it.
            </p>
          </>
        }
      />
      <TeachComposer />
    </PageShell>
  )
}
