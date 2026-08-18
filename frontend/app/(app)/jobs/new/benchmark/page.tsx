import { BenchmarkComposer } from "@/components/jobs/benchmark-composer"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function NewBenchmarkJobPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        back={{ href: "/admin", label: "Admin" }}
        title="Benchmark a model"
        description="Held-out perplexity, minimal-pair grammar and surface metrics. The run takes a minute or two on the GPU and waits its turn in the queue."
      />
      <BenchmarkComposer />
    </PageShell>
  )
}
