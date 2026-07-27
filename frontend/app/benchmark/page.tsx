import { BenchmarkView } from "@/components/benchmark/benchmark-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function BenchmarkPage() {
  return (
    <PageShell>
      <PageHeader
        title="Benchmark"
        description="Held-out perplexity, minimal-pair grammar, and surface metrics — numbers instead of an impression of how close to English a model sounds."
      />
      <BenchmarkView />
    </PageShell>
  )
}
