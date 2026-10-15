import { BenchmarkComposer } from "@/components/jobs/benchmark-composer"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/** Opens the form for queueing a benchmark against a model. */
export default function NewBenchmarkJobPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        back={{ href: "/admin", label: "Admin" }}
        title="Benchmark a model"
      />
      <BenchmarkComposer />
    </PageShell>
  )
}
