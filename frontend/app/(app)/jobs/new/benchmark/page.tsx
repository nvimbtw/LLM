import { BenchmarkComposer } from "@/components/jobs/benchmark-composer"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function NewBenchmarkJobPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        className="mb-6 border-b-0 pb-0"
        back={{ href: "/admin", label: "Admin" }}
        title="Benchmark a model"
      />
      <BenchmarkComposer />
    </PageShell>
  )
}
