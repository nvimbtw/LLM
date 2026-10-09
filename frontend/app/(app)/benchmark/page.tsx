import { BenchmarkView } from "@/components/benchmark/benchmark-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function BenchmarkPage() {
  return (
    <PageShell>
      <PageHeader
        className="mb-6 border-b-0 pb-0"
        title="Benchmark"
        back={{ href: "/admin", label: "Admin" }}
      />
      <BenchmarkView />
    </PageShell>
  )
}
