import { BenchmarkView } from "@/components/benchmark/benchmark-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/** Opens recorded benchmark comparisons and individual reports. */
export default function BenchmarkPage() {
  return (
    <PageShell>
      <PageHeader title="Benchmark" back={{ href: "/admin", label: "Admin" }} />
      <BenchmarkView />
    </PageShell>
  )
}
