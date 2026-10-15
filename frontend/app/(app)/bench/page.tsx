import { BenchmarkView } from "@/components/benchmark/benchmark-view"
import { ButtonLink } from "@/components/ui/button-link"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/** Opens recorded benchmark comparisons and individual reports. */
export default function BenchmarkPage() {
  return (
    <PageShell>
      <PageHeader
        title="Benchmark"
        back={{ href: "/admin", label: "Admin" }}
        action={
          <ButtonLink href="/bench/new-bench" size="sm">
            Bench a model
          </ButtonLink>
        }
      />
      <BenchmarkView />
    </PageShell>
  )
}
