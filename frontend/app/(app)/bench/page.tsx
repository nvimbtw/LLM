import { BenchmarkView } from "@/components/benchmark/benchmark-view"
import { PageShell } from "@/components/ui/page-shell"
import { BENCHMARK_SUITES } from "@/lib/benchmark"

/** Opens the requested recorded suite, falling back to supported Language for unknown query values. */
export default async function BenchmarkPage({
  searchParams,
}: {
  searchParams: Promise<{ suite?: string | string[] }>
}) {
  const params = await searchParams
  const suite =
    BENCHMARK_SUITES.find((suite) => suite.id === params.suite)?.id ??
    "language"
  return (
    <PageShell>
      <BenchmarkView key={suite} initialSuite={suite} />
    </PageShell>
  )
}
