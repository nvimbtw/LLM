import { UsageOverview } from "@/components/usage/usage-overview"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function UsagePage() {
  return (
    <PageShell>
      <PageHeader
        title="Usage"
        description="Requests, tokens, and cost over the last 30 days."
      />
      <UsageOverview />
    </PageShell>
  )
}
