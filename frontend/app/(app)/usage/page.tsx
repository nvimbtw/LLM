import { UsageOverview } from "@/components/usage/usage-overview"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function UsagePage() {
  return (
    <PageShell>
      {/* No rule under the header — the ruled ledger opens with its own. */}
      <PageHeader
        className="mb-6 border-b-0 pb-0"
        title="Usage"
        description="Requests, tokens, and cost over the last 30 days."
      />
      <UsageOverview />
    </PageShell>
  )
}
