import { ThrobberShowcase } from "@/components/throbber/throbber-showcase"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function ThrobberPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        title="Throbber"
        description="Ledger ticks — three graduations in a fixed footprint. Every state is the same three marks with different numbers, so they morph rather than swap."
      />
      <ThrobberShowcase />
    </PageShell>
  )
}
