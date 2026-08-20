import { ModelsGrid } from "@/components/models/models-grid"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function ModelsPage() {
  return (
    <PageShell>
      {/* No rule under the header: the ruled ledger below opens with one of its
          own, and two parallel rules 32px apart read as an accident rather than
          as a section break. */}
      <PageHeader
        className="mb-6 border-b-0 pb-0"
        title="Models"
        description="Every model available for your conversations."
      />
      <ModelsGrid />
    </PageShell>
  )
}
