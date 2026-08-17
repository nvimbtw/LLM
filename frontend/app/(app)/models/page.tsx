import { ModelsGrid } from "@/components/models/models-grid"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function ModelsPage() {
  return (
    <PageShell>
      <PageHeader
        title="Models"
        description="Every model available for your conversations."
      />
      <ModelsGrid />
    </PageShell>
  )
}
