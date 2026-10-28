import { ModelsGrid } from "@/components/models/models-grid"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/** Opens the shared model catalogue within Slate's dashboard frame. */
export default function ModelsPage() {
  return (
    <PageShell>
      <PageHeader title="Models" />
      <ModelsGrid />
    </PageShell>
  )
}
