import { DataView } from "@/components/data/data-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/** Opens stored datasets, acquisition and cleaning history. */
export default function DataPage() {
  return (
    <PageShell width="board">
      <PageHeader
        title="Data"
        description="Acquire datasets, inspect documents and preview cleaning pipelines."
        back={{ href: "/admin", label: "Admin" }}
      />
      <DataView />
    </PageShell>
  )
}
