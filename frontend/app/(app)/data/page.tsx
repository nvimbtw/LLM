import { DataView } from "@/components/data/data-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function DataPage() {
  return (
    <PageShell width="board">
      <PageHeader
        className="mb-6 border-b-0 pb-0"
        title="Data"
        back={{ href: "/admin", label: "Admin" }}
      />
      <DataView />
    </PageShell>
  )
}
