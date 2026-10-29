import { AdminOverview } from "@/components/admin/admin-overview"
import { PageShell } from "@/components/ui/page-shell"

/** Render the administrator dashboard within the shared board-width shell. */
export default function AdminPage() {
  return (
    <PageShell className="@container max-w-none">
      <AdminOverview />
    </PageShell>
  )
}
