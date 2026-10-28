import { AdminDirectory } from "@/components/admin/admin-directory"
import { HistoryView } from "@/components/jobs/history-view"
import { QueueView } from "@/components/jobs/queue-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/** Places active work and pageable history beside the grouped Workbench, stacking them on narrow screens. */
export default function AdminPage() {
  return (
    <PageShell>
      <PageHeader title="Admin" />
      <div className="grid items-start gap-x-12 gap-y-10 lg:grid-cols-[minmax(0,1fr)_15rem]">
        <div className="min-w-0 space-y-10">
          <QueueView />
          <HistoryView />
        </div>
        <AdminDirectory />
      </div>
    </PageShell>
  )
}
