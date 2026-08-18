import { AdminDirectory } from "@/components/admin/admin-directory"
import { JobsView } from "@/components/jobs/jobs-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/**
 * The admin panel: the queue, and the way to everything else behind the flag.
 *
 * The queue is the hero rather than one card among several, because it is the
 * only thing here that changes on its own — an admin opens this page to see what
 * the GPU is doing, and a summary that then needs a click to become the real
 * ledger would make the common case the slow one. The directory below it is the
 * part that is merely navigation, and it sits below because it never moves.
 */
export default function AdminPage() {
  return (
    <PageShell>
      <PageHeader
        title="Admin"
        description="One GPU, one line. Training runs, benchmarks and chat turns queue here and execute in order — and the queue survives a restart, so what you leave waiting is still waiting."
      />
      <div className="space-y-8">
        <JobsView />
        <AdminDirectory />
      </div>
    </PageShell>
  )
}
