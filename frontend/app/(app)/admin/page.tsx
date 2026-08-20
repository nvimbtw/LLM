import { AdminDirectory } from "@/components/admin/admin-directory"
import { HistoryView } from "@/components/jobs/history-view"
import { QueueView } from "@/components/jobs/queue-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/**
 * The admin panel: the queue, the way to everything else behind the flag, and
 * then the ledger of what has already run.
 *
 * The queue is the hero rather than one card among several, because it is the
 * only thing here that changes on its own — an admin opens this page to see what
 * the GPU is doing, and a summary that then needs a click to become the real
 * ledger would make the common case the slow one.
 *
 * The directory sits *between* the queue and the history, which reads oddly as an
 * outline and correctly on screen: history is unbounded — a page of it is longer
 * than the rest of the panel put together — so anything under it is reached only
 * by scrolling past every run ever made. Queue, then the doors out, then the
 * archive.
 */
export default function AdminPage() {
  return (
    <PageShell>
      <PageHeader
        title="Admin"
        description="One GPU, one line. Training runs, benchmarks and chat turns queue here and execute in order — and the queue survives a restart, so what you leave waiting is still waiting."
      />
      <div className="space-y-8">
        <QueueView />
        <AdminDirectory />
        <HistoryView />
      </div>
    </PageShell>
  )
}
