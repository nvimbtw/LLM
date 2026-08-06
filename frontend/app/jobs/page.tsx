import { JobsView } from "@/components/jobs/jobs-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function JobsPage() {
  return (
    <PageShell>
      <PageHeader
        title="Jobs"
        description="One GPU, one line. Training runs, benchmarks and chat turns queue here and execute in order — and the queue survives a restart, so what you leave waiting is still waiting."
      />
      <JobsView />
    </PageShell>
  )
}
