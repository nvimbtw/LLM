import { JobDetail } from "@/components/jobs/job-detail"
import { PageShell } from "@/components/ui/page-shell"

export default async function JobPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return (
    <PageShell>
      <JobDetail idJob={id} />
    </PageShell>
  )
}
