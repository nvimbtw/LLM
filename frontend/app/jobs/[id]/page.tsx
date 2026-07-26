import { JobDetail } from "@/components/jobs/job-detail"

export default async function JobPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return (
    <div className="mx-auto w-full max-w-[1400px] flex-1 px-5 py-12 sm:px-8">
      <JobDetail idJob={id} />
    </div>
  )
}
