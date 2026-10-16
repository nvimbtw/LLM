import { ProcessorDetail } from "@/components/data/processor-detail"
import { PageShell } from "@/components/ui/page-shell"

export interface ProcessorPageProps {
  params: Promise<{ processor_id: string }>
}

/** Opens a processor's stable name as its source workspace identity. */
export default async function ProcessorPage({ params }: ProcessorPageProps) {
  const { processor_id } = await params
  return (
    <PageShell width="board">
      <ProcessorDetail name={processor_id} />
    </PageShell>
  )
}
