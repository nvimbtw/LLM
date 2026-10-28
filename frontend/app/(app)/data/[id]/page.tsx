import { DatasetDetail } from "@/components/data/dataset-detail"
import { PageShell } from "@/components/ui/page-shell"

/** Opens the requested dataset’s documents and cleaning workspace. */
export default async function DatasetPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return (
    <PageShell width="board">
      <DatasetDetail datasetId={id} />
    </PageShell>
  )
}
