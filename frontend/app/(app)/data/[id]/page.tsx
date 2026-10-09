import { DatasetDetail } from "@/components/data/dataset-detail"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default async function DatasetPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return (
    <PageShell width="board">
      <PageHeader
        className="mb-6 border-b-0 pb-0"
        title="Dataset"
        back={{ href: "/data", label: "Data" }}
      />
      <DatasetDetail datasetId={id} />
    </PageShell>
  )
}
