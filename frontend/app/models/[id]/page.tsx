import { ModelDetail } from "@/components/models/model-detail"
import { PageShell } from "@/components/ui/page-shell"

export default async function ModelDetailPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return (
    <PageShell>
      <ModelDetail modelId={id} />
    </PageShell>
  )
}
