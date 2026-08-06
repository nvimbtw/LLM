import { CycleDetail } from "@/components/cycles/cycle-detail"
import { PageShell } from "@/components/ui/page-shell"

export default async function CyclePage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return (
    <PageShell>
      <CycleDetail idCycle={id} />
    </PageShell>
  )
}
