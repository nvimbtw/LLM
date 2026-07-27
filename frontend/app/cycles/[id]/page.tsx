import { CycleDetail } from "@/components/cycles/cycle-detail"

export default async function CyclePage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-5 py-12 sm:px-8">
      <CycleDetail idCycle={id} />
    </div>
  )
}
