import { ValidationDetail } from "@/components/authoring/validation-detail"
import { PageShell } from "@/components/ui/page-shell"

/** Opens one durable validation execution without treating the resource draft or observer as its lifecycle owner. */
export default async function ValidationPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return (
    <PageShell width="board" className="max-w-5xl">
      <ValidationDetail id={id.toLowerCase()} />
    </PageShell>
  )
}
