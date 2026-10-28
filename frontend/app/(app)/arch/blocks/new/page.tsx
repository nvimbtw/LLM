import { NewArchitectureResource } from "@/components/authoring/new-resource"
import { PageShell } from "@/components/ui/page-shell"

/** Opens an administrator block draft or the eligible source copy of one exact installed builtin revision. */
export default async function NewBlockPage({
  searchParams,
}: {
  searchParams: Promise<{ copy?: string; revision?: string }>
}) {
  const query = await searchParams
  return (
    <PageShell width="board" className="max-w-5xl">
      <NewArchitectureResource
        kind="block"
        copy={query.copy}
        revision={query.revision}
      />
    </PageShell>
  )
}
