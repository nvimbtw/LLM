import { NewArchitectureResource } from "@/components/authoring/new-resource"
import { PageShell } from "@/components/ui/page-shell"

/** Opens an administrator recipe draft, optionally copying an exact immutable accepted revision. */
export default async function NewArchitecturePage({
  searchParams,
}: {
  searchParams: Promise<{ copy?: string; revision?: string }>
}) {
  const query = await searchParams
  return (
    <PageShell width="board" className="max-w-5xl">
      <NewArchitectureResource
        kind="architecture"
        copy={query.copy}
        revision={query.revision}
      />
    </PageShell>
  )
}
