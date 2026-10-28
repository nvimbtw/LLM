import { notFound } from "next/navigation"
import {
  revisionLocation,
  type RevisionLocationQuery,
} from "@/lib/architecture/revision-location"
import { ArchitectureWorkspace } from "@/components/authoring/workspace"
import { PageShell } from "@/components/ui/page-shell"

/** Resolves one block resource identity without reading browser credentials or private source in the server component. */
export default async function BlockPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>
  searchParams: Promise<RevisionLocationQuery>
}) {
  const { id } = await params
  const location = revisionLocation(await searchParams)
  if (!location) notFound()
  return (
    <PageShell width="board" className="max-w-5xl">
      <ArchitectureWorkspace
        kind="block"
        id={id.toLowerCase()}
        initialRevision={location.revision}
        line={location.line}
      />
    </PageShell>
  )
}
