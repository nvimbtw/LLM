import { notFound } from "next/navigation"
import {
  revisionLocation,
  type RevisionLocationQuery,
} from "@/lib/architecture/revision-location"
import { ArchitectureWorkspace } from "@/components/authoring/workspace"
import { PageShell } from "@/components/ui/page-shell"

/** Resolves the route identity while the authenticated client owns private architecture reads and edits. */
export default async function ArchitecturePage({
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
        kind="architecture"
        id={id.toLowerCase()}
        initialRevision={location.revision}
        line={location.line}
      />
    </PageShell>
  )
}
