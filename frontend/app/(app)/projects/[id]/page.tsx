import { ProjectView } from "@/components/projects/project-view"
import { PageShell } from "@/components/ui/page-shell"

/** Resolves the routed project identity and renders its client-authenticated workspace. */
export default async function ProjectPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return (
    <PageShell width="reading">
      <ProjectView key={id} projectId={id} />
    </PageShell>
  )
}
