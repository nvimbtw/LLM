import { ProjectsView } from "@/components/projects/projects-view"
import { PageShell } from "@/components/ui/page-shell"

/** Renders the private-project directory within the existing signed-in page frame. */
export default function ProjectsPage() {
  return (
    <PageShell>
      <ProjectsView />
    </PageShell>
  )
}
