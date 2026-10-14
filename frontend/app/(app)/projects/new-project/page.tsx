import { NewProjectView } from "@/components/projects/new-project-view"
import { PageShell } from "@/components/ui/page-shell"

/** Renders the dedicated project-creation page within the signed-in reading frame. */
export default function NewProjectPage() {
  return (
    <PageShell width="reading">
      <NewProjectView />
    </PageShell>
  )
}
