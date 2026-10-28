import { ArchitectureCatalog } from "@/components/authoring/catalog"
import { PageShell } from "@/components/ui/page-shell"

/** Opens the administrator block directory without fetching private data from the server component. */
export default function BlocksPage() {
  return (
    <PageShell width="board" className="max-w-5xl">
      <ArchitectureCatalog kind="block" />
    </PageShell>
  )
}
