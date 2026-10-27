import { ArchitectureCatalog } from "@/components/authoring/catalog"
import { PageShell } from "@/components/ui/page-shell"

/** Opens the administrator architecture directory; authenticated reads belong to its client owner. */
export default function ArchitecturesPage() {
  return (
    <PageShell width="board" className="max-w-5xl">
      <ArchitectureCatalog kind="architecture" />
    </PageShell>
  )
}
