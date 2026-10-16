import { NewProcessor } from "@/components/data/new-processor"
import { PageShell } from "@/components/ui/page-shell"

/** Opens the dedicated administrator authoring page in the shared application frame. */
export default function NewProcessorPage() {
  return (
    <PageShell width="board" className="max-w-4xl">
      <NewProcessor />
    </PageShell>
  )
}
