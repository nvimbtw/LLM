import { ProcessorDirectory } from "@/components/data/processor-directory"
import { PageShell } from "@/components/ui/page-shell"

/** Opens the authenticated directory of builtin and operator-authored processors. */
export default function ProcessorsPage() {
  return (
    <PageShell width="board">
      <ProcessorDirectory />
    </PageShell>
  )
}
