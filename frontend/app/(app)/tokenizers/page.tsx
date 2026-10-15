import { TokenizersView } from "@/components/tokenizers/tokenizers-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/** Opens the board-width vocabulary catalogue with links to detail and creation pages. */
export default function TokenizersPage() {
  return (
    <PageShell width="board">
      <PageHeader
        title="Tokenizers"
        back={{ href: "/admin", label: "Admin" }}
      />
      <TokenizersView />
    </PageShell>
  )
}
