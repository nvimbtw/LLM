import { TokenizersView } from "@/components/tokenizers/tokenizers-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function TokenizersPage() {
  return (
    <PageShell width="board">
      <PageHeader
        titleClassName="text-2xl font-normal"
        className="mb-6 border-b-0 pb-0"
        title="Tokenizers"
        back={{ href: "/admin", label: "Admin" }}
      />
      <TokenizersView />
    </PageShell>
  )
}
