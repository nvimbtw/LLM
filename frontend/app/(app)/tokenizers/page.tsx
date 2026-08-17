import { TokenizersView } from "@/components/tokenizers/tokenizers-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function TokenizersPage() {
  return (
    <PageShell width="board">
      <PageHeader
        title="Tokenizers"
        description={
          <p>
            A vocabulary is learned from the corpora you pick, then kept. Point
            a training run at one and it skips learning its own — and every run
            sharing a vocabulary shares the packed corpus underneath it.
          </p>
        }
      />
      <TokenizersView />
    </PageShell>
  )
}
