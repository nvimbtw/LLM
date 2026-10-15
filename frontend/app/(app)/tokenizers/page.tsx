import { TokenizersView } from "@/components/tokenizers/tokenizers-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/** Opens the vocabulary catalogue and build dialog in a reading-width workspace. */
export default function TokenizersPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        title="Tokenizers"
        description="Vocabularies turn dataset text into the tokens a model learns and predicts."
        back={{ href: "/admin", label: "Admin" }}
      />
      <TokenizersView />
    </PageShell>
  )
}
