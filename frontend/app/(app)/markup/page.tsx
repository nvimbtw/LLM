import { MarkupShowcase } from "@/components/markup/markup-showcase"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/**
 * A dev page, like /throbber: no data, no auth, no link in the sidebar. Visit it
 * directly to see how the model's markup renders without having to coax a model
 * into emitting the rare tokens.
 */
export default function MarkupPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        title="Markup"
        description="Preview how model tokens render in conversations and inspect their source."
        back={{ href: "/admin", label: "Admin" }}
      />
      <MarkupShowcase />
    </PageShell>
  )
}
