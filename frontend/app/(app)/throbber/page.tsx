import { ThrobberShowcase } from "@/components/throbber/throbber-showcase"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function ThrobberPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        title="Throbber"
        description="One pill, in a fixed footprint. Every state is that same bar with different numbers, so they morph rather than swap — and what tells them apart is how colour moves through it."
      />
      <ThrobberShowcase />
    </PageShell>
  )
}
