import { ThrobberShowcase } from "@/components/throbber/throbber-showcase"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function ThrobberPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        title="Throbber"
        description="A tree grown on a hexagonal dot field, regrown from scratch on every pass, so the mark is never twice the same. What stays constant is the character of the growth — one long route, a shallow bush, the full net — and that is what tells the eight states apart."
      />
      <ThrobberShowcase />
    </PageShell>
  )
}
