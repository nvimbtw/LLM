import { ThrobberShowcase } from "@/components/throbber/throbber-showcase"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/** Previews the shared activity indicators and live token art. */
export default function ThrobberPage() {
  return (
    <PageShell width="reading">
      <PageHeader title="Throbber" back={{ href: "/admin", label: "Admin" }} />
      <ThrobberShowcase />
    </PageShell>
  )
}
