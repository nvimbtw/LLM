import { ThrobberShowcase } from "@/components/throbber/throbber-showcase"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function ThrobberPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        className="mb-6 border-b-0 pb-0"
        title="Throbber"
        back={{ href: "/admin", label: "Admin" }}
      />
      <ThrobberShowcase />
    </PageShell>
  )
}
