import { CycleComposer } from "@/components/cycles/cycle-composer"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function NewPostTrainCyclePage() {
  return (
    <PageShell width="reading">
      <PageHeader
        back={{ href: "/cycles/new", label: "Cycle kinds" }}
        title="Post-train"
        description="Unattended post-training. Leave it running; the cycle page is where it reports what it did and why it stopped."
      />
      <CycleComposer />
    </PageShell>
  )
}
