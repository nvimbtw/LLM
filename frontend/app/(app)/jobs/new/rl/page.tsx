import { RlComposer } from "@/components/jobs/rl-composer"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/**
 * Practising against a gym — the composer for the run that *repeats*.
 *
 * Starting a gym run from `/gyms/<id>` evaluates a model once. This queues the
 * loop: roll a batch out, grade it, take one step, checkpoint, again.
 */
export default function NewRlJobPage() {
  return (
    <PageShell width="board">
      <PageHeader
        back={{ href: "/admin", label: "Admin" }}
        title="Practise in a gym"
        description="The model does the tasks, is graded on them, and is stepped toward whichever of its own attempts went better — once per iteration, each iteration a gym run of its own."
      />
      <RlComposer />
    </PageShell>
  )
}
