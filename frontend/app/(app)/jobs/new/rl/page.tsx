import { RlComposer } from "@/components/jobs/rl-composer"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/**
 * Practising against a gym — the composer for the run that *repeats*.
 *
 * Starting a gym run from `/gyms/<id>` evaluates a model once. This queues the
 * loop: roll a batch out, grade it, take one step, checkpoint, again.
 *
 * `?gym=<id>` is how the catalogue hands a gym over: a card there is one click
 * from practising in that gym, so the choice is made before the page loads. It
 * is read here rather than by `useSearchParams` in the composer, which would
 * put the whole page behind a Suspense boundary to say the same thing. An id
 * for a gym that no longer exists falls back to the first, the same as no id
 * at all — the composer validates against the list it fetches.
 */
export default async function NewRlJobPage({
  searchParams,
}: {
  searchParams: Promise<{ gym?: string }>
}) {
  const { gym } = await searchParams
  return (
    <PageShell width="board">
      <PageHeader
        back={
          gym
            ? { href: "/gyms", label: "Gyms" }
            : { href: "/admin", label: "Admin" }
        }
        title="Practise in a gym"
      />
      <RlComposer gym={gym} />
    </PageShell>
  )
}
