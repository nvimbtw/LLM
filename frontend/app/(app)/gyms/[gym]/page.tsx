import { redirect } from "next/navigation"
import { ExecutionView } from "@/components/jobs/execution-view"
import { withRouteQuery, type RouteQuery } from "@/lib/routes"

/** Opens a gym execution UUID; old source slugs redirect to their score-once form and history. */
export default async function GymExecutionPage({
  params,
  searchParams,
}: {
  params: Promise<{ gym: string }>
  searchParams: Promise<RouteQuery>
}) {
  const { gym } = await params
  if (
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(gym)
  )
    redirect(
      withRouteQuery(
        `/gyms/sources/${encodeURIComponent(gym)}`,
        await searchParams
      )
    )
  return <ExecutionView id={gym} kind="rl" />
}
