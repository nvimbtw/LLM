import { ExecutionView } from "@/components/jobs/execution-view"
import type { RouteQuery } from "@/lib/routes"

/** Resolves a legacy job UUID to its domain route, preserving queries and parent context. */
export default async function LegacyJobPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>
  searchParams: Promise<RouteQuery>
}) {
  const { id } = await params
  return <ExecutionView id={id} kind="train" legacyQuery={await searchParams} />
}
