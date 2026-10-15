import { redirect } from "next/navigation"
import { withRouteQuery, type RouteQuery } from "@/lib/routes"

/** Redirects old evaluation bookmarks to the canonical run URL without dropping query context. */
export default async function LegacyGymRunPage({
  params,
  searchParams,
}: {
  params: Promise<{ gym: string; id: string }>
  searchParams: Promise<RouteQuery>
}) {
  const { id } = await params
  redirect(withRouteQuery(`/gyms/${id}`, await searchParams))
}
