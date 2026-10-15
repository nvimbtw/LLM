import { redirect } from "next/navigation"
import { withRouteQuery, type RouteQuery } from "@/lib/routes"

/** Keeps the former queue bookmark and its query context pointing at Admin. */
export default async function JobsPage({
  searchParams,
}: {
  searchParams: Promise<RouteQuery>
}) {
  redirect(withRouteQuery("/admin", await searchParams))
}
