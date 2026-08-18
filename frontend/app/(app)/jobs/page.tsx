import { redirect } from "next/navigation"

/**
 * The queue moved to /admin, where it is the hero of the panel.
 *
 * Kept as a redirect rather than deleted: `/jobs` is where every job detail page
 * lives (`/jobs/{id}`) and where every composer submits from, so the bare path is
 * a URL people have typed and bookmarked. It costs one file to keep those working.
 */
export default function JobsPage() {
  redirect("/admin")
}
