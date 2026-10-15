import { redirect } from "next/navigation"

interface NewTrainRedirectPageProps {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}

/** Redirects legacy training bookmarks to the new composer, preserving query values including repeated seeds. */
export default async function NewTrainRedirectPage({
  searchParams,
}: NewTrainRedirectPageProps) {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(await searchParams)) {
    if (value === undefined) continue
    for (const entry of Array.isArray(value) ? value : [value])
      params.append(key, entry)
  }
  const query = params.toString()
  redirect(`/jobs/new-train${query ? `?${query}` : ""}`)
}
