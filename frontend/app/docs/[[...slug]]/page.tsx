import { notFound } from "next/navigation"
import { DocMarkdown } from "@/components/docs/doc-markdown"
import { DocToc } from "@/components/docs/doc-toc"
import { allPages, findPage, headings, read, splitTitle } from "@/lib/docs"

interface DocsPageProps {
  params: Promise<{ slug?: string[] }>
}

/**
 * Every doc is generated at build time, which is what keeps `docs/` — a
 * directory outside this module — from being a runtime dependency.
 */
export function generateStaticParams() {
  return allPages().map((page) => ({ slug: page.slug }))
}

export const dynamicParams = false

export async function generateMetadata({ params }: DocsPageProps) {
  const page = findPage((await params).slug ?? [])
  return { title: page ? `${page.title} · Docs` : "Docs" }
}

/** `["backend", "schema"]` → `Backend`. The eyebrow over the title. */
function sectionLabel(slug: string[]): string | undefined {
  if (slug.length < 2) return undefined
  return slug[0].replace(/-/g, " ")
}

export default async function DocsPage({ params }: DocsPageProps) {
  const page = findPage((await params).slug ?? [])
  if (!page) notFound()

  // The title moves to the masthead, so the body starts at the first paragraph
  // rather than repeating the <h1> in a second face and size.
  const { title, body } = splitTitle(read(page))
  const eyebrow = sectionLabel(page.slug)

  return (
    <div className="flex gap-12">
      <article className="min-w-0 flex-1">
        <header className="mb-10 border-b border-border pb-6">
          {eyebrow && (
            <span className="eyebrow text-muted-foreground">
              {eyebrow}
            </span>
          )}
          <h1 className="mt-2 font-display text-3xl font-semibold tracking-tight text-balance">
            {title ?? page.title}
          </h1>
        </header>
        <div className="doc-prose">
          <DocMarkdown page={page} source={body} />
        </div>
      </article>
      <aside className="hidden w-56 shrink-0 xl:block">
        <DocToc headings={headings(body)} />
      </aside>
    </div>
  )
}
