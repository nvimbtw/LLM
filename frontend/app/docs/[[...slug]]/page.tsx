import { notFound } from "next/navigation"
import { DocMarkdown } from "@/components/docs/doc-markdown"
import { DocToc } from "@/components/docs/doc-toc"
import { SectionTexture } from "@/components/ui/section-texture"
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

/** Names the statically generated documentation page in browser metadata. */
export async function generateMetadata({ params }: DocsPageProps) {
  const page = findPage((await params).slug ?? [])
  return { title: page ? `${page.title} · Docs` : "Docs" }
}

/** `["backend", "schema"]` → `Backend`. The eyebrow over the title. */
function sectionLabel(slug: string[]): string | undefined {
  if (slug.length < 2) return undefined
  return slug[0].replace(/-/g, " ")
}

/** Renders one documentation article with shared page-identity texture and its independent table of contents. */
export default async function DocsPage({ params }: DocsPageProps) {
  const page = findPage((await params).slug ?? [])
  if (!page) notFound()

  const { title, body } = splitTitle(read(page))
  const eyebrow = sectionLabel(page.slug)

  return (
    <div className="flex gap-12">
      <article className="min-w-0 flex-1">
        <header className="relative isolate mb-10 pb-6">
          <SectionTexture className="-z-10" />
          {eyebrow && (
            <span className="eyebrow text-muted-foreground">{eyebrow}</span>
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
