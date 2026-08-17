import { isValidElement, type ReactNode } from "react"
import Link from "next/link"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { headingId, resolveLink, type DocPage } from "@/lib/docs"

interface DocMarkdownProps {
  page: DocPage
  source: string
}

const EXTERNAL = /^(https?:|mailto:|#)/

/** The heading's own text, for its anchor id — children may be nested nodes. */
function textOf(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node)
  if (Array.isArray(node)) return node.map(textOf).join("")
  if (isValidElement<{ children?: ReactNode }>(node))
    return textOf(node.props.children)
  return ""
}

function Heading({ level, children }: { level: 2 | 3; children: ReactNode }) {
  const Tag = `h${level}` as const
  const id = headingId(textOf(children))
  return (
    <Tag id={id}>
      {children}
      <a href={`#${id}`} className="doc-anchor" aria-label="Link to this section">
        #
      </a>
    </Tag>
  )
}

/**
 * Renders one doc.
 *
 * Relative links are rewritten to their route so the same markdown reads
 * correctly here and in an editor; links that resolve to something outside the
 * docs tree (a source file) become plain text, and links that resolve to
 * nothing throw during the build (see `resolveLink`).
 */
export function DocMarkdown({ page, source }: DocMarkdownProps) {
  return (
    <Markdown
      remarkPlugins={[remarkGfm]}
      components={{
        h2: ({ children }) => <Heading level={2}>{children}</Heading>,
        h3: ({ children }) => <Heading level={3}>{children}</Heading>,
        // Wrapped so a wide table scrolls inside its own box instead of
        // widening the column and taking the prose measure with it.
        table: ({ children }) => (
          <div className="doc-table">
            <table>{children}</table>
          </div>
        ),
        a({ href, children }) {
          if (!href || EXTERNAL.test(href)) {
            return (
              <a href={href} rel="noreferrer">
                {children}
              </a>
            )
          }
          const route = resolveLink(page, href)
          if (!route) {
            // Resolves to a real file with no page of its own — a source path.
            // Printed as the path it is rather than a link to nowhere.
            return <span className="text-muted-foreground">{children}</span>
          }
          return <Link href={route}>{children}</Link>
        },
      }}
    >
      {source}
    </Markdown>
  )
}

