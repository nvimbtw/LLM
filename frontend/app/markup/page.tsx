import Link from "next/link"

import { MarkupShowcase } from "@/components/markup/markup-showcase"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

/**
 * A dev page, like /throbber: no data, no auth, no link in the sidebar. Visit it
 * directly to see how the model's markup renders without having to coax a model
 * into emitting the rare tokens.
 */
export default function MarkupPage() {
  return (
    <PageShell width="reading">
      <PageHeader
        title="Markup"
        description={
          <>
            <p>
              Every renderable special token, in one conversation, through the
              real chat bubble. A fixture rather than a generated thread —
              nesting cases like a quote inside a list item are hard to provoke
              on purpose and gone on reload. The turn delimiters (
              <code className="text-xs">user</code>,{" "}
              <code className="text-xs">assistant</code>,{" "}
              <code className="text-xs">system</code>,{" "}
              <code className="text-xs">endoftext</code>) are frame, not
              content, so they never reach the renderer.
            </p>
            <p>
              <code className="text-xs">quote</code>,{" "}
              <code className="text-xs">warn</code> and{" "}
              <code className="text-xs">reply</code> are rendered here ahead of
              the model: the display exists first so there is something to
              generate SFT against. Nothing emits them yet, and the backend
              vocabulary has not been extended for them.
            </p>
            <p>
              Thinking blocks follow the{" "}
              <span className="font-medium text-foreground">
                Show reasoning
              </span>{" "}
              setting (⌘K), so toggle it to see both states. The vocabulary
              itself lives in{" "}
              <code className="text-xs">
                backend/src/tokenize/src/pretokenize.rs
              </code>
              ; if a token is added there and not here, this page is stale.
            </p>
            <p className="text-xs">
              Sibling dev page:{" "}
              <Link href="/throbber" className="underline underline-offset-4">
                /throbber
              </Link>
            </p>
          </>
        }
      />
      <MarkupShowcase />
    </PageShell>
  )
}
