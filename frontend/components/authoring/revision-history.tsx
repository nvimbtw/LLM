"use client"

import * as React from "react"
import dynamic from "next/dynamic"
import Link from "next/link"
import { useArchitectureHistory } from "@/hooks/use-architecture-history"
import { useArchitectureRevision } from "@/hooks/use-architecture-revision"
import type { ResourceSummary } from "@/lib/architecture/contracts.generated"
import { Button } from "@/components/ui/button"
import { Panel } from "@/components/ui/panel"
import { LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"

const SourceEditor = dynamic(
  () => import("./source-editor").then((module) => module.SourceEditor),
  { ssr: false }
)

export interface RevisionHistoryProps {
  summary: ResourceSummary
  initialRevision?: string
  line?: number
}

/** Presents bounded immutable history and exact selected source without replacing the independently owned editable draft. */
export function RevisionHistory({
  summary,
  initialRevision,
  line,
}: RevisionHistoryProps) {
  const history = useArchitectureHistory(
    summary.kind,
    summary.resource_id,
    summary.accepted_revision?.revision_id ?? null
  )
  const [selected, select] = React.useState<string | null>(
    initialRevision ?? null
  )
  const source = useArchitectureRevision({
    kind: summary.kind,
    id: summary.resource_id,
    origin: summary.origin,
    revision: selected,
  })
  const root = summary.kind === "block" ? "/arch/blocks" : "/arch"
  return (
    <Panel variant="open" title="Accepted revisions">
      <div className="grid gap-4">
        {history.loading && <Skeleton className="h-24 w-full" />}
        {history.error && (
          <LoadError
            title="History did not load."
            hint={history.error}
            onRetry={history.retry}
          />
        )}
        {history.value && (
          <>
            <ol className="divide-y divide-border">
              {history.value.items.map((item) => (
                <li
                  key={item.reference.revision_id}
                  className="grid gap-2 py-3 text-sm"
                >
                  <div className="flex flex-wrap items-center gap-3">
                    <Button
                      variant="ghost"
                      onClick={() => select(item.reference.revision_id)}
                    >
                      Inspect revision {item.revision_number}
                    </Button>
                    <Link
                      className="underline"
                      href={`${root}/${summary.kind === "block" ? "new-block" : "new-arch"}?copy=${summary.resource_id}&revision=${item.reference.revision_id}`}
                    >
                      Copy this revision
                    </Link>
                  </div>
                  <code className="break-all">
                    {item.reference.content_hash}
                  </code>
                </li>
              ))}
            </ol>
            {!history.value.items.length && (
              <p className="text-sm text-muted-foreground">
                No accepted revisions. Saving a draft does not validate or
                accept it.
              </p>
            )}
            <div className="flex gap-2">
              <Button
                variant="outline"
                disabled={!history.cursor}
                onClick={history.first}
              >
                First page
              </Button>
              <Button
                variant="outline"
                disabled={!history.value.next_cursor}
                onClick={history.next}
              >
                Next page
              </Button>
            </div>
          </>
        )}
        {selected && (
          <section
            aria-label="Immutable accepted source"
            className="grid gap-3 border-t border-border pt-4"
          >
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm break-all">Accepted revision {selected}</p>
              <Button variant="ghost" onClick={() => select(null)}>
                Close source
              </Button>
            </div>
            {source.loading && <Skeleton className="h-64 w-full" />}
            {source.error && (
              <LoadError
                title="Accepted source did not load."
                hint={source.error}
                onRetry={source.retry}
              />
            )}
            {source.value && (
              <SourceEditor
                language={summary.kind === "block" ? "python" : "yaml"}
                ariaLabel="Immutable accepted source"
                readOnly
                line={selected === initialRevision ? line : undefined}
                value={
                  "source" in source.value
                    ? source.value.source
                    : source.value.yaml
                }
                onChange={() => {}}
              />
            )}
          </section>
        )}
      </div>
    </Panel>
  )
}
