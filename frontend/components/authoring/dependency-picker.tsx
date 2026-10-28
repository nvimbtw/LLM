"use client"

import * as React from "react"
import Link from "next/link"
import {
  useArchitectureCatalog,
  type ArchitectureKind,
} from "@/hooks/use-architecture-catalog"
import { useArchitectureHistory } from "@/hooks/use-architecture-history"
import { useArchitectureRevision } from "@/hooks/use-architecture-revision"
import type { ResourceSummary } from "@/lib/architecture/contracts.generated"
import { dependencySnippet } from "@/lib/architecture/dependency-snippet"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { SearchField } from "@/components/ui/search-field"
import { ResourceList, ResourceRow } from "@/components/ui/resource-list"
import { Panel } from "@/components/ui/panel"
import { FormError, LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"

/** Offers bounded accepted block/subarchitecture choices and exact-revision snippets; it never changes editor bytes or adopts newer revisions automatically. */
export function DependencyPicker() {
  const [kind, setKind] = React.useState<ArchitectureKind>("block")
  return (
    <Panel variant="open" title="Pinned dependencies">
      <div className="grid gap-4">
        <p className="max-w-prose text-sm text-muted-foreground">
          Choose an accepted revision and copy its module entry into your YAML.
          Add bindings and flow explicitly. Refreshing the catalog never
          replaces an existing pin.
        </p>
        <div className="flex gap-2">
          <Button
            variant="text"
            aria-pressed={kind === "block"}
            className={
              kind === "block" ? "text-primary" : "text-muted-foreground"
            }
            onClick={() => setKind("block")}
          >
            Blocks
          </Button>
          <Button
            variant="text"
            aria-pressed={kind === "architecture"}
            className={
              kind === "architecture" ? "text-primary" : "text-muted-foreground"
            }
            onClick={() => setKind("architecture")}
          >
            Subarchitectures
          </Button>
        </div>
        <DependencyCatalog key={kind} kind={kind} />
      </div>
    </Panel>
  )
}

export interface DependencyCatalogProps {
  kind: ArchitectureKind
}

/** Lists only this bounded unarchived catalog page; selecting a resource is deliberate and retains its immutable history choice. */
function DependencyCatalog({ kind }: DependencyCatalogProps) {
  const catalog = useArchitectureCatalog(kind)
  const [selected, select] = React.useState<ResourceSummary | null>(null)
  return (
    <div className="grid gap-4">
      <form
        className="flex items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          catalog.filter({ search: catalog.search })
        }}
      >
        <label className="grid gap-2 text-sm">
          Search dependencies
          <SearchField
            value={catalog.search}
            maxLength={200}
            onChange={(event) => catalog.setSearch(event.target.value)}
          />
        </label>
        <Button type="submit" variant="outline">
          Search
        </Button>
        <Button type="button" variant="ghost" onClick={catalog.refresh}>
          Refresh
        </Button>
      </form>
      {catalog.loading ? (
        <Skeleton className="h-24 w-full" />
      ) : catalog.error ? (
        <LoadError
          title="Dependencies did not load."
          hint={catalog.error}
          onRetry={catalog.refresh}
        />
      ) : (
        <>
          <ResourceList density="compact" className="max-h-72 overflow-auto">
            {catalog.page?.items
              .filter((item) => !item.archived && item.accepted_revision)
              .map((item) => (
                <ResourceRow density="compact" key={item.resource_id}>
                  <Button
                    variant="ghost"
                    className="h-auto w-full justify-start py-0 text-left whitespace-normal"
                    onClick={() => select(item)}
                  >
                    {item.name} ·{" "}
                    {item.accepted_revision!.content_hash.slice(0, 12)}
                  </Button>
                </ResourceRow>
              ))}
          </ResourceList>
          <div className="flex gap-2">
            <Button
              variant="outline"
              disabled={!catalog.query.cursor}
              onClick={() => catalog.filter({})}
            >
              First page
            </Button>
            <Button
              variant="outline"
              disabled={!catalog.page?.next_cursor}
              onClick={catalog.next}
            >
              Next page
            </Button>
          </div>
        </>
      )}
      {selected && (
        <DependencyRevision key={selected.resource_id} summary={selected} />
      )}
    </div>
  )
}

export interface DependencyRevisionProps {
  summary: ResourceSummary
}

/** Presents an explicitly selected immutable dependency revision and a copyable source snippet; no latest-revision fallback can replace it. */
function DependencyRevision({ summary }: DependencyRevisionProps) {
  const history = useArchitectureHistory(
    summary.kind,
    summary.resource_id,
    summary.accepted_revision?.revision_id ?? null
  )
  const [revision, setRevision] = React.useState(
    summary.accepted_revision!.revision_id
  )
  const selected = useArchitectureRevision({
    kind: summary.kind,
    id: summary.resource_id,
    origin: summary.origin,
    revision,
  })
  const [instance, setInstance] = React.useState("my_module")
  const [copyError, setCopyError] = React.useState<string | null>(null)
  const [copied, setCopied] = React.useState<string | null>(null)
  const owner = React.useRef({ active: true })
  React.useEffect(() => {
    const current = { active: true }
    owner.current = current
    return () => {
      current.active = false
    }
  }, [])
  let snippet: string | null = null
  let error: string | null = null
  try {
    if (selected.value)
      snippet = dependencySnippet(
        instance,
        summary,
        selected.value.accepted.reference
      )
  } catch (cause) {
    error =
      cause instanceof Error
        ? cause.message
        : "Module entry could not be formatted."
  }

  /** Copies only the visible deliberate pin; clipboard failures leave the text available for manual selection. */
  async function copy() {
    if (!snippet) return
    const current = owner.current
    try {
      await navigator.clipboard.writeText(snippet)
      if (current.active) {
        setCopied(snippet)
        setCopyError(null)
      }
    } catch {
      if (current.active)
        setCopyError(
          "Clipboard is unavailable. Select and copy the snippet below."
        )
    }
  }

  return (
    <section aria-label="Selected dependency" className="grid gap-3 pt-4">
      <Link
        className="text-sm underline"
        href={`${summary.kind === "block" ? "/arch/blocks" : "/arch"}/${summary.resource_id}`}
      >
        {summary.name}
      </Link>
      {history.loading && <Skeleton className="h-16 w-full" />}
      {history.error && (
        <LoadError
          title="Dependency history did not load."
          hint={history.error}
          onRetry={history.retry}
        />
      )}
      <ul className="flex flex-wrap gap-2">
        {history.value?.items.map((item) => (
          <li key={item.reference.revision_id}>
            <Button
              size="sm"
              variant={
                revision === item.reference.revision_id ? "default" : "outline"
              }
              onClick={() => setRevision(item.reference.revision_id)}
            >
              Use revision {item.revision_number}
            </Button>
          </li>
        ))}
      </ul>
      {history.value && (
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="ghost"
            disabled={!history.cursor}
            onClick={history.first}
          >
            First page
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={!history.value.next_cursor}
            onClick={history.next}
          >
            Next page
          </Button>
        </div>
      )}
      {selected.loading && (
        <p className="text-sm" role="status">
          Reading selected revision…
        </p>
      )}
      {selected.error && (
        <LoadError
          title="Selected dependency did not load."
          hint={selected.error}
          onRetry={selected.retry}
        />
      )}
      <label className="grid gap-2 text-sm">
        Module instance name
        <Input
          value={instance}
          maxLength={64}
          onChange={(event) => setInstance(event.target.value)}
        />
      </label>
      {(error || copyError) && <FormError>{error ?? copyError}</FormError>}
      {snippet && (
        <>
          <pre className="overflow-auto rounded border border-border p-3 text-xs">
            {snippet}
          </pre>
          <Button
            className="w-fit"
            variant="outline"
            onClick={() => void copy()}
          >
            {copied === snippet ? "Copied" : "Copy pinned module entry"}
          </Button>
        </>
      )}
      {selected.value && (
        <details>
          <summary className="text-sm">Accepted declarations</summary>
          <pre className="max-h-80 overflow-auto p-3 text-xs">
            {JSON.stringify(
              "declaration" in selected.value
                ? selected.value.declaration
                : selected.value.recipe,
              null,
              2
            )}
          </pre>
        </details>
      )}
    </section>
  )
}
