"use client"

import { EmptyState } from "@/components/ui/empty-state"

import Link from "next/link"
import { AdminBoundary } from "@/components/admin/admin-boundary"
import {
  useArchitectureCatalog,
  type ArchitectureKind,
} from "@/hooks/use-architecture-catalog"
import { Button } from "@/components/ui/button"
import { SearchField } from "@/components/ui/search-field"
import {
  ResourceList,
  ResourceRow,
  ResourceTitle,
} from "@/components/ui/resource-list"
import { LoadError } from "@/components/ui/error-state"
import { PageHeader } from "@/components/ui/page-header"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export interface ArchitectureCatalogProps {
  kind: ArchitectureKind
}

/** Gates the resource directory before any source/catalog requests and scopes its browser state to the resource family. */
export function ArchitectureCatalog({ kind }: ArchitectureCatalogProps) {
  return (
    <AdminBoundary>
      <CatalogContent key={kind} kind={kind} />
    </AdminBoundary>
  )
}

/** Shows bounded metadata pages with explicit search/archive filters and immutable accepted identities. */
function CatalogContent({ kind }: ArchitectureCatalogProps) {
  const catalog = useArchitectureCatalog(kind)
  const root = kind === "block" ? "/arch/blocks" : "/arch"
  const title = kind === "block" ? "Blocks" : "Architectures"
  return (
    <>
      <PageHeader
        title={title}
        back={{ href: "/admin", label: "Admin" }}
        action={
          <div className="flex gap-2">
            <Button
              nativeButton={false}
              render={
                <Link
                  href={`${root}/${kind === "block" ? "new-block" : "new-arch"}`}
                />
              }
            >
              New {kind}
            </Button>
          </div>
        }
      />
      <form
        className="mb-6 flex flex-wrap items-end gap-3"
        onSubmit={(event) => {
          event.preventDefault()
          catalog.filter({ search: catalog.search })
        }}
      >
        <label className="grid gap-1 text-sm">
          Search
          <SearchField
            value={catalog.search}
            maxLength={200}
            onChange={(event) => catalog.setSearch(event.target.value)}
          />
        </label>
        <Button type="submit" variant="outline">
          Search
        </Button>
        <label className="grid gap-1 text-sm">
          Origin
          <Select
            value={catalog.query.origin ?? "all"}
            onValueChange={(value) =>
              catalog.filter({
                origin:
                  value === "builtin" || value === "custom" ? value : null,
              })
            }
          >
            <SelectTrigger>
              <SelectValue>
                {(value) =>
                  value === "builtin"
                    ? "Builtin"
                    : value === "custom"
                      ? "Custom"
                      : "All origins"
                }
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All origins</SelectItem>
              <SelectItem value="builtin">Builtin</SelectItem>
              <SelectItem value="custom">Custom</SelectItem>
            </SelectContent>
          </Select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <Switch
            checked={catalog.query.archived ?? false}
            onCheckedChange={(archived) => catalog.filter({ archived })}
          />
          Archived
        </label>
      </form>
      {catalog.loading ? (
        <Skeleton className="h-48 w-full" />
      ) : catalog.error ? (
        <LoadError
          title="Catalog did not load."
          hint={catalog.error}
          onRetry={catalog.refresh}
        />
      ) : (
        <>
          <ResourceList>
            {catalog.page?.items.map((item) => (
              <ResourceRow key={item.resource_id}>
                <Link
                  href={`${root}/${item.resource_id}`}
                  className="block rounded-(--radius-control) py-0 outline-none hover:bg-accent/40 focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <ResourceTitle>{item.name}</ResourceTitle>
                  {item.description && (
                    <p className="mt-1 text-sm text-muted-foreground">
                      {item.description}
                    </p>
                  )}
                  <p className="mt-2 text-xs text-muted-foreground">
                    {item.origin} · Draft {item.draft_version} ·{" "}
                    {item.accepted_revision
                      ? `Accepted ${item.accepted_revision.content_hash.slice(0, 12)}`
                      : "No accepted revision"}
                  </p>
                </Link>
              </ResourceRow>
            ))}
          </ResourceList>
          {!catalog.page?.items.length && (
            <EmptyState title={`No matching ${title.toLowerCase()}`} />
          )}
          <div className="mt-4 flex gap-2">
            {catalog.query.cursor && (
              <Button variant="outline" onClick={() => catalog.filter({})}>
                First page
              </Button>
            )}
            {catalog.page?.next_cursor && (
              <Button variant="outline" onClick={catalog.next}>
                Next page
              </Button>
            )}
          </div>
        </>
      )}
    </>
  )
}
