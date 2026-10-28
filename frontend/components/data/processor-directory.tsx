"use client"

import { EmptyState } from "@/components/ui/empty-state"

import Link from "next/link"
import { Plus } from "lucide-react"

import { Button } from "@/components/ui/button"
import { LoadError } from "@/components/ui/error-state"
import { PageHeader } from "@/components/ui/page-header"
import {
  ResourceList,
  ResourceRow,
  ResourceTitle,
} from "@/components/ui/resource-list"
import { Skeleton } from "@/components/ui/skeleton"
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import { useProcessors } from "@/hooks/use-data"

/** Scopes the processor directory to the signed-in session. */
export function ProcessorDirectory() {
  const token = useSessionToken()
  return <Directory key={token ?? "anonymous"} />
}

/** Lists builtin and custom processors with links to their dedicated source pages. */
function Directory() {
  const { isAdmin } = useAuth()
  const { data, isLoading, error } = useProcessors(null)
  return (
    <>
      <PageHeader
        title="Processors"
        back={{ href: "/admin", label: "Admin" }}
        meta={`${data.length} processors`}
        action={
          isAdmin ? (
            <Button
              nativeButton={false}
              render={<Link href="/processors/new-processor" />}
            >
              <Plus />
              New processor
            </Button>
          ) : undefined
        }
      />
      {isLoading ? (
        <Skeleton className="h-48 w-full" />
      ) : error ? (
        <LoadError title="Processors did not load." hint={error} />
      ) : (
        <>
          {data.length === 0 && <EmptyState title="No processors available" />}
          <ResourceList>
            {data.map((processor) => (
              <ResourceRow key={processor.name}>
                <Link
                  href={`/processors/${encodeURIComponent(processor.name)}`}
                  className="flex flex-wrap items-start justify-between gap-4 rounded-(--radius-control) py-0 outline-none hover:bg-accent/40 focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <div className="min-w-0 flex-1">
                    <ResourceTitle>{processor.title}</ResourceTitle>
                    {processor.description && (
                      <p className="mt-1 text-sm text-muted-foreground">
                        {processor.description}
                      </p>
                    )}
                    <p className="mt-2 text-xs text-muted-foreground">
                      {processor.name} · {processor.applies_to.join(" · ")}
                    </p>
                  </div>
                  <code
                    className="text-xs text-muted-foreground"
                    title={processor.version_hash}
                  >
                    {processor.version_hash.slice(0, 12)}
                  </code>
                </Link>
              </ResourceRow>
            ))}
          </ResourceList>
        </>
      )}
    </>
  )
}
