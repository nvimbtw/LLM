"use client"

import * as React from "react"

import { ProcessorSource } from "@/components/data/processor-source"
import { LoadError } from "@/components/ui/error-state"
import { PageHeader } from "@/components/ui/page-header"
import { Skeleton } from "@/components/ui/skeleton"
import { useSessionToken } from "@/hooks/use-auth"
import { useProcessors } from "@/hooks/use-data"

export interface ProcessorDetailProps {
  name: string
}

/** Resets source ownership when either the processor or browser session changes. */
export function ProcessorDetail({ name }: ProcessorDetailProps) {
  const token = useSessionToken()
  return <Detail key={`${token}:${name}`} name={name} />
}

/** Presents one processor's metadata and full-file source workspace. */
function Detail({ name }: ProcessorDetailProps) {
  const [revision, setRevision] = React.useState(0)
  const { data, isLoading, error } = useProcessors(null, revision)
  const processor = data.find((entry) => entry.name === name)
  return (
    <>
      <PageHeader
        title={processor?.title ?? name}
        back={{ href: "/processors", label: "Processors" }}
        meta={processor?.applies_to.join(" · ")}
      />
      {isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : error ? (
        <LoadError title="Processor did not load." hint={error} />
      ) : !processor ? (
        <LoadError title="No such processor." />
      ) : (
        <>
          {processor.description && (
            <p className="mb-5 text-sm text-muted-foreground">
              {processor.description}
            </p>
          )}
          <ProcessorSource
            name={name}
            height="65vh"
            onSaved={() => setRevision((value) => value + 1)}
          />
        </>
      )}
    </>
  )
}
