"use client"

import * as React from "react"

import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"
import { LoadError } from "@/components/ui/error-state"
import { EmptyState } from "@/components/ui/empty-state"
import { ResourceList } from "@/components/ui/resource-list"
import { Skeleton } from "@/components/ui/skeleton"
import { GymCard } from "@/components/gyms/gym-card"
import { listGyms, type Gym } from "@/lib/gyms"

/** Loads the service-owned gym catalogue and its practice and evaluation links. */
export default function GymsPage() {
  const [gyms, setGyms] = React.useState<Gym[] | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    let active = true
    listGyms()
      .then((all) => active && setGyms(all))
      .catch(
        (e) =>
          active && setError(e instanceof Error ? e.message : "Failed to load.")
      )
    return () => {
      active = false
    }
  }, [])

  const orderedGyms = gyms && [
    ...gyms.filter((gym) => gym.id === "general"),
    ...gyms.filter((gym) => gym.id !== "general"),
  ]

  return (
    <PageShell>
      <PageHeader title="Gyms" back={{ href: "/admin", label: "Admin" }} />
      {error ? (
        <LoadError hint={error} />
      ) : orderedGyms === null ? (
        <div className="grid gap-4">
          <Skeleton className="h-36" />
          <Skeleton className="h-36" />
        </div>
      ) : orderedGyms.length === 0 ? (
        <EmptyState title="No gyms available" />
      ) : (
        <ResourceList>
          {orderedGyms.map((gym) => (
            <GymCard
              key={gym.id}
              id={gym.id}
              name={gym.name}
              description={gym.description}
            />
          ))}
        </ResourceList>
      )}
    </PageShell>
  )
}
