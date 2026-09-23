"use client"

import * as React from "react"

import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"
import { LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"
import { GymCard } from "@/components/gyms/gym-card"
import { listGyms, type Gym } from "@/lib/gyms"

export default function GymsPage() {
  const [gyms, setGyms] = React.useState<Gym[] | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  // The catalogue is the service's, not a constant here: a task type added in
  // `gyms-engine/gyms/registry.py` shows up without a frontend change.
  React.useEffect(() => {
    let active = true
    listGyms()
      .then((all) => active && setGyms(all))
      .catch((e) => active && setError(e instanceof Error ? e.message : "Failed to load."))
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
      <PageHeader
        title="Gyms"
        back={{ href: "/admin", label: "Admin" }}
        description="Environments a model is put to work in. A run generates tasks, lets the model use that gym's tools, and scores the result."
      />
      {error ? (
        <LoadError hint={error} />
      ) : orderedGyms === null ? (
        <div className="grid gap-4 md:grid-cols-2">
          <Skeleton className="h-36" />
          <Skeleton className="h-36" />
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {orderedGyms.map((gym) => (
            <GymCard
              key={gym.id}
              id={gym.id}
              name={gym.name}
              description={gym.description}
              featured={gym.id === "general"}
            />
          ))}
        </div>
      )}
    </PageShell>
  )
}
