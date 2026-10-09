"use client"

import * as React from "react"
import { useParams } from "next/navigation"

import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"
import { Panel } from "@/components/ui/panel"
import { LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"
import { RunForm } from "@/components/gyms/run-form"
import { RunList } from "@/components/gyms/run-list"
import { listGymRuns, listGyms, type Gym, type GymRun } from "@/lib/gyms"

/** One gym: how to start a run of it, and every run there has been. */
export default function GymPage() {
  const { gym: gymId } = useParams<{ gym: string }>()
  const [gym, setGym] = React.useState<Gym | null>(null)
  const [runs, setRuns] = React.useState<GymRun[] | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    let active = true
    Promise.all([listGyms(), listGymRuns()])
      .then(([gyms, all]) => {
        if (!active) return
        setGym(gyms.find((g) => g.id === gymId) ?? null)
        setRuns(all.filter((run) => run.gym_id === gymId))
      })
      .catch(
        (e) =>
          active && setError(e instanceof Error ? e.message : "Failed to load.")
      )
    return () => {
      active = false
    }
  }, [gymId])

  if (error) {
    return (
      <PageShell>
        <LoadError hint={error} />
      </PageShell>
    )
  }

  return (
    <PageShell>
      <PageHeader
        className="mb-6 border-b-0 pb-0"
        title={gym?.name ?? gymId}
        back={{ href: "/gyms", label: "Gyms" }}
      />
      <div className="space-y-6">
        {gym ? <RunForm gym={gym} /> : <Skeleton className="h-64" />}
        <Panel variant="open" title="Runs">
          {runs === null ? (
            <Skeleton className="h-24" />
          ) : (
            <RunList runs={runs} />
          )}
        </Panel>
      </div>
    </PageShell>
  )
}
