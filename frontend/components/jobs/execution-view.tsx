"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { useSessionToken } from "@/hooks/use-auth"
import { useExecutionTarget } from "@/hooks/use-execution-target"
import { jobHref, type JobKind } from "@/lib/jobs"
import { withRouteQuery, type RouteQuery } from "@/lib/routes"
import { LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"
import { PageShell } from "@/components/ui/page-shell"
import { GymRunView } from "@/components/gyms/gym-run-view"
import { BenchmarkReportView } from "@/components/benchmark/benchmark-report-view"
import { JobDetail } from "./job-detail"

export interface ExecutionViewProps {
  id: string
  kind: JobKind
  legacyQuery?: RouteQuery
}

/** Scopes canonical execution routing and legacy redirects to the actual browser session. */
export function ExecutionView(props: ExecutionViewProps) {
  const token = useSessionToken()
  return token ? (
    <ExecutionContent
      key={`${token}:${props.id}:${props.kind}`}
      token={token}
      {...props}
    />
  ) : null
}

/** Selects the existing job, gym or report renderer without opening an unrelated workflow's socket. */
function ExecutionContent({
  token,
  id,
  kind,
  legacyQuery,
}: ExecutionViewProps & { token: string }) {
  const router = useRouter()
  const { target, error } = useExecutionTarget(token, id, kind)
  const destination =
    target?.type === "job" &&
    (legacyQuery !== undefined || target.job.kind !== kind)
      ? withRouteQuery(jobHref(target.job), legacyQuery ?? {})
      : null
  React.useEffect(() => {
    if (destination) router.replace(destination)
  }, [destination, router])
  if (error)
    return (
      <PageShell>
        <LoadError title={error} />
      </PageShell>
    )
  if (!target || destination)
    return (
      <PageShell>
        <Skeleton className="h-64 w-full" />
      </PageShell>
    )
  if (target.type === "gym") return <GymRunView runId={target.id} />
  if (target.type === "report")
    return (
      <PageShell>
        <BenchmarkReportView run={target.run} />
      </PageShell>
    )
  return (
    <PageShell>
      <JobDetail idJob={id} />
    </PageShell>
  )
}
