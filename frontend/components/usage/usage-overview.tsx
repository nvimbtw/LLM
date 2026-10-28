"use client"

import { useUsage } from "@/hooks/use-usage"
import { LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"
import { UsageStats } from "./usage-stats"
import { UsageChart } from "./usage-chart"
import { UsageByModel } from "./usage-by-model"

export interface UsageOverviewProps {
  compact?: boolean
}

/** Presents account usage or a notification-backed retry boundary when its read fails. */
export function UsageOverview({ compact = false }: UsageOverviewProps) {
  const { usage, isLoading, error, retry } = useUsage()

  if (error) {
    return (
      <LoadError
        title="Failed to load usage."
        hint={error.message}
        onRetry={retry}
      />
    )
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <Skeleton className="h-32 rounded-(--radius-sheet)" />
        <Skeleton className="h-80 rounded-(--radius-sheet)" />
        <Skeleton className="h-40 rounded-(--radius-sheet)" />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <UsageStats compact={compact} usage={usage} />
      <UsageChart compact={compact} series={usage.series} />
      <UsageByModel compact={compact} models={usage.byModel} />
    </div>
  )
}
