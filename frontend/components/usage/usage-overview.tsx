"use client"

import { useUsage } from "@/hooks/use-usage"
import { Skeleton } from "@/components/ui/skeleton"
import { UsageStats } from "./usage-stats"
import { UsageChart } from "./usage-chart"
import { UsageByModel } from "./usage-by-model"

export interface UsageOverviewProps {
  compact?: boolean
}

export function UsageOverview({ compact = false }: UsageOverviewProps) {
  const { usage, isLoading, error } = useUsage()

  if (error) {
    return (
      <p className="text-sm text-destructive">
        Failed to load usage. Please try again.
      </p>
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

  // Chart first, models underneath — the per-model breakdown is secondary to the
  // trend, and full-width lets a long list flow into columns instead of a tall
  // side rail that quadruples the page height.
  return (
    <div className="flex flex-col gap-6">
      <UsageStats compact={compact} usage={usage} />
      <UsageChart compact={compact} series={usage.series} />
      <UsageByModel compact={compact} models={usage.byModel} />
    </div>
  )
}
