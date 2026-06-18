"use client"

import { useUsage } from "@/hooks/use-usage"
import { Skeleton } from "@/components/ui/skeleton"
import { UsageStats } from "./usage-stats"
import { UsageChart } from "./usage-chart"
import { UsageByModel } from "./usage-by-model"

export function UsageOverview() {
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
      <div className="flex flex-col gap-4">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-xl" />
          ))}
        </div>
        <div className="grid gap-4 lg:grid-cols-3">
          <Skeleton className="h-80 rounded-xl lg:col-span-2" />
          <Skeleton className="h-80 rounded-xl" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <UsageStats usage={usage} />
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <UsageChart series={usage.series} />
        </div>
        <UsageByModel models={usage.byModel} />
      </div>
    </div>
  )
}
