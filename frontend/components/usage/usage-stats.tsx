import type { Usage } from "@/hooks/use-usage"
import { formatCompact, formatCurrency, formatNumber } from "./format"

/** Presents a measured usage total and its optional unit or reporting period. */
function Stat({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint?: string
}) {
  return (
    <div className="pr-6">
      <dt className="eyebrow text-muted-foreground">{label}</dt>
      <dd className="mt-2 figure text-3xl">{value}</dd>
      {hint && (
        <p className="mt-1 text-xs text-muted-foreground tabular-nums">
          {hint}
        </p>
      )}
    </div>
  )
}

export interface UsageStatsProps {
  usage: Usage
  compact?: boolean
}

/** Summarizes the supplied usage measurements in an open grid sized for its containing panel. */
export function UsageStats({ usage, compact = false }: UsageStatsProps) {
  const avgPerRequest = usage.totalRequests
    ? Math.round(usage.totalTokens / usage.totalRequests)
    : 0

  return (
    <dl
      className={
        compact
          ? "grid grid-cols-1 gap-x-4 gap-y-4 py-4 sm:grid-cols-2"
          : "grid grid-cols-2 gap-x-8 gap-y-5 py-4 sm:grid-cols-4"
      }
    >
      <Stat
        label="Requests"
        value={formatNumber(usage.totalRequests)}
        hint="Last 30 days"
      />
      <Stat
        label="Tokens"
        value={formatCompact(usage.totalTokens)}
        hint={`${formatNumber(usage.totalTokens)} total`}
      />
      <Stat
        label="Cost"
        value={formatCurrency(usage.totalCost)}
        hint="Last 30 days"
      />
      <Stat
        label="Avg / request"
        value={formatCompact(avgPerRequest)}
        hint="Tokens per request"
      />
    </dl>
  )
}
