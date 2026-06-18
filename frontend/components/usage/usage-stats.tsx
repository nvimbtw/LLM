import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { Usage } from "@/hooks/use-usage"
import { formatCompact, formatCurrency, formatNumber } from "./format"

function StatCard({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint?: string
}) {
  return (
    <Card className="gap-2">
      <CardHeader>
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-2xl tabular-nums">{value}</CardTitle>
        {hint && (
          <p className="text-xs text-muted-foreground tabular-nums">{hint}</p>
        )}
      </CardHeader>
    </Card>
  )
}

export function UsageStats({ usage }: { usage: Usage }) {
  const avgPerRequest = Math.round(usage.totalTokens / usage.totalRequests)

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard
        label="Requests"
        value={formatNumber(usage.totalRequests)}
        hint="Last 30 days"
      />
      <StatCard
        label="Tokens"
        value={formatCompact(usage.totalTokens)}
        hint={`${formatNumber(usage.totalTokens)} total`}
      />
      <StatCard
        label="Cost"
        value={formatCurrency(usage.totalCost)}
        hint="Last 30 days"
      />
      <StatCard
        label="Avg / request"
        value={formatCompact(avgPerRequest)}
        hint="Tokens per request"
      />
    </div>
  )
}
