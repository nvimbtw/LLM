import type { Usage } from "@/hooks/use-usage"
import { formatCompact, formatCurrency, formatNumber } from "./format"

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="bg-background px-5 py-5">
      <dt className="text-[10px] font-medium uppercase tracking-[0.16em] text-muted-foreground/60">
        {label}
      </dt>
      <dd className="mt-2 font-display text-3xl font-semibold tracking-tight tabular-nums">
        {value}
      </dd>
      {hint && <p className="mt-1 text-[11px] text-muted-foreground/70 tabular-nums">{hint}</p>}
    </div>
  )
}

export function UsageStats({ usage }: { usage: Usage }) {
  const avgPerRequest = usage.totalRequests
    ? Math.round(usage.totalTokens / usage.totalRequests)
    : 0

  // gap-px over a border-toned background paints hairline rules between cells —
  // a divided ledger rather than four boxes.
  return (
    <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-border/60 bg-border/60 sm:grid-cols-4">
      <Stat label="Requests" value={formatNumber(usage.totalRequests)} hint="Last 30 days" />
      <Stat
        label="Tokens"
        value={formatCompact(usage.totalTokens)}
        hint={`${formatNumber(usage.totalTokens)} total`}
      />
      <Stat label="Cost" value={formatCurrency(usage.totalCost)} hint="Last 30 days" />
      <Stat
        label="Avg / request"
        value={formatCompact(avgPerRequest)}
        hint="Tokens per request"
      />
    </dl>
  )
}
