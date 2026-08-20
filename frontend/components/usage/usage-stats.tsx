import type { Usage } from "@/hooks/use-usage"
import { formatCompact, formatCurrency, formatNumber } from "./format"

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
      <dt className="eyebrow text-muted-foreground">
        {label}
      </dt>
      <dd className="figure mt-2 text-3xl">
        {value}
      </dd>
      {hint && (
        <p className="mt-1 text-xs text-muted-foreground tabular-nums">
          {hint}
        </p>
      )}
    </div>
  )
}

export function UsageStats({ usage }: { usage: Usage }) {
  const avgPerRequest = usage.totalRequests
    ? Math.round(usage.totalTokens / usage.totalRequests)
    : 0

  // The ruled ledger: summary figures are furniture, not content, so they take
  // rules and no box (see the elevation rule in globals.css). A rule above and
  // below, the figures spaced across it, and the page showing through — the
  // band no longer competes with the cards it sits over.
  return (
    <dl className="grid grid-cols-2 gap-x-8 gap-y-5 border-y border-border py-4 sm:grid-cols-4">
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
