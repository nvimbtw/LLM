import type { ModelUsage } from "@/hooks/use-usage"
import { formatCompact, formatNumber } from "./format"
import { Panel } from "@/components/ui/panel"

export interface UsageByModelProps {
  models: ModelUsage[]
  compact?: boolean
}

/** Renders token shares by model, using a single column in compact settings. */
export function UsageByModel({ models, compact = false }: UsageByModelProps) {
  const total = models.reduce((sum, m) => sum + m.tokens, 0) || 1
  const sorted = [...models].sort((a, b) => b.tokens - a.tokens)

  return (
    <Panel
      variant="open"
      title="By model"
      description="Share of tokens"
    >
      {sorted.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted-foreground">
          No usage yet.
        </p>
      ) : (
        <ul
          className={
            compact
              ? "grid gap-x-8 gap-y-5"
              : "grid gap-x-8 gap-y-5 sm:grid-cols-2 lg:grid-cols-3"
          }
        >
          {sorted.map((model) => {
            const share = model.tokens / total
            return (
              <li key={model.id}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="truncate text-sm font-medium">
                    {model.name}
                  </span>
                  <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
                    {formatCompact(model.tokens)}
                    <span className="ml-2 text-muted-foreground">
                      {Math.round(share * 100)}%
                    </span>
                  </span>
                </div>
                <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                  <div
                    className="h-full rounded-full bg-primary"
                    style={{ width: `${Math.max(share * 100, 2)}%` }}
                  />
                </div>
                <span className="mt-1.5 block eyebrow text-muted-foreground tabular-nums">
                  {formatNumber(model.requests)} requests
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </Panel>
  )
}
