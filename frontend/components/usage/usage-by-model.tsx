import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card"
import type { ModelUsage } from "@/hooks/use-usage"
import { formatCompact, formatNumber } from "./format"

// Cycle through theme chart tokens so bars stay on-palette in both themes.
const COLORS = ["var(--chart-1)", "var(--chart-3)", "var(--chart-2)", "var(--chart-4)", "var(--chart-5)"]

export function UsageByModel({ models }: { models: ModelUsage[] }) {
  const total = models.reduce((sum, m) => sum + m.tokens, 0) || 1
  const sorted = [...models].sort((a, b) => b.tokens - a.tokens)

  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>By model</CardTitle>
        <CardDescription>Share of tokens used</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {sorted.map((model, i) => {
          const share = model.tokens / total
          return (
            <div key={model.id} className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between gap-2">
                <span className="flex items-center gap-2 text-sm font-medium">
                  <span
                    className="size-2.5 rounded-[3px]"
                    style={{ backgroundColor: COLORS[i % COLORS.length] }}
                  />
                  {model.name}
                </span>
                <span className="text-sm text-muted-foreground tabular-nums">
                  {formatCompact(model.tokens)}
                  <span className="ml-2 text-xs">{Math.round(share * 100)}%</span>
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${share * 100}%`,
                    backgroundColor: COLORS[i % COLORS.length],
                  }}
                />
              </div>
              <span className="text-xs text-muted-foreground tabular-nums">
                {formatNumber(model.requests)} requests
              </span>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}
