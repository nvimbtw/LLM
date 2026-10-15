"use client"

import { Area, AreaChart, CartesianGrid, XAxis } from "recharts"

import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart"
import type { UsageDay } from "@/hooks/use-usage"
import { Panel } from "@/components/ui/panel"

// One series — total tokens — in the single accent. The input/output split isn't
// dropped, it moves into the tooltip: a plot stays legible with one hue, and the
// theme's muted chart tokens can't reliably separate two categories in dark mode.
const chartConfig = {
  total: { label: "Tokens", color: "var(--primary)" },
} satisfies ChartConfig

type Point = UsageDay & { total: number }

/** Formats an ISO usage date as a short chart label. */
function formatDate(value: string) {
  return new Date(value).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  })
}

/** Displays a labeled token count, emphasizing totals when requested. */
function Row({
  label,
  value,
  strong,
}: {
  label: string
  value: number
  strong?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between gap-6">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={
          strong
            ? "font-medium tabular-nums"
            : "text-muted-foreground tabular-nums"
        }
      >
        {value.toLocaleString("en-US")}
      </span>
    </div>
  )
}

/** Shows the total and input/output counts for the hovered usage day. */
function TokenTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: { payload: Point }[]
  label?: string
}) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="min-w-36 rounded-(--radius-sheet) border border-border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-overlay">
      <div className="mb-1.5 eyebrow text-muted-foreground">
        {formatDate(label ?? "")}
      </div>
      <div className="flex flex-col gap-0.5">
        <Row label="Total" value={d.total} strong />
        <Row label="Input" value={d.input} />
        <Row label="Output" value={d.output} />
      </div>
    </div>
  )
}

export interface UsageChartProps {
  series: UsageDay[]
  compact?: boolean
}

/** Plots daily token totals in an open panel, with tighter spacing in settings. */
export function UsageChart({ series, compact = false }: UsageChartProps) {
  const data: Point[] = series.map((d) => ({ ...d, total: d.input + d.output }))

  return (
    <Panel
      variant="open"
      title="Tokens over time"
      description="Total, last 30 days"
      bodyClassName={compact ? "py-3" : "px-2 py-3"}
    >
      <ChartContainer config={chartConfig} className="aspect-auto h-64 w-full">
        <AreaChart data={data} margin={{ left: 4, right: 4, top: 8 }}>
          <defs>
            <linearGradient id="fillTotal" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.28} />
              <stop
                offset="95%"
                stopColor="var(--primary)"
                stopOpacity={0.02}
              />
            </linearGradient>
          </defs>
          <CartesianGrid
            vertical={false}
            stroke="var(--border)"
            strokeOpacity={0.6}
          />
          <XAxis
            dataKey="date"
            tickLine={false}
            axisLine={false}
            tickMargin={10}
            minTickGap={32}
            tickFormatter={formatDate}
            className="text-xs"
          />
          <ChartTooltip
            cursor={{ stroke: "var(--border)" }}
            content={<TokenTooltip />}
          />
          <Area
            dataKey="total"
            type="monotone"
            stroke="var(--primary)"
            strokeWidth={2}
            fill="url(#fillTotal)"
          />
        </AreaChart>
      </ChartContainer>
    </Panel>
  )
}
