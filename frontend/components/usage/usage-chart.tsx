"use client"

import { Area, AreaChart, CartesianGrid, XAxis } from "recharts"

import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import type { UsageDay } from "@/hooks/use-usage"

const chartConfig = {
  input: { label: "Input", color: "var(--chart-3)" },
  output: { label: "Output", color: "var(--chart-1)" },
} satisfies ChartConfig

function formatDate(value: string) {
  return new Date(value).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  })
}

export function UsageChart({ series }: { series: UsageDay[] }) {
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>Tokens over time</CardTitle>
        <CardDescription>Input and output tokens, last 30 days</CardDescription>
      </CardHeader>
      <div className="px-(--card-spacing)">
        <ChartContainer config={chartConfig} className="aspect-auto h-64 w-full">
          <AreaChart data={series} margin={{ left: 4, right: 4, top: 4 }}>
            <defs>
              <linearGradient id="fillInput" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-input)" stopOpacity={0.7} />
                <stop offset="95%" stopColor="var(--color-input)" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="fillOutput" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-output)" stopOpacity={0.7} />
                <stop offset="95%" stopColor="var(--color-output)" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              minTickGap={32}
              tickFormatter={formatDate}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={(value) => formatDate(value as string)}
                  indicator="dot"
                />
              }
            />
            <Area
              dataKey="input"
              type="natural"
              stackId="tokens"
              stroke="var(--color-input)"
              fill="url(#fillInput)"
            />
            <Area
              dataKey="output"
              type="natural"
              stackId="tokens"
              stroke="var(--color-output)"
              fill="url(#fillOutput)"
            />
            <ChartLegend content={<ChartLegendContent />} />
          </AreaChart>
        </ChartContainer>
      </div>
    </Card>
  )
}
