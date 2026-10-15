"use client"

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { Panel } from "@/components/ui/panel"

export interface RewardChartProps {
  data: { step: number; reward: number; total: number }[]
}

/** Plots dense reward and cumulative return for the supplied episode steps; empty series render nothing. */
export function RewardChart({ data }: RewardChartProps) {
  if (data.length === 0) return null
  return (
    <Panel
      variant="open"
      title="Rewards"
      description="Per-step dense reward and cumulative return"
    >
      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis
              dataKey="step"
              tick={{
                fontSize: 11,
                style: { fontVariantNumeric: "tabular-nums" },
              }}
            />
            <YAxis
              tick={{
                fontSize: 11,
                style: { fontVariantNumeric: "tabular-nums" },
              }}
            />
            <Tooltip />
            <Area
              type="monotone"
              dataKey="reward"
              stroke="var(--chart-1)"
              fill="var(--chart-1)"
              fillOpacity={0.2}
            />
            <Area
              type="monotone"
              dataKey="total"
              stroke="var(--chart-2)"
              fill="transparent"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}
