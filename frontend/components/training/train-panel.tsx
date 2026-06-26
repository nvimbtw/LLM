"use client"

import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import { Progress } from "@/components/ui/progress"
import type { LossPoint } from "@/hooks/use-training"
import type { JobStatus, TrainStats } from "@/lib/training-socket"

const chartConfig = {
  loss: { label: "Loss", color: "var(--chart-1)" },
} satisfies ChartConfig

function formatEta(secs: number | undefined): string {
  if (secs == null || !Number.isFinite(secs)) return "—"
  const s = Math.max(0, Math.round(secs))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-card p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 font-heading text-xl font-semibold tabular-nums">
        {value}
      </div>
    </div>
  )
}

export function TrainPanel({
  train,
  loss,
  checkpoints,
  status,
  onCancel,
}: {
  train: Partial<TrainStats>
  loss: LossPoint[]
  checkpoints: string[]
  status: JobStatus
  onCancel: () => void
}) {
  const totalSteps =
    (train.total_epochs ?? 0) * (train.total_chunks ?? 0) || undefined
  const pct = totalSteps
    ? Math.min(100, Math.round(((train.step ?? 0) / totalSteps) * 100))
    : 0
  const active = status === "training"

  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>Training</CardTitle>
        <CardDescription>
          {active
            ? "Optimizing the transformer — loss streaming live."
            : `Run ${status}.`}
        </CardDescription>
        {active && (
          <CardAction>
            <Button variant="outline" size="sm" onClick={onCancel}>
              Cancel
            </Button>
          </CardAction>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat
            label="Epoch"
            value={`${train.epoch ?? 0} / ${train.total_epochs ?? 0}`}
          />
          <Stat
            label="Loss"
            value={
              train.loss != null && Number.isFinite(train.loss)
                ? train.loss.toFixed(4)
                : "—"
            }
          />
          <Stat
            label="Avg loss"
            value={
              train.avg_loss != null && Number.isFinite(train.avg_loss)
                ? train.avg_loss.toFixed(4)
                : "—"
            }
          />
          <Stat label="ETA" value={formatEta(train.eta_secs)} />
        </div>

        <div className="space-y-1.5">
          <Progress value={pct} />
          <div className="text-right text-xs text-muted-foreground tabular-nums">
            {pct}%
          </div>
        </div>

        {loss.length > 1 ? (
          <ChartContainer config={chartConfig} className="aspect-auto h-64 w-full">
            <LineChart data={loss} margin={{ left: 4, right: 4, top: 4 }}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="step"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={32}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                width={44}
                tickMargin={4}
                domain={["auto", "auto"]}
              />
              <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
              <Line
                dataKey="loss"
                type="monotone"
                stroke="var(--color-loss)"
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ChartContainer>
        ) : (
          <div className="flex h-64 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
            Waiting for the first loss values…
          </div>
        )}

        {checkpoints.length > 0 && (
          <div className="space-y-1">
            <div className="text-sm font-medium">Checkpoints</div>
            <ul className="space-y-0.5 font-mono text-xs text-muted-foreground">
              {checkpoints.map((cp) => (
                <li key={cp} className="truncate">
                  {cp}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
