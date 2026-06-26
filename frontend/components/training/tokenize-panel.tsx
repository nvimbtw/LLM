"use client"

import * as React from "react"
import { ChevronDown, ChevronRight } from "lucide-react"
import { CartesianGrid, Line, LineChart, XAxis } from "recharts"

import {
  Card,
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
import type { VocabPoint } from "@/hooks/use-training"
import type { TokenizeStats } from "@/lib/training-socket"

const chartConfig = {
  vocab: { label: "Vocab size", color: "var(--chart-2)" },
} satisfies ChartConfig

function compact(n: number | undefined): string {
  if (n == null) return "—"
  return Intl.NumberFormat("en-US", { notation: "compact" }).format(n)
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

export function TokenizePanel({
  tokenize,
  vocabSeries,
  log,
  active,
}: {
  tokenize: Partial<TokenizeStats>
  vocabSeries: VocabPoint[]
  log: string[]
  active: boolean
}) {
  const [showLog, setShowLog] = React.useState(false)
  const logEndRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (showLog) logEndRef.current?.scrollIntoView({ block: "nearest" })
  }, [log, showLog])

  const pct = Math.round((tokenize.progress ?? 0) * 100)

  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>Tokenization</CardTitle>
        <CardDescription>
          {active
            ? "Building the BPE tokenizer and corpus from your uploads…"
            : "Tokenizer and corpus built."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Files" value={compact(tokenize.files)} />
          <Stat label="Characters" value={compact(tokenize.chars)} />
          <Stat label="Vocab size" value={compact(tokenize.vocab)} />
          <Stat label="Merges" value={compact(tokenize.merges)} />
        </div>

        <div className="space-y-1.5">
          <Progress value={pct} />
          <div className="text-right text-xs text-muted-foreground tabular-nums">
            {pct}%
          </div>
        </div>

        {vocabSeries.length > 1 && (
          <ChartContainer config={chartConfig} className="aspect-auto h-40 w-full">
            <LineChart data={vocabSeries} margin={{ left: 4, right: 4, top: 4 }}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="merges"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={32}
              />
              <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
              <Line
                dataKey="vocab"
                type="monotone"
                stroke="var(--color-vocab)"
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ChartContainer>
        )}

        <div>
          <button
            type="button"
            onClick={() => setShowLog((v) => !v)}
            className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
          >
            {showLog ? (
              <ChevronDown className="size-4" />
            ) : (
              <ChevronRight className="size-4" />
            )}
            {showLog ? "Hide" : "Show"} log ({log.length} lines)
          </button>
          {showLog && (
            <div className="mt-2 max-h-48 overflow-y-auto rounded-md border bg-muted/40 p-3 font-mono text-xs leading-relaxed">
              {log.length === 0 ? (
                <span className="text-muted-foreground">No output yet.</span>
              ) : (
                log.map((line, i) => <div key={i}>{line}</div>)
              )}
              <div ref={logEndRef} />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
