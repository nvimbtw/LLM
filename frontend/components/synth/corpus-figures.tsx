"use client"

import * as React from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts"

import {
  axisProps,
  gridProps,
  verdictColors,
  yAxisProps,
} from "@/components/training/chart-theme"
import { Figure, FigurePlaceholder } from "@/components/training/stat-run"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import type { CorpusStats } from "@/lib/synth"

/**
 * How long the conversations are, against the window they have to fit in.
 *
 * This is the one figure the corpus page exists for. A conversation longer than the
 * context window is not dropped by `build_sft_dataset` — it is *chunked*, so the
 * model trains on windows holding the middle of a reply with the question no longer
 * visible. That failure is invisible in every other view: the file looks fine, the
 * conversation count goes up, and the loss curve does not complain.
 *
 * Lengths are real tokens, rendered through the same chat template and merges SFT
 * uses, so the boundary drawn here is the exact one that will apply.
 *
 * Fits / does not fit is a polarity pair, so it reuses the catalogue's blue/amber
 * (see `chart-theme.ts`) — the CVD-validated opposition, never red/green. The bar's
 * side of the boundary is also readable from its position, so colour is a second
 * cue rather than the only one.
 */

const chartConfig = {
  fits: { label: "Fits", color: verdictColors.approved },
  over: { label: "Too long", color: verdictColors.rejected },
} satisfies ChartConfig

export interface CorpusFiguresProps {
  corpus: CorpusStats
}

export function CorpusFigures({ corpus }: CorpusFiguresProps) {
  const tokens = React.useMemo(
    () => corpus.files.flatMap((f) => f.tokens),
    [corpus.files]
  )
  const ctx = corpus.ctx_window
  const histogram = React.useMemo(() => bin(tokens, ctx), [tokens, ctx])

  if (ctx == null) {
    return (
      <Figure caption="Conversation length">
        <FigurePlaceholder className="h-64">
          Pick the model these will train, and every conversation is measured
          against its context window.
        </FigurePlaceholder>
      </Figure>
    )
  }

  const fits = tokens.length - corpus.over_budget

  return (
    <Figure
      caption="Conversation length against the context window"
      aside={
        <span className="flex items-center gap-3 tabular-nums">
          <Swatch color={verdictColors.approved} /> {fits} fit
          <Swatch color={verdictColors.rejected} /> {corpus.over_budget} too
          long
        </span>
      }
    >
      {histogram.length > 0 ? (
        <ChartContainer
          config={chartConfig}
          className="aspect-auto h-64 w-full"
        >
          {/* Extra top margin so the boundary label has somewhere to sit. */}
          <BarChart data={histogram} margin={{ left: 4, right: 4, top: 16 }}>
            <CartesianGrid {...gridProps} />
            <XAxis
              dataKey="mid"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(v: number) => `${Math.round(v)}`}
              {...axisProps}
            />
            <YAxis allowDecimals={false} {...yAxisProps} />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(v) => `≈ ${Math.round(Number(v))} tokens`}
                />
              }
            />
            <ReferenceLine
              x={ctx}
              stroke="var(--muted-foreground)"
              strokeDasharray="3 3"
              label={{
                value: `${ctx} token window`,
                position: "top",
                fontSize: 10,
                fill: "var(--muted-foreground)",
              }}
            />
            {/* One bar per bin, coloured by which side of the window it fell on —
                the split is the boundary, so a stacked pair would be two colours
                where the data only ever has one. */}
            <Bar dataKey="count" name="Conversations" isAnimationActive={false}>
              {histogram.map((b) => (
                <Cell
                  key={b.mid}
                  fill={
                    b.fits ? verdictColors.approved : verdictColors.rejected
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      ) : (
        <FigurePlaceholder className="h-64">
          No conversations in the SFT directory yet.
        </FigurePlaceholder>
      )}
    </Figure>
  )
}

function Swatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden
      className="inline-block size-2 rounded-[2px] align-middle"
      style={{ background: color }}
    />
  )
}

const BINS = 28

/**
 * Buckets token lengths, tagging each bin with whether it fits.
 *
 * The upper edge is clamped to four windows: a single 112k-token conversation (the
 * gold set has one) would otherwise put every real conversation in the first bin
 * and render the figure useless. The clamped ones still count — they land in the
 * last bin, which is on the "too long" side either way.
 */
function bin(tokens: number[], ctx: number | null) {
  if (tokens.length === 0) return []

  const cap = ctx ? ctx * 4 : Math.max(...tokens)
  const hi = Math.min(Math.max(...tokens), cap)
  const width = hi > 0 ? hi / BINS : 1

  const buckets = Array.from({ length: BINS }, (_, i) => ({
    mid: width * (i + 0.5),
    count: 0,
  }))
  for (const t of tokens) {
    const idx = Math.min(BINS - 1, Math.floor(t / width))
    buckets[idx].count += 1
  }

  return buckets
    .filter((b) => b.count > 0)
    .map((b) => ({ ...b, fits: ctx == null || b.mid <= ctx }))
}
