"use client"

import * as React from "react"
import { Bar, BarChart, CartesianGrid, Cell, XAxis, YAxis } from "recharts"

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import {
  axisProps,
  BUILD_COLOR,
  BUILD_OVERFLOW_COLOR,
  gridProps,
  yAxisProps,
} from "@/components/training/chart-theme"
import { SpanRibbon } from "@/components/training/span-ribbon"
import { compact, Figure, StatRun, type Stat } from "@/components/training/stat-run"
import type { Fertility, TokenizeStats } from "@/lib/training"

/**
 * What the tokenization and the packing actually produced.
 *
 * Phase 1's counters say how much there was; these say what it *is* — where the
 * context budget goes per source, how much of the vocabulary the corpus never
 * uses, how often a training window spans two documents, and how document
 * lengths sit against the window. Each one changes a decision: the vocabulary
 * size, the selection, the window, or whether to pack differently.
 *
 * Rendered from the frame rather than from the config, and kept on screen once
 * training starts: these are facts about the corpus the run is training on, and
 * a finished run's page is where they are actually read.
 */
export interface BuildMetricsProps {
  tokenize: Partial<TokenizeStats>
  /** The window the run trains at, which the length histogram is measured against. */
  ctxWindow: number
}

// How many x labels the length histogram carries. Four across two windows reads
// as 0×, ½×, 1×, 1½× — the marks that mean something — where a tick per bin is
// 24 labels nobody reads.
const HIST_LABELS = 4

const histConfig = {
  fraction: { label: "Documents", color: BUILD_COLOR },
} satisfies ChartConfig

const fertilityConfig = {
  chars_per_token: { label: "Chars / token", color: BUILD_COLOR },
} satisfies ChartConfig

export function BuildMetrics({ tokenize, ctxWindow }: BuildMetricsProps) {
  const {
    fertility,
    dead_vocab: dead,
    merge_sequence: merges,
    boundary_straddle: straddle,
    doc_length_hist: hist,
    docs_over_ctx: over,
    span_ribbon: ribbon,
    vocab,
  } = tokenize

  const bars = React.useMemo(() => {
    if (!hist?.length || !ctxWindow) return []
    const perBin = (2 * ctxWindow) / hist.length
    return hist.map((fraction, bin) => ({
      bin,
      windows: (2 * bin) / hist.length,
      range: `${Math.round(bin * perBin).toLocaleString()}–${Math.round((bin + 1) * perBin).toLocaleString()} tokens`,
      fraction,
      // The histogram reaches two windows, so its own midpoint is where a
      // document stops fitting one — everything from there on is the tail no
      // single window can hold.
      over: bin >= hist.length / 2,
    }))
  }, [hist, ctxWindow])

  if (!fertility && dead == null && straddle == null && !merges) return null

  return (
    <section className="space-y-6">
      <StatRun stats={headline(fertility, dead, vocab, straddle, over)} />

      {/* Directly under the straddle it explains, and above the length
          histogram: the number, then what it looks like, then the distribution
          behind both. */}
      {ribbon && (
        <Figure
          caption={`${ribbon.windows.length} training windows sampled evenly across the corpus, top to bottom in packing order — each bar is one ${ctxWindow}-token window, split by the documents inside it`}
          aside={
            <span>Hairlines are document boundaries; a thin bar is a tail too fine to draw</span>
          }
          minHeight="0"
        >
          <SpanRibbon ribbon={ribbon} ctxWindow={ctxWindow} />
        </Figure>
      )}

      {bars.length > 0 && (
        <Figure
          caption={`Document length against the ${ctxWindow}-token window — the last bin holds everything longer than two windows`}
          minHeight="14rem"
        >
          <ChartContainer config={histConfig} className="aspect-auto h-56 w-full">
            <BarChart data={bars} margin={{ left: 4, right: 4, top: 4 }}>
              <CartesianGrid {...gridProps} />
              <XAxis
                dataKey="windows"
                interval={Math.max(0, Math.round(bars.length / HIST_LABELS) - 1)}
                tickFormatter={(v: number) => `${v}×`}
                {...axisProps}
              />
              <YAxis
                tickFormatter={(v: number) => percent(v)}
                {...yAxisProps}
              />
              <ChartTooltip
                cursor={false}
                content={
                  <ChartTooltipContent
                    labelFormatter={(_, payload) =>
                      String(payload?.[0]?.payload?.range ?? "")
                    }
                    formatter={(v) => percent(Number(v))}
                  />
                }
              />
              <Bar dataKey="fraction" isAnimationActive={false}>
                {bars.map((b) => (
                  <Cell
                    key={b.bin}
                    fill={b.over ? BUILD_OVERFLOW_COLOR : BUILD_COLOR}
                  />
                ))}
              </Bar>
            </BarChart>
          </ChartContainer>
        </Figure>
      )}

      {fertility && fertility.length > 0 && (
        <Figure
          caption="Characters per token, by source — a low bar spends more of the window on the same text"
          minHeight="9rem"
        >
          <ChartContainer
            config={fertilityConfig}
            className="aspect-auto w-full"
            style={{ height: `${2 + fertility.length * 1.75}rem` }}
          >
            <BarChart
              layout="vertical"
              data={fertility}
              margin={{ left: 4, right: 4, top: 4 }}
            >
              <CartesianGrid {...gridProps} horizontal={false} />
              <XAxis type="number" {...axisProps} />
              {/* Wider than the shared width: this axis carries dataset names
                  rather than a number. */}
              <YAxis
                type="category"
                dataKey="dataset"
                {...yAxisProps}
                width={120}
              />
              <ChartTooltip
                cursor={false}
                content={
                  <ChartTooltipContent
                    formatter={(v) => `${Number(v).toFixed(2)} chars / token`}
                  />
                }
              />
              <Bar
                dataKey="chars_per_token"
                fill={BUILD_COLOR}
                radius={2}
                isAnimationActive={false}
              />
            </BarChart>
          </ChartContainer>
        </Figure>
      )}

      {merges && merges.length > 0 && (
        <Figure
          caption={`The first ${merges.length} merges, in the order BPE learned them — with how often each lands in the packed corpus`}
          minHeight="0"
        >
          <ol className="flex max-h-64 flex-wrap gap-x-2 gap-y-1 overflow-y-auto rounded-(--radius-control-sm) border border-dashed p-3">
            {merges.map((merge, rank) => (
              <li
                key={`${rank}-${merge.token}`}
                className="flex items-baseline gap-1 text-xs"
                title={`Merge ${rank + 1}`}
              >
                <span className="whitespace-pre font-mono text-foreground">
                  {merge.token}
                </span>
                {merge.count != null && (
                  <span className="text-muted-foreground tabular-nums">
                    {compact(merge.count)}
                  </span>
                )}
              </li>
            ))}
          </ol>
        </Figure>
      )}
    </section>
  )
}

/**
 * The four figures over the charts. Each is omitted rather than zeroed when the
 * build did not measure it — a corpus packed before these existed reports none
 * of them, and "0%" would read as a real reading of nothing.
 */
function headline(
  fertility: Fertility[] | undefined,
  dead: number | undefined,
  vocab: number | undefined,
  straddle: number | undefined,
  over: number | undefined
): Stat[] {
  const stats: Stat[] = [
    { label: "Chars / token", value: overall(fertility) },
    {
      label: "Dead vocab",
      value:
        dead == null
          ? "—"
          : vocab
            ? `${compact(dead)} · ${percent(dead / vocab)}`
            : compact(dead),
    },
    {
      label: "Windows crossing a document",
      value: straddle == null ? "—" : percent(straddle),
    },
    {
      label: "Documents over the window",
      value: over == null ? "—" : percent(over),
    },
  ]
  return stats
}

/** The whole corpus's fertility: total characters over total tokens, not a mean
 * of the per-source ratios, which would weight a 400 KB dataset like a 40 GB one. */
function overall(fertility: Fertility[] | undefined): string {
  if (!fertility?.length) return "—"
  let chars = 0
  let tokens = 0
  for (const one of fertility) {
    chars += one.chars_per_token * one.tokens
    tokens += one.tokens
  }
  return tokens > 0 ? (chars / tokens).toFixed(2) : "—"
}

function percent(fraction: number): string {
  if (!Number.isFinite(fraction)) return "—"
  const pct = fraction * 100
  return pct > 0 && pct < 0.1 ? "<0.1%" : `${pct.toFixed(pct < 10 ? 1 : 0)}%`
}
