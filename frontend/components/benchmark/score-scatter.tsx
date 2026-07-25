"use client"

import { CartesianGrid, Scatter, ScatterChart, XAxis, YAxis, ZAxis } from "recharts"

import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart"
import { axisProps, gridProps, yAxisProps } from "@/components/training/chart-theme"
import { formatDate, formatNumber } from "@/lib/benchmark"

// Score on y against one reversed, logarithmic measure on x. Shared by the scaling
// panel (x = training tokens) and the duration panel (x = time to complete), which
// are the same figure asking two versions of one question: what does a point of eval
// score cost?
//
// Scatter, NOT a line. Consecutive points are different models — different widths,
// depths, curricula — so a line through them would draw one trajectory through things
// that are not a series, and imply that the segment between two points is a path some
// model took. The cloud's shape carries the trend on its own.
//
// The x axis is REVERSED and LOGARITHMIC. Reversed because the interesting end is the
// expensive, well-trained one, and putting it first means the eye lands on the best
// model before it reads the decline. Logarithmic because both measures grow by
// multiples, not increments: on a linear axis, the moment one model costs 10× another,
// every cheaper run collapses into a stack against one edge while the empty space
// above the largest takes half the plot.
//
// The cost of log spacing is that neither end can be labelled — dividing toward zero
// and multiplying toward infinity are both journeys of infinite length, so zero has no
// position on this axis. The ticks are round values inside the data's range instead,
// which is what gives a dot its orientation.
//
// One series, so no legend: the panel title names it.

const chartConfig = {
  score: { label: "Eval score", color: "var(--primary)" },
} satisfies ChartConfig

export interface ScorePoint {
  /** The x measure: tokens, seconds — whatever the panel is plotting against. */
  x: number
  score: number
  name: string
  created_at: string
}

// A point with its label's placement worked out. Identity is carried by a name beside
// the dot rather than by colour: the site's five chart hues only separate three ways
// under a colourblind check (violet and blue are 6 ΔE apart, which is hard to tell
// apart with full colour vision), and a name is unambiguous for any number of models
// where a fourth hue would not be.
interface PlacedPoint extends ScorePoint {
  /** Which side of the dot the label sits on, so it never runs off the plot. */
  side: "left" | "right"
  /** Vertical nudge, in px, that keeps two crowded labels off each other. */
  dy: number
  label: string
  /** The label's horizontal extent in px, so two of them can be tested for overlap. */
  x0: number
  x1: number
  /** Where the label's own row ended up, in px down the plot. */
  labelY: number
}

// Longest model name kept intact. Past this the label starts stealing plot width from
// the dots it's meant to be annotating.
const MAX_LABEL = 16

// Gap between a dot and its label, in px. Matches the one the shape draws with.
const LABEL_GAP = 9

// Rough width of one character at the label's 10px size, for estimating how far a label
// reaches. Approximate on purpose — the text isn't measurable before it's drawn, and
// this only has to be close enough to decide whether two labels want the same row.
const CHAR_WIDTH = 5.3

// The plot area's width in px. Like PLOT_HEIGHT this is nominal: the chart is responsive,
// so a narrower viewport makes labels reach relatively further and a wider one gives them
// more room than assumed here.
const PLOT_WIDTH = 480

// Rows a label will try before giving up and sitting where it lands. Six covers a
// cluster of seven models at the same cost; past that the figure is the wrong tool.
const MAX_LABEL_ROWS = 6

// Labels flip to the left of their dot past this much of the way across the plot.
const FLIP_AT = 0.7

// One row of vertical clearance between two nudged labels, in px.
const LABEL_ROW = 13

// The plot area's height in px, used only to convert a score into a label row. Tracks
// the `h-56` container below minus its margins and the x axis; it doesn't have to be
// exact, since it only sets how far apart two crowded labels end up.
const PLOT_HEIGHT = 188

// Works out where each label goes, in normalized plot space, before recharts has drawn
// anything: the parent is the only place that can see all the points at once, and
// therefore the only place that can tell whether two of them collide.
function placeLabels(points: ScorePoint[], [min, max]: [number, number]): PlacedPoint[] {
  const span = Math.log10(max) - Math.log10(min)
  const placed: PlacedPoint[] = []

  for (const point of points) {
    // The axis is reversed, so a large x lands at a *small* screen fraction.
    const fx = span > 0 ? 1 - (Math.log10(point.x) - Math.log10(min)) / span : 0.5
    const cy = (1 - Math.min(Math.max(point.score, 0), 100) / 100) * PLOT_HEIGHT

    const label =
      point.name.length > MAX_LABEL ? `${point.name.slice(0, MAX_LABEL - 1)}…` : point.name
    const side = fx > FLIP_AT ? "left" : "right"
    // Where this label will actually sit. Two dots far enough apart to look unrelated
    // still collide when both labels point inward at each other, so the test has to be
    // on the text's own extent rather than on the distance between the dots.
    const reach = LABEL_GAP + label.length * CHAR_WIDTH
    const dotX = fx * PLOT_WIDTH
    const x0 = side === "left" ? dotX - reach : dotX
    const x1 = side === "left" ? dotX : dotX + reach

    // Tested against where earlier labels *ended up*, never against their dots: a label
    // nudged clear of its own neighbour can land on a third label a row above, and the
    // dot positions never reveal that.
    const occupied = (y: number) =>
      placed.some(
        (other) =>
          other.x0 < x1 && x0 < other.x1 && Math.abs(other.labelY - y) < LABEL_ROW
      )

    // Alternating rows outward from the dot — up, down, further up — until one is free.
    let labelY = cy
    for (let step = 1; occupied(labelY) && step <= MAX_LABEL_ROWS; step++) {
      labelY = cy + (step % 2 === 1 ? -1 : 1) * LABEL_ROW * Math.ceil(step / 2)
    }

    placed.push({ ...point, side, label, x0, x1, labelY, dy: labelY - cy })
  }

  return placed
}

export interface ScoreScatterProps {
  points: ScorePoint[]
  /** Axis tick labels, and the same value spelled out in the tooltip. */
  formatX: (value: number) => string
  /** How the tooltip names the x measure, e.g. "training tokens". */
  xLabel: string
  /** Tick positions, from `logTicks(logDomain(...))`. */
  ticks: number[]
  domain: [number, number]
}

export function ScoreScatter({ points, formatX, xLabel, ticks, domain }: ScoreScatterProps) {
  const placed = placeLabels(points, domain)

  return (
    <ChartContainer config={chartConfig} className="aspect-auto h-56 w-full">
      <ScatterChart margin={{ left: 4, right: 16, top: 8, bottom: 4 }}>
        <CartesianGrid {...gridProps} stroke="var(--border)" strokeOpacity={0.6} />
        <XAxis
          {...axisProps}
          dataKey="x"
          type="number"
          scale="log"
          reversed
          domain={domain}
          // The domain is padded past the data, so recharts has to be allowed to draw
          // outside the data extent or it clamps back to it and re-crops.
          allowDataOverflow
          ticks={ticks}
          tickFormatter={formatX}
          // The default 32px gap silently drops labels to avoid collisions; the ticks
          // are already spaced to fit, so only a genuine overlap should cost one.
          minTickGap={4}
          className="text-[10px]"
        />
        <YAxis
          {...yAxisProps}
          dataKey="score"
          type="number"
          domain={[0, 100]}
          className="text-[10px]"
        />
        {/* Fixed marker size: the z channel carries nothing here, and letting recharts
            scale dots by an absent value would invent a third dimension. */}
        <ZAxis range={[48, 48]} />
        <ChartTooltip
          cursor={{ stroke: "var(--border)" }}
          content={<PointTooltip formatX={formatX} xLabel={xLabel} />}
        />
        <Scatter
          data={placed}
          shape={(props: unknown) => <LabelledDot {...(props as LabelledDotProps)} />}
          isAnimationActive={false}
        />
      </ScatterChart>
    </ChartContainer>
  )
}

interface LabelledDotProps {
  cx?: number
  cy?: number
  payload?: PlacedPoint
}

// One model: an 8px dot with its name beside it.
//
// Drawn as a custom shape rather than a `<LabelList>` because the label's side and
// vertical nudge are per-point decisions made in `placeLabels`, and a LabelList applies
// one static position to the whole series.
function LabelledDot({ cx, cy, payload }: LabelledDotProps) {
  if (cx === undefined || cy === undefined || !payload) return null
  const gap = 9
  const toLeft = payload.side === "left"

  return (
    <g>
      {/* A surface-coloured ring, so two models with similar costs stay countable as
          two marks where they overlap. */}
      <circle
        cx={cx}
        cy={cy}
        r={4}
        fill="var(--primary)"
        fillOpacity={0.75}
        stroke="var(--background)"
        strokeWidth={2}
      />
      {/* Text wears a text token, never the mark's colour: the dot carries the data,
          the label only names it. */}
      <text
        x={toLeft ? cx - gap : cx + gap}
        y={cy + payload.dy}
        dy="0.32em"
        textAnchor={toLeft ? "end" : "start"}
        className="fill-muted-foreground text-[10px]"
      >
        {payload.label}
      </text>
    </g>
  )
}

// Multiplicative padding, because the axis is logarithmic: a constant margin would be
// huge at the low end and invisible at the high end. Keeps the outermost points off
// the frame.
export function logDomain(values: number[]): [number, number] {
  return [Math.min(...values) / 1.3, Math.max(...values) * 1.3]
}

// Rungs to try per decade, sparsest first. On a log axis the number of round numbers
// inside the range depends entirely on how many decades it spans: the 1/2/5 ladder
// that gives nine sensible labels across three decades gives only two across a range
// where every model is within a factor of two of the others. So the ladder adapts
// instead of being fixed.
const LADDERS = [[1], [1, 2, 5], [1, 2, 3, 5, 7], [1, 2, 3, 4, 5, 6, 7, 8, 9]]

// Above this many labels the axis is a wall of text and they start to collide.
const MAX_TICKS = 9

// Round-number ticks inside a log domain: the densest ladder that still fits.
//
// Every tick is a genuine round number at its own magnitude (…400K, 500K, 600K… or
// …1M, 2M, 5M…), never an interpolated value, so a label means the same thing wherever
// it lands on the axis.
export function logTicks(domain: [number, number]): number[] {
  const [min, max] = domain
  return pickDensest(
    LADDERS.map((ladder) => rungs(ladder, min, max)),
    domain
  )
}

/**
 * Picks the densest set of ticks that still fits the axis, from tiers ordered
 * sparsest-first and already filtered to the domain.
 *
 * Exported because "round" is measure-specific: powers of ten are round for tokens,
 * but time is round at 30s, 5m and 1h — 100s and 2000s are not, and no mantissa ladder
 * will produce the former. A measure with its own idea of a round value supplies its
 * own tiers and reuses this.
 */
export function pickDensest(tiers: number[][], domain: [number, number]): number[] {
  let best: number[] = []
  for (const tier of tiers) {
    if (tier.length > MAX_TICKS) break
    best = tier
  }
  // A domain so narrow that not even the finest tier has two values in it (a single
  // model, or several within a few percent) falls back to the ends, so the axis is
  // never left blank.
  return best.length >= 2 ? best : domain.map(Math.round)
}

function rungs(ladder: number[], min: number, max: number): number[] {
  const out: number[] = []
  for (let exponent = Math.floor(Math.log10(min)); exponent <= Math.ceil(Math.log10(max)); exponent++) {
    for (const mantissa of ladder) {
      const value = mantissa * 10 ** exponent
      if (value >= min && value <= max) out.push(value)
    }
  }
  return out
}

function PointTooltip({
  active,
  payload,
  formatX,
  xLabel,
}: {
  active?: boolean
  payload?: { payload: ScorePoint }[]
  formatX: (value: number) => string
  xLabel: string
}) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  return (
    <div className="min-w-40 rounded-lg border border-border/60 bg-popover px-3 py-2 text-xs text-popover-foreground shadow-md">
      <div className="mb-1 text-[10px] font-medium uppercase tracking-[0.12em] text-muted-foreground/70">
        {formatDate(point.created_at)}
      </div>
      <div className="flex items-baseline justify-between gap-6">
        <span className="truncate">{point.name}</span>
        <span className="font-medium tabular-nums">{formatNumber(point.score, 1)}</span>
      </div>
      <div className="mt-1 text-[11px] tabular-nums text-muted-foreground">
        {formatX(point.x)} {xLabel}
      </div>
    </div>
  )
}
