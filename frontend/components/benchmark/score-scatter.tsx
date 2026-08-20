"use client"

import * as React from "react"
import {
  CartesianGrid,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts"

import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart"
import {
  axisProps,
  gridProps,
  yAxisProps,
} from "@/components/training/chart-theme"
import {
  exactDecimals,
  formatDate,
  formatNumber,
  TIER_ORDER,
  type Tier,
} from "@/lib/benchmark"
import { TIER_LABEL } from "@/lib/jobs"

// Score on y against one reversed, logarithmic measure on x. Shared by the tokens
// panel (x = tokens the run forwarded) and the duration panel (x = time to complete),
// which are the same figure asking two versions of one question: what does a point of
// eval score cost?
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
// No legend. A lone run is labelled with its model name; a budget sweep is a joined
// line labelled once, at its highest-budget end, with the model whose sweep it is —
// its dots then carry the budgets, which are what differ along it. A legend would be a
// second place to look for the same names, and the names are already on the plot.

const chartConfig = {
  score: { label: "Eval score", color: "var(--primary)" },
} satisfies ChartConfig

export interface ScorePoint {
  /** The x measure: tokens, seconds — whatever the panel is plotting against. */
  x: number
  score: number
  /** The dot's own label: the model for a lone run, the budget within a sweep. */
  name: string
  /**
   * The model, always — even when `name` is a budget. A sweep's three dots are one
   * model and are labelled by budget, so without this a plot of two swept models is
   * two lines both reading "Short · Medium · Long" with nothing saying whose is
   * whose. It is drawn once per line, on the sweep's anchor point.
   */
  model: string
  created_at: string
  /**
   * The budget sweep this point belongs to, if any. Points sharing one are drawn as
   * a connected line in budget order: they are the same model measured three ways,
   * which is the one case where joining dots is not the lie the scatter otherwise
   * refuses to tell — a line here traces a real path, because a single model really
   * did move along it as it was told to think harder.
   */
  series?: string | null
  /** Which budget this point was scored at, when it belongs to a sweep. */
  budget?: Tier | null
  /** Held-out conversations behind the score, when the measure has any. */
  conversations?: number | null
  /**
   * Too few conversations to trust as a measurement. Drawn hollow, so a point resting
   * on three conversations cannot be read as one resting on fifteen.
   */
  provisional?: boolean
  /**
   * This point carries its sweep's model name. Exactly one point per sweep has it —
   * see `markSeriesAnchors` — so the line is named once rather than three times.
   */
  anchor?: boolean
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

// The model name a sweep's anchor shows, truncated on the same rule as a dot label.
function seriesLabel(point: ScorePoint): string {
  return truncate(point.model)
}

function truncate(text: string): string {
  return text.length > MAX_LABEL ? `${text.slice(0, MAX_LABEL - 1)}…` : text
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
// more room than assumed here. Deliberately an under-estimate of the full-width panel on a
// large monitor — guessing narrow costs a label an unnecessary row, guessing wide lets two
// labels overlap, and only one of those is visible as a mistake.
const PLOT_WIDTH = 760

// Rows a label will try before giving up and sitting where it lands. Six covers a
// cluster of seven models at the same cost; past that the figure is the wrong tool.
const MAX_LABEL_ROWS = 6

// Labels flip to the left of their dot past this much of the way across the plot.
const FLIP_AT = 0.7

// One row of vertical clearance between two nudged labels, in px.
const LABEL_ROW = 13

// The plot area's height in px, used only to convert a score into a label row. Tracks
// the `PLOT_CLASS` container below minus its margins and the x axis; it doesn't have to
// be exact, since it only sets how far apart two crowded labels end up.
const PLOT_HEIGHT = 344

// The figure's own height. A scatter separating a dozen labelled runs needs room in both
// directions — at the old 224px, half of it spent on axis and margins, two models a point
// apart in score were a point apart in *pixels* and their labels were nudged onto rows
// that had nothing to do with their dots.
const PLOT_CLASS = "aspect-auto h-96 w-full"

/**
 * Marks the one point per sweep that carries the model name.
 *
 * The **highest budget** end, which is the end of the path the line traces and — since
 * more thinking costs more tokens and x is reversed — normally its leftmost, least
 * crowded point. Naming the line once beats naming every dot: three dots reading
 * "Balanced 1.01" tell you nothing the line already showed, and they crowd out the
 * budget labels, which are the thing that varies.
 *
 * A sweep of one surviving run is not a line and gets no anchor — it draws as an
 * ordinary dot, already labelled with its own model name by the panel.
 */
function markSeriesAnchors(points: ScorePoint[]): ScorePoint[] {
  const best = new Map<string, ScorePoint>()
  const counts = new Map<string, number>()
  for (const point of points) {
    if (!point.series) continue
    counts.set(point.series, (counts.get(point.series) ?? 0) + 1)
    const incumbent = best.get(point.series)
    const rank = TIER_ORDER.indexOf(point.budget as Tier)
    if (!incumbent || rank > TIER_ORDER.indexOf(incumbent.budget as Tier)) {
      best.set(point.series, point)
    }
  }

  const anchors = new Set<ScorePoint>()
  for (const [series, point] of best) {
    if ((counts.get(series) ?? 0) >= 2) anchors.add(point)
  }
  return points.map((point) =>
    anchors.has(point) ? { ...point, anchor: true } : point
  )
}

// Works out where each label goes, in normalized plot space, before recharts has drawn
// anything: the parent is the only place that can see all the points at once, and
// therefore the only place that can tell whether two of them collide.
function placeLabels(
  points: ScorePoint[],
  [min, max]: [number, number],
  [scoreMin, scoreMax]: [number, number]
): PlacedPoint[] {
  const span = Math.log10(max) - Math.log10(min)
  // Both domains, because a label's position is a point in the plot and the plot has
  // two axes. When y was pinned to [0, 100] this only had to know the constant; now
  // that it fits the data, a `cy` computed against the old range would put every
  // label in a different place from its own dot.
  const scoreSpan = scoreMax - scoreMin
  const placed: PlacedPoint[] = []

  for (const point of points) {
    // The axis is reversed, so a large x lands at a *small* screen fraction.
    const fx =
      span > 0 ? 1 - (Math.log10(point.x) - Math.log10(min)) / span : 0.5
    const fy =
      scoreSpan > 0
        ? Math.min(Math.max((point.score - scoreMin) / scoreSpan, 0), 1)
        : 0.5
    const cy = (1 - fy) * PLOT_HEIGHT

    const label = truncate(point.name)
    const side = fx > FLIP_AT ? "left" : "right"
    // Where this label will actually sit. Two dots far enough apart to look unrelated
    // still collide when both labels point inward at each other, so the test has to be
    // on the text's own extent rather than on the distance between the dots.
    //
    // An anchor draws two rows — the model name above its budget — so it reserves the
    // width of the wider row and, below, the height of both.
    const widest = point.anchor
      ? Math.max(label.length, seriesLabel(point).length)
      : label.length
    const reach = LABEL_GAP + widest * CHAR_WIDTH
    const dotX = fx * PLOT_WIDTH
    const x0 = side === "left" ? dotX - reach : dotX
    const x1 = side === "left" ? dotX : dotX + reach

    // Tested against where earlier labels *ended up*, never against their dots: a label
    // nudged clear of its own neighbour can land on a third label a row above, and the
    // dot positions never reveal that.
    //
    // The clearance is a row taller when either label is an anchor, because an anchor
    // occupies the row above its own as well.
    const occupied = (y: number) =>
      placed.some((other) => {
        if (other.x0 >= x1 || x0 >= other.x1) return false
        const rows = 1 + (point.anchor ? 1 : 0) + (other.anchor ? 1 : 0)
        return Math.abs(other.labelY - y) < LABEL_ROW * rows
      })

    // A sweep's labels start a row *below* their dot rather than beside it: the line
    // joining the three runs through them, and text with a stroke drawn through it is
    // the one collision `occupied` cannot see. An anchor clears two rows, since its
    // model name sits above its budget and both have to end up under the line.
    const bias = point.series ? LABEL_ROW * (point.anchor ? 2 : 1) : 0

    // Alternating rows outward from the dot — up, down, further up — until one is free.
    let labelY = cy + bias
    for (let step = 1; occupied(labelY) && step <= MAX_LABEL_ROWS; step++) {
      labelY =
        cy + bias + (step % 2 === 1 ? -1 : 1) * LABEL_ROW * Math.ceil(step / 2)
    }

    placed.push({ ...point, side, label, x0, x1, labelY, dy: labelY - cy })
  }

  return placed
}

export interface ScoreScatterProps {
  points: ScorePoint[]
  /** Tick positions, from the panel's own idea of a round value. */
  ticks: number[]
  /** Axis tick labels, and the same value spelled out in the tooltip. */
  formatX: (value: number) => string
  /** How the tooltip names the x measure, e.g. "training tokens". */
  xLabel: string
  /** The x range, fitted to the points by the panel. */
  domain: [number, number]
}

/**
 * Splits the points into budget sweeps and loners.
 *
 * A sweep becomes its own `<Scatter>` so recharts draws a line through it, and its
 * members are ordered low → med → high rather than by x: the line is a path through
 * increasing thinking, and sorting it by cost would draw the same three dots joined
 * in whatever order the token counts happened to fall.
 */
function groupSeries(placed: PlacedPoint[]): {
  sweeps: PlacedPoint[][]
  loners: PlacedPoint[]
} {
  const bySeries = new Map<string, PlacedPoint[]>()
  const loners: PlacedPoint[] = []
  for (const point of placed) {
    if (!point.series) {
      loners.push(point)
      continue
    }
    const existing = bySeries.get(point.series)
    if (existing) existing.push(point)
    else bySeries.set(point.series, [point])
  }

  const sweeps: PlacedPoint[][] = []
  for (const members of bySeries.values()) {
    // A sweep whose other two runs failed or are still queued is one dot, and a
    // one-point line is not a line — it joins the loners and draws as an ordinary
    // mark rather than as a series that looks broken.
    if (members.length < 2) {
      loners.push(...members)
      continue
    }
    sweeps.push(
      [...members].sort(
        (a, b) =>
          TIER_ORDER.indexOf(a.budget as Tier) -
          TIER_ORDER.indexOf(b.budget as Tier)
      )
    )
  }
  return { sweeps, loners }
}

export function ScoreScatter({
  points,
  formatX,
  xLabel,
  ticks,
  domain,
}: ScoreScatterProps) {
  // Fitted to the scores present, like x is to the costs. Computed here rather than
  // taken as a prop: every caller would derive it the same way from the same points,
  // and `placeLabels` has to be handed the identical range.
  const scoreRange = scoreDomain(points.map((p) => p.score))
  const scoreTicks = steppedTicks(scoreRange[0], scoreRange[1])
  // Anchors are decided before placement, because an anchor's label is two rows tall
  // and the collision pass has to know that while it is choosing rows.
  const placed = placeLabels(markSeriesAnchors(points), domain, scoreRange)
  const { sweeps, loners } = groupSeries(placed)

  return (
    <ChartContainer config={chartConfig} className={PLOT_CLASS}>
      <ScatterChart margin={{ left: 4, right: 16, top: 8, bottom: 4 }}>
        <CartesianGrid
          {...gridProps}
          stroke="var(--border)"
          strokeOpacity={0.6}
        />
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
          className="text-xs"
        />
        <YAxis
          {...yAxisProps}
          dataKey="score"
          type="number"
          domain={scoreRange}
          // Same reason as the x axis: the domain is padded past the data, so
          // recharts has to be allowed to draw outside the data extent rather than
          // clamping back to it.
          allowDataOverflow
          ticks={scoreTicks.length >= 2 ? scoreTicks : undefined}
          tickFormatter={scoreTickFormat(scoreTicks)}
          className="text-xs"
        />
        {/* Fixed marker size: the z channel carries nothing here, and letting recharts
            scale dots by an absent value would invent a third dimension. */}
        <ZAxis range={[48, 48]} />
        <ChartTooltip
          cursor={{ stroke: "var(--border)" }}
          content={<PointTooltip formatX={formatX} xLabel={xLabel} />}
        />
        {/* One series per sweep, so recharts joins its three points; `line` draws in
            data order, which `groupSeries` has already put in budget order. */}
        {sweeps.map((members) => (
          <Scatter
            key={members[0].series}
            data={members}
            line={{ stroke: "var(--primary)", strokeWidth: 1.5 }}
            lineType="joint"
            shape={(props: unknown) => (
              <LabelledDot {...(props as LabelledDotProps)} />
            )}
            isAnimationActive={false}
          />
        ))}
        <Scatter
          data={loners}
          shape={(props: unknown) => (
            <LabelledDot {...(props as LabelledDotProps)} />
          )}
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
          two marks where they overlap.

          A provisional point is drawn hollow instead — filled with the page rather
          than the mark colour, ringed in primary. It reads as "a measurement that is
          there but thin" without needing a legend, and it is the same convention a
          dashed line uses for a projection. */}
      <circle
        cx={cx}
        cy={cy}
        r={4}
        fill={payload.provisional ? "var(--background)" : "var(--primary)"}
        fillOpacity={payload.provisional ? 1 : 0.75}
        stroke={payload.provisional ? "var(--primary)" : "var(--background)"}
        strokeWidth={payload.provisional ? 1.5 : 2}
      />
      {/* The sweep's name, on its anchor point only — one line, one name. Carried in
          the foreground token and a heavier weight than the budget below it, because
          it identifies the whole line where the budget identifies one dot. */}
      {payload.anchor && (
        <text
          x={toLeft ? cx - gap : cx + gap}
          y={cy + payload.dy - LABEL_ROW}
          dy="0.32em"
          textAnchor={toLeft ? "end" : "start"}
          className="fill-foreground text-xs font-medium"
        >
          {seriesLabel(payload)}
        </text>
      )}
      {/* Text wears a text token, never the mark's colour: the dot carries the data,
          the label only names it. */}
      <text
        x={toLeft ? cx - gap : cx + gap}
        y={cy + payload.dy}
        dy="0.32em"
        textAnchor={toLeft ? "end" : "start"}
        className="fill-muted-foreground text-xs"
      >
        {payload.label}
      </text>
    </g>
  )
}

// Share of the data's own log span added at each end as breathing room. Proportional,
// not a fixed factor, which is the whole point: these measures cluster far harder than
// they look like they should. Four language runs came in between 308,449 and 308,630
// tokens — a spread of 0.0003 decades, against the 0.23 decades a flat ÷1.3 / ×1.3
// padding added. The data occupied **0.11% of the axis** and drew as one dot.
const PAD_SHARE = 0.12

// Padding, in decades, when every point shares one x. There is no span to take a share
// of, and a zero-width domain is not a log axis at all, so the axis becomes a ±5% band
// with the stack of points down its middle.
const FLAT_PAD = 0.02

// Multiplicative padding, because the axis is logarithmic: a constant margin would be
// huge at the low end and invisible at the high end. Keeps the outermost points off the
// frame while letting the axis fit whatever range the runs actually cover.
//
// **The axis fits the data, so it says nothing about how big a difference is.** A tight
// cluster is magnified until it fills the plot exactly like a spread-out one — which is
// what makes the shape readable, and why the tick labels have to carry enough precision
// to show that the ends are nearly the same number (see `compactAxisFormat`).
export function logDomain(values: number[]): [number, number] {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const decades = Math.log10(max / min)
  const pad = decades > 0 ? decades * PAD_SHARE : FLAT_PAD
  return [min / 10 ** pad, max * 10 ** pad]
}

// Padding when every run scored the same, in score points. The y counterpart of
// `FLAT_PAD`: a zero-height domain is not an axis, and half a point either side puts
// the flat line in the middle of a band fine enough to show it really is flat.
const FLAT_SCORE_PAD = 0.5

// The range a score can occupy at all. The domain is clamped to it, so a run at 99.6
// does not pad the axis into territory no model can reach.
const SCORE_FLOOR = 0
const SCORE_CEILING = 100

/**
 * The y domain, fitted to the scores on the plot.
 *
 * It used to be a hard `[0, 100]`, which is the *scale's* range and not the data's:
 * these four language runs sit between 57.35 and 59.36, so they had two points of a
 * hundred to move in and drew as one flat row of dots. Every real difference — a
 * fine-tune gaining a point, a sweep's high tier dropping two — was compressed into
 * 2% of the plot height and read as no difference at all.
 *
 * Padded by a share of the data's own span, exactly as `logDomain` does on x, then
 * snapped outward to a round step so the top and bottom gridlines are labelled rather
 * than falling on 57.11 and 59.60.
 *
 * **The same caveat applies as on x, and harder.** A fitted score axis says nothing
 * about whether a gap is big: 0.8 points of markup spread will fill the plot exactly
 * like 40 would. The panels say so in their footnotes, and the scores carry their own
 * standard error — which is the number that actually decides whether a gap is real.
 */
export function scoreDomain(values: number[]): [number, number] {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min
  const pad = span > 0 ? span * PAD_SHARE : FLAT_SCORE_PAD

  const step = niceStep(
    Math.max((span + pad * 2) / TARGET_TICKS, Number.EPSILON)
  )
  const lo = Math.floor((min - pad) / step) * step
  const hi = Math.ceil((max + pad) / step) * step
  return [Math.max(lo, SCORE_FLOOR), Math.min(hi, SCORE_CEILING)]
}

/**
 * Tick labels carrying exactly the decimals the step needs.
 *
 * A step of 0.5 has to print "57.0" beside "57.5" — recharts would render the first
 * as "57", and an axis whose labels change width row to row reads as sloppy rather
 * than as precise.
 */
export function scoreTickFormat(ticks: number[]): (value: number) => string {
  const digits = exactDecimals(ticks, 2)
  return (value: number) => value.toFixed(digits)
}

// Rungs to try per decade, sparsest first. On a log axis the number of round numbers
// inside the range depends entirely on how many decades it spans: the 1/2/5 ladder
// that gives nine sensible labels across three decades gives only two across a range
// where every model is within a factor of two of the others. So the ladder adapts
// instead of being fixed.
const LADDERS = [[1], [1, 2, 5], [1, 2, 3, 5, 7], [1, 2, 3, 4, 5, 6, 7, 8, 9]]

// Above this many labels the axis is a wall of text and they start to collide.
const MAX_TICKS = 9

// Below this many decades the mantissa ladders stop working: a domain from 308K to
// 309K contains no power of ten and no 1/2/5 rung, so even the finest ladder returns at
// most one value and the axis falls back to its bare ends. A range this narrow is
// visually linear anyway — the log curve over 0.03 decades is a straight line — so it
// gets linear-style round steps instead.
const LADDER_FLOOR_DECADES = 0.5

// Round-number ticks inside a log domain: the densest ladder that still fits.
//
// Every tick is a genuine round number at its own magnitude (…400K, 500K, 600K… or
// …1M, 2M, 5M…), never an interpolated value, so a label means the same thing wherever
// it lands on the axis. Under half a decade that means a round *step* (…308.45K,
// 308.50K, 308.55K…) rather than a round mantissa, which is the same promise kept at a
// scale where no mantissa lands inside the range.
export function logTicks(domain: [number, number]): number[] {
  const [min, max] = domain
  if (Math.log10(max / min) < LADDER_FLOOR_DECADES) {
    const stepped = steppedTicks(min, max)
    if (stepped.length >= 2) return stepped
  }
  return pickDensest(
    LADDERS.map((ladder) => rungs(ladder, min, max)),
    domain
  )
}

// Steps worth landing on, as multiples of a power of ten. The same ladder every linear
// axis uses, for the same reason: these are the intervals people count in.
const NICE_STEPS = [1, 2, 2.5, 5, 10]

// Ticks a narrow domain can actually carry: the roundest step that yields about this
// many of them.
const TARGET_TICKS = 5

// The roundest step at least as large as `rough`. Shared by the x ticks and the y
// axis, which need the same ladder for the same reason — one of them just happens to
// be riding a log scale so narrow it has gone straight.
function niceStep(rough: number): number {
  const magnitude = 10 ** Math.floor(Math.log10(rough))
  return (NICE_STEPS.find((s) => s * magnitude >= rough) ?? 10) * magnitude
}

export function steppedTicks(min: number, max: number): number[] {
  // A non-positive span gives NaN here, which fails the guard — an empty tick list,
  // which the axis treats as "choose your own".
  const chosen = niceStep((max - min) / TARGET_TICKS)
  if (!(chosen > 0)) return []

  const out: number[] = []
  // Stepping by repeated addition accumulates float error into the labels, so each
  // tick is computed as a multiple of the step instead — which is also what makes it
  // an exactly round number rather than one that prints as 308449.99999.
  for (let i = Math.ceil(min / chosen); i * chosen <= max; i++) {
    out.push(i * chosen)
    if (out.length > MAX_TICKS) return []
  }
  return out
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
export function pickDensest(
  tiers: number[][],
  domain: [number, number]
): number[] {
  let best: number[] = []
  for (const tier of tiers) {
    if (tier.length > MAX_TICKS) break
    best = tier
  }
  if (best.length >= 2) return best
  // A domain so narrow that not even the finest tier has two values in it — four runs
  // inside one second, say — has no round *value* to label, so it gets a round *step*
  // instead. This is the common case now that the domain fits the data rather than
  // padding out to the nearest decade, and the tiers a measure supplies can only be as
  // fine as someone thought to write down.
  const stepped = steppedTicks(domain[0], domain[1])
  // Last resort, so the axis is never left blank.
  return stepped.length >= 2 ? stepped : domain.map(Math.round)
}

function rungs(ladder: number[], min: number, max: number): number[] {
  const out: number[] = []
  for (
    let exponent = Math.floor(Math.log10(min));
    exponent <= Math.ceil(Math.log10(max));
    exponent++
  ) {
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
    <div className="min-w-40 rounded-(--radius-sheet) border border-border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-overlay">
      <div className="eyebrow mb-1 text-muted-foreground">
        {formatDate(point.created_at)}
      </div>
      <div className="flex items-baseline justify-between gap-6">
        {/* The model, never `name`: on a sweep point that is a budget, and a tooltip
            reading "Medium — 57.4" does not say whose. */}
        <span className="truncate">{point.model}</span>
        <span className="font-medium tabular-nums">
          {formatNumber(point.score, 1)}
        </span>
      </div>
      <div className="mt-1 text-xs text-muted-foreground tabular-nums">
        {formatX(point.x)} {xLabel}
        {point.budget && ` · ${TIER_LABEL[point.budget].toLowerCase()} budget`}
      </div>
      {/* Only when it is thin. A conversation count on every tooltip is noise; on a
          point drawn hollow it is the answer to the question the hollow raised. */}
      {point.provisional && (
        <div className="mt-1 text-xs text-muted-foreground">
          {point.conversations ?? 0} held-out conversation
          {point.conversations === 1 ? "" : "s"} — too few to read as a
          difference.
        </div>
      )}
    </div>
  )
}
