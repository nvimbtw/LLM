// Single source of truth for the training charts' visual language, so the loss,
// batch-spread, and vocab figures read as one family. Two ideas drive it:
//
//   1. Recessive axes/grid. Hairline horizontal grid only (no vertical rules),
//      no axis lines or tick lines, muted tick labels — the data carries the ink,
//      the frame recedes. Matches the editorial theme.
//   2. Magnitude, not identity. The loss-per-epoch overlay and the min/avg/max
//      spread are both *magnitude* series (a progression / an ordered band), not
//      independent categories — so each is coloured with a single-hue sequential
//      ramp (light → dark), never a rainbow or a categorical cycle. This follows
//      the dataviz "sequential = one hue, light→dark" rule.

// Shared axis/grid props. Spread onto the recharts primitives so every figure is
// configured identically (`<XAxis {...axisProps} dataKey=… />`).
export const axisProps = {
  tickLine: false,
  axisLine: false,
  tickMargin: 8,
  minTickGap: 32,
} as const

export const yAxisProps = {
  tickLine: false,
  axisLine: false,
  width: 44,
  tickMargin: 4,
} as const

// Horizontal-only hairline grid. `vertical={false}` keeps the plot calm; the
// stroke inherits the CSS `--border` token via the ChartContainer's styling.
export const gridProps = {
  vertical: false,
} as const

// A single-hue sequential ramp: hold hue + saturation, sweep lightness from light
// (t=0) to dark (t=1), so a higher `t` reads as "further along / larger". Monotone
// in lightness by construction, which is what makes a sequential ramp legible.
function sequential(hue: number, sat: number, t: number): string {
  const clamped = Math.min(1, Math.max(0, t))
  const light = 68 - clamped * 34 // 68% (light) → 34% (dark)
  return `hsl(${hue} ${sat}% ${light.toFixed(0)}%)`
}

// Loss-per-epoch overlay: one line per epoch, early epochs light → late epochs
// dark, so the run's progression reads as a deepening of a single hue rather than
// a hue change. Blue family, aligned with the site's `--chart-3` (#4d699b).
export function lossEpochColor(epoch: number, totalEpochs: number): string {
  const t = totalEpochs > 1 ? (epoch - 1) / (totalEpochs - 1) : 0
  return sequential(220, 34, t)
}

// Loss overlay for a CURRICULUM run: each stage trains on a different dataset, so
// stages ARE distinct categories (the one place identity, not magnitude, matters)
// and get distinct hues. Within a stage the same light→dark sweep applies across
// its epochs, so a stage still reads as a deepening progression, just in its own
// colour. Hues are spaced around the wheel and kept clear of the amber batch band;
// `sat` is tuned per hue so they read at even strength. Cycles past its length.
const STAGE_HUES: [number, number][] = [
  [220, 34], // blue (the classic loss hue — first stage keeps the familiar look)
  [150, 42], // green
  [280, 40], // violet
  [338, 46], // rose
  [190, 44], // teal
  [28, 60], // warm ochre (kept off the batch amber at 41)
]

// `stage` selects the hue; `tWithinStage` (0 = first epoch of the stage → 1 = last)
// sweeps lightness within it.
export function lossStageColor(stage: number, tWithinStage: number): string {
  const [hue, sat] =
    STAGE_HUES[
      ((stage % STAGE_HUES.length) + STAGE_HUES.length) % STAGE_HUES.length
    ]
  return sequential(hue, sat, tWithinStage)
}

// Intra-batch spread band. min ≤ avg ≤ max is an ordered magnitude, so it's the
// same single-hue ramp (amber family, `--chart-1` #de9800): min light, max dark.
export const batchRamp = {
  min: sequential(41, 70, 0),
  avg: sequential(41, 70, 0.5),
  max: sequential(41, 70, 1),
} as const

// The vocab-growth line is a lone series — the site's curated `--chart-2` token
// (referenced as a CSS var so it tracks the theme in light/dark).
export const VOCAB_COLOR = "var(--chart-2)"

// ── Diagnostics figures ─────────────────────────────────────────────────────
// The per-layer diagnostics (grad norms, update:weight ratios, activation RMS)
// are the same *magnitude-by-depth* idea as the loss overlay, so they reuse the
// single-hue sequential ramp — but in a teal family, distinct from the loss
// overlay's blue, so a per-layer diagnostic never reads as "loss". A given layer
// keeps one colour across all three figures.
export function layerColor(layer: number, totalLayers: number): string {
  const t = totalLayers > 1 ? layer / (totalLayers - 1) : 0
  return sequential(168, 44, t)
}

// Attention entropy is per *head*; one hue (violet) swept light→dark by head
// index, again a sequential ramp rather than a categorical cycle.
export function headColor(head: number, totalHeads: number): string {
  const t = totalHeads > 1 ? head / (totalHeads - 1) : 0
  return sequential(280, 40, t)
}

// Accent for an aggregate line drawn over a per-layer ramp (the global grad norm)
// or a reference marker — the site's curated amber `--chart-1`.
export const ACCENT_COLOR = "var(--chart-1)"

// A recessive hue for the two shared tables (token-embedding / output-projection
// grad) plotted alongside the per-layer ramp, kept muted so the layers lead.
export const MUTED_SERIES = "var(--muted-foreground)"

// Embedding-anisotropy figure: one blue hue (`--chart-3`) for the mean line and
// its p5–p95 band fill.
export const EMB_COLOR = "var(--chart-3)"

// ── Preference catalogue figures ────────────────────────────────────────────
// Approved vs rejected is the one *polarity* pair in the app: two opposed classes,
// not a progression and not arbitrary categories. Two rules govern it.
//
//   1. **Not red/green.** The obvious good/bad pair is the classic colourblind
//      failure — the site's own olive `--chart-2` against rose `--chart-4` measures
//      ΔE 5.8 under deuteranopia, well under the 8 floor, i.e. one in twelve men
//      would read the two series as one. Blue/amber is the canonical CVD-safe
//      opposition and measures ΔE 29.4 (protan) / 30.1 (tritan) instead.
//   2. **Dark mode is selected, not flipped.** The dark band is tighter
//      (OKLCH L 0.48–0.67 vs 0.43–0.77), so the light steps sit outside it and
//      lightening them by rule would fail the chroma floor. These are their own
//      steps, validated against the dark surface #282727.
//
// Both pairs pass all six checks of the dataviz validator (lightness band, chroma
// floor, CVD separation, normal-vision floor, contrast vs surface). Re-run it
// before changing either value:
//
//   node scripts/validate_palette.js "#3f6bb8,#de9800" --mode light --surface "#f7f4e4"
//   node scripts/validate_palette.js "#6d92da,#b98a2c" --mode dark  --surface "#282727"
//
// Colour is never the only cue regardless: both figures carry a legend, and the
// feed states the verdict in words.
//
// Referenced as CSS vars rather than literals so they track the theme: dark mode
// here is a `.dark` class, not `color-scheme`, so `light-dark()` would never
// switch. The values live in app/globals.css beside the other chart tokens.
export const verdictColors = {
  approved: "var(--chart-approved)",
  rejected: "var(--chart-rejected)",
} as const

// ── Teacher figures ─────────────────────────────────────────────────────────
// The three thinking budgets are an *ordered* scale — short, medium, long — not
// three independent categories, so they take a single-hue sequential ramp
// (light→dark in light mode, dark→light on a dark surface) rather than a
// categorical cycle. More thinking, more ink, in both modes.
//
// The hue is the teach kind's terracotta, so a teaching run's figures read as
// belonging to its chip. Values and their validation live in app/globals.css.
export const tierColors: Record<"low" | "med" | "high", string> = {
  low: "var(--tier-low)",
  med: "var(--tier-med)",
  high: "var(--tier-high)",
}

// Rank ramp for the PCA biplot's token arrows: the same blue family swept by
// frequency rank, most-frequent (rank 0) darkest → least-frequent lightest, so the
// eye can tell the salient tokens from the tail.
export function embRankColor(rank: number, total: number): string {
  const t = total > 1 ? 1 - rank / (total - 1) : 1
  return sequential(212, 48, t)
}
