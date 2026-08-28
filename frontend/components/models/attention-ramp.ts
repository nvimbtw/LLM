// The colour of one attention cell.
//
// A per-cell weight is a *magnitude*, so it takes a single-hue sequential ramp —
// the rule `components/training/chart-theme.ts` states for every by-index series.
// The rule is borrowed; the ramp is not, and that is deliberate rather than a
// second palette. A line at t=0 still has to be visible against the surface,
// while a cell at t=0 has to *vanish* into it, so the bottom of this ramp is
// inverted relative to every ramp in that file. Chart theme owns the charts;
// this owns one figure that is not one.
//
// It is expressed as one hue at varying alpha rather than as a lightness sweep,
// which is what makes it theme-aware for nothing: the panel's own surface shows
// through, so the figure reads light→dark on paper and dark→light on the dragon
// ground — "more ink is more attention" in both — with no CSS token to sample at
// draw time and no dark-mode second ladder to keep in step.
//
// The hue is chart-theme's per-head violet, `hsl(280 40% 51%)`, resolved once.
// Attention entropy in the diagnostics panel is per head and is drawn in that
// family, and a reader arriving from it should recognise what they are looking
// at.
const HUE = { r: 147, g: 80, b: 180 } as const

// Attention rows sum to 1 across up to 512 keys, so a *linear* ramp puts almost
// the whole matrix inside the bottom one percent of the scale and draws a
// diffuse head as an empty square. The gamma lifts the floor without touching
// the ordering; it is a display transform, and the exact weight is always one
// hover away in the readout beside the figure.
const GAMMA = 0.4

// One entry per byte of alpha. Building the ramp per cell would mean 262,144
// `Math.pow` calls a draw at the 512-token ceiling; building it once means 256.
const RAMP_STEPS = 256
const MAX_ALPHA = 255

/**
 * `RAMP_STEPS × 4` bytes of RGBA, indexed by `weight * (RAMP_STEPS - 1)`.
 *
 * Cached at module scope: it is a pure function of two constants, and every
 * matrix drawn in a session wants the same one.
 */
export const attentionRamp: Uint8ClampedArray = buildRamp()

function buildRamp(): Uint8ClampedArray {
  const ramp = new Uint8ClampedArray(RAMP_STEPS * 4)
  for (let step = 0; step < RAMP_STEPS; step += 1) {
    const at = step * 4
    ramp[at] = HUE.r
    ramp[at + 1] = HUE.g
    ramp[at + 2] = HUE.b
    ramp[at + 3] = Math.round(
      Math.pow(step / (RAMP_STEPS - 1), GAMMA) * MAX_ALPHA
    )
  }
  return ramp
}

/** The index into `attentionRamp` for a weight in [0, 1]. */
export function rampIndex(weight: number): number {
  const clamped = weight < 0 ? 0 : weight > 1 ? 1 : weight
  return Math.round(clamped * (RAMP_STEPS - 1)) * 4
}
