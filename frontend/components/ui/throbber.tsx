"use client"

import * as React from "react"

import { cn } from "@/lib/utils"

/**
 * The rings — one loading indicator for the whole site.
 *
 * Five irregular closed loops nested inside one another, like rubber bands dropped
 * on the floor and traced. Flat, drawn in line only. States differ in how the
 * figure moves and in what the colour does — nothing, a slow drift, a two-ring scan
 * travelling outward, the whole figure breathing, every ring flashing at once.
 *
 * This is the fourth design. The one before it was a rotating faceted ball, and the
 * two before that were a pattern with a period — three ticks whose heights cycled,
 * then an arc travelling a ring. The period was the problem: however carefully the
 * seam is made continuous, the eye tracks the motion, notices it arrive back where
 * it began, and reads that as the animation cutting off and restarting. The ball
 * solved it by turning, since a body at a new orientation every frame has nothing to
 * restart. These rings solve it a different way, and the difference is the whole
 * design:
 *
 *  1. **The rings are contours of one shared field, not five independent shapes.**
 *     Every loop samples the same sum of harmonics; a ring differs from its
 *     neighbour only by a small phase lag proportional to its radius. That is what
 *     makes the figure hold together as one object when it moves — the rings bulge
 *     and slacken *in sympathy*, never independently, and because the lag is small
 *     relative to their spacing they can never cross. Turn the lag up and it stops
 *     being a figure and becomes five squiggles.
 *  2. **The field is aperiodic.** Its harmonics advance at frequencies with no
 *     common multiple, so the figure is a shape it has never held before on every
 *     frame it will ever be drawn. There is no loop point to notice.
 *  3. **Phases are integrated, never computed from the clock.** Each frame adds
 *     `rate × dt` to a running phase. A mode change therefore changes the *speed*
 *     the figure deforms at, and its current shape carries straight through — a
 *     state change cannot move the figure, only change how it is moving. `mode`
 *     swaps crossfade the rates, so it accelerates into its new behaviour.
 *  4. **Every state is the same figure**, differing only in a vector of numbers
 *     (`Look`). Because a state is just numbers, any two mix by weight. Give one
 *     instance a changing `mode` rather than swapping instances.
 *  5. **Idle is where the field stops.** Not a special case — `flow` simply goes to
 *     near zero, so the shape is held rather than frozen, and the crossfade out of
 *     it starts from exactly the shape the eye was already looking at.
 *  6. **States are told apart by motion first, hue second.** Two states spend the
 *     second colour — orange — because for them the message is about *where* in the
 *     figure something is happening, which no amount of movement says: `thinking`
 *     runs a two-ring-wide scan from the centre outward, and `tool` flashes every
 *     ring at once. In both, the accent underneath dims as the spark comes up, so
 *     the ring changes colour rather than gaining a second one.
 */
export type ThrobberMode =
  | "idle"
  | "loading"
  | "thinking"
  | "generating"
  | "tool"
  | "training"
  | "benchmarking"

export interface ThrobberProps {
  mode?: ThrobberMode
  className?: string
  /**
   * Colour of the rings. Override where the mark already sits on a primary surface
   * (e.g. the send button) so it doesn't vanish.
   */
  accent?: string
  /**
   * The second colour — what a scanned or flashing ring turns into. Its only job is
   * to be unmistakable for `accent` at a hairline width, which rules out anything
   * that differs from it mainly in saturation: the default is the burnt-orange
   * `--throbber-spark`, opposite the blue-grey primary on the wheel.
   */
  spark?: string
}

const TAU = Math.PI * 2

// Geometry, in viewBox units. `RADIUS` is the outermost ring's *base* radius; the
// field pushes it out by up to a fifth of that again, so the figure leaves room
// around itself rather than filling the box.
const VIEW = 16
const CENTER = VIEW / 2
// The outermost ring's *base* radius. Well short of the box's half-width of 8,
// because the field and the breath both push it out: at the largest `warp` and
// `breathe` any state asks for, the figure reaches about 7.6 and still has room for
// its own line weight.
const RADIUS = 5.2

// How long a mode change takes to cross-fade. Long enough to read as the figure
// changing pace, short enough that it never delays feedback.
const BLEND_MS = 420

// A frame long enough that it was probably a background tab rather than a slow
// frame. Clamped so returning to the tab doesn't teleport the figure through a
// minute's worth of deformation.
const MAX_FRAME_S = 1 / 15

/**
 * How big the mark is, in CSS pixels — everywhere, without exception.
 *
 * The mark used to be sized by its caller, which meant a different figure in every
 * row it appeared in, and no two of them looked like the same object. One number
 * instead: callers position it but never size it. Change this and the whole app
 * changes with it.
 *
 * 64px is where two constraints meet. It is small enough to sit in the chat
 * thread's status row beside a 14px label without the row becoming a panel, and
 * large enough that five nested rings still separate — at this size they sit about
 * 3.5px apart, which is the floor for reading them as five loops rather than as a
 * smudge.
 */
export const THROBBER_PX = 64

// How many rings, and where their base radii sit as a fraction of `RADIUS`. Five is
// the count the shape was designed at: enough for the scan to travel and read as
// travelling, few enough that they stay apart at `THROBBER_PX`.
//
// The spacing is the budget every moving state spends out of. How far a ring may
// deviate before it collides with its neighbour is set by the gap between them, so
// these are spread as wide as the innermost ring can stand — pull the inner one in
// any further and it stops being a loop and becomes a dot.
const RING_RADII = [0.34, 0.5, 0.66, 0.83, 1]
const RINGS = RING_RADII.length

// Points sampled per ring. The curve is smoothed through them, and the fastest
// harmonic in the field is the 7th, so this is roughly eight samples per wave of
// the finest detail there is — past which more points buy nothing but string.
const SAMPLES = 60

/**
 * The interpolable state vector. Every field is a plain number precisely so that
 * any two modes can be mixed by weight — and the rates are *rates*, which is what
 * lets a mode change alter the motion without moving the figure.
 */
interface Look {
  /** Turns per second of the whole figure about its centre. */
  spin: number
  /** How fast the shape field advances, in cycles per second. 0 = the shape is held. */
  flow: number
  /** Multiplier on the field's amplitude — how far from round the rings are. */
  warp: number
  /** Uniform in-and-out of every ring together, as a fraction of the radius. */
  breathe: number
  /** How fast that breath cycles, in cycles per second. */
  breatheRate: number
  /** A swell travelling outward through the rings, as a fraction of the radius. */
  ripple: number
  /** How fast it travels, in cycles per second. */
  rippleRate: number
  /** Strength of the colour scan running outward through the rings, 0 = none. */
  scan: number
  /** How fast the scan travels, in sweeps per second. */
  scanRate: number
  /** How many rings it covers at once — 2 for a band, ~0.5 for one ring at a time. */
  scanWidth: number
  /** Strength of the all-at-once colour flash, 0 = none. */
  flash: number
  /** How often it flashes, in flashes per second. */
  flashRate: number
  /** Overall opacity of the figure. */
  o: number
  /** Multiplier on the line weight. */
  weight: number
}

const LABELS: Record<ThrobberMode, string> = {
  idle: "Idle",
  loading: "Loading",
  thinking: "Thinking",
  generating: "Generating",
  tool: "Running a tool",
  training: "Training",
  benchmarking: "Benchmarking",
}

const REDUCED_MOTION = "(prefers-reduced-motion: reduce)"

// Live-updating, SSR-safe read of the OS motion preference. A plain
// `matchMedia(...).matches` read is a snapshot: it never notices the setting
// changing, and on the server there's no `window` at all.
function usePrefersReducedMotion(): boolean {
  return React.useSyncExternalStore(
    (onChange) => {
      const query = window.matchMedia(REDUCED_MOTION)
      query.addEventListener("change", onChange)
      return () => query.removeEventListener("change", onChange)
    },
    () => window.matchMedia(REDUCED_MOTION).matches,
    () => false
  )
}

export function Throbber({
  mode = "idle",
  className,
  accent = "var(--primary)",
  spark = "var(--throbber-spark)",
}: ThrobberProps) {
  const reduced = usePrefersReducedMotion()
  const ringsRef = React.useRef<(SVGPathElement | null)[]>([])
  const sparksRef = React.useRef<(SVGPathElement | null)[]>([])
  // Mode crossfade. Held in refs so a transition never triggers a React render —
  // the whole animation runs outside the reconciler.
  const currentMode = React.useRef(mode)
  const previousMode = React.useRef(mode)
  const blendStart = React.useRef(0)

  // Bookkeeping happens in an effect rather than during render: a render can be
  // discarded or replayed under concurrent React, which would corrupt which mode we
  // think we're coming *from*. The rAF loop is the only reader, so starting the
  // crossfade a frame later than the render is imperceptible.
  React.useEffect(() => {
    if (currentMode.current === mode) return
    previousMode.current = currentMode.current
    currentMode.current = mode
    blendStart.current = performance.now()
  }, [mode])

  // The figure's running phases. These persist across mode changes and across the
  // effect being torn down and rebuilt, which is the point: they are the state of a
  // shape that has been drifting since the page loaded, not the progress of an
  // animation that plays.
  const turn = React.useRef(0)
  const flow = React.useRef(0)
  const breath = React.useRef(0)
  const ripple = React.useRef(0)
  const scan = React.useRef(0)
  const flash = React.useRef(0)
  const lastFrame = React.useRef(0)
  const staggered = React.useRef(false)

  React.useEffect(() => {
    // Each instance starts at its own point in the field, so the several marks a
    // page can show at once (the sidebar, the queue's now-band, a job header) aren't
    // the same shape cloned. Claimed here rather than at render because taking a
    // number from a shared counter is a side effect, and doing it during render
    // would also make the first frame differ between server and client for no
    // benefit — nothing about the stagger reaches the markup.
    if (!staggered.current) {
      staggered.current = true
      const offset = nextStagger()
      turn.current = offset
      flow.current = offset * 40
    }

    let frame = 0

    const tick = (now: number) => {
      const dt = Math.min(
        Math.max((now - (lastFrame.current || now)) / 1000, 0),
        MAX_FRAME_S
      )
      lastFrame.current = now

      const blend = Math.min((now - blendStart.current) / BLEND_MS, 1)
      let look =
        blend >= 1
          ? LOOKS[currentMode.current]
          : mix(
              LOOKS[previousMode.current],
              LOOKS[currentMode.current],
              smoothstep(0, 1, blend)
            )
      // Reduced motion damps the rates and the relief rather than freezing them.
      // Stopping was the obvious reading of the preference and the wrong one: this
      // is a status indicator, and a motionless one reports nothing. The guidance
      // targets large, vestibular-triggering movement, so a slow drift honours the
      // intent while the mark still indicates.
      if (reduced) look = mix(look, LOOKS.idle, 0.65)

      turn.current += look.spin * dt
      flow.current += look.flow * dt
      breath.current += look.breatheRate * dt
      ripple.current += look.rippleRate * dt
      scan.current += look.scanRate * dt
      flash.current += look.flashRate * dt

      draw(
        look,
        turn.current,
        flow.current,
        breath.current,
        ripple.current,
        scan.current,
        flash.current,
        ringsRef.current,
        sparksRef.current
      )

      frame = requestAnimationFrame(tick)
    }

    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [reduced])

  return (
    <span
      role="status"
      aria-label={LABELS[mode]}
      // The size is the component's, not the caller's — see `THROBBER_PX`. An
      // inline style rather than a utility class so the one constant is the only
      // place it is written down. `className` is still honoured for everything
      // else a caller legitimately controls: margins, placement, colour.
      style={{ width: THROBBER_PX, height: THROBBER_PX }}
      className={cn(
        "inline-flex shrink-0 items-center justify-center",
        className
      )}
    >
      <svg
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        className="block h-full w-auto overflow-visible"
        aria-hidden
      >
        {RING_RADII.map((_, i) => (
          <path
            key={i}
            ref={(node) => {
              ringsRef.current[i] = node
            }}
            fill="none"
            stroke={accent}
            strokeWidth={1.1}
            strokeOpacity={0}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        ))}
        {/* The second colour, as its own layer over the top rather than a recolouring
            of the strokes below. A line that is *both* structure and signal would
            have to switch hue mid-fade, and the switch is visible at the size this is
            drawn at; laying one over the other and trading their opacities is a clean
            crossfade instead. Slightly heavier, so it covers the accent under it
            completely at full strength. */}
        {RING_RADII.map((_, i) => (
          <path
            key={i}
            ref={(node) => {
              sparksRef.current[i] = node
            }}
            fill="none"
            stroke={spark}
            strokeWidth={1.35}
            strokeOpacity={0}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
    </span>
  )
}

// Successive starting offsets into the field, in turns. Stepping by the golden
// ratio spreads any number of instances about as evenly as they can be spread,
// without needing to know in advance how many there will be.
let stagger = 0
function nextStagger(): number {
  stagger = fract(stagger + 0.6180339887)
  return stagger
}

// ── the field ───────────────────────────────────────────────────────────────

/**
 * The harmonics every ring is a contour of.
 *
 *  - `m` is how many lobes that term puts around a ring. `m: 1` is an off-centre
 *    push, which is what stops the rings from being concentric; the rest are the
 *    slack and the bulges.
 *  - `amp` is its share of the deviation, as a fraction of `RADIUS`. They fall off
 *    with `m` because a real dropped band has long lazy curves and only a little
 *    fine detail, and because five nested loops with sharp detail read as noise.
 *  - `lag` is turns of phase per unit radius: how much a ring's shape differs from
 *    the ring inside it. **This number is the design, and it is the one that decides
 *    how much the figure is allowed to move at all.** Near zero and the rings are
 *    five copies of one outline; too large and they stop being one figure, and then
 *    they cross. Since only the *difference* between neighbours can make them
 *    collide, keeping the lags low is what buys the amplitude to deform visibly —
 *    the rings move further precisely because they move more nearly in parallel.
 *  - `freq` is how fast it advances relative to `flow`. Chosen with no common
 *    multiple, so the sum never returns to a shape it has held before. They are kept
 *    within about 4× of each other on purpose: at the fast end of `flow`, a harmonic
 *    several times quicker than the rest stops reading as the shape changing and
 *    starts reading as the line buzzing.
 *
 * The lags and amplitudes together set the ceiling on `Look.warp`. Two neighbouring
 * rings sit 0.165 apart and differ by at most `Σ 2·amp·sin(π·lag·0.165)` ≈ 0.049 of
 * the radius, plus another ~0.029 from the `ROUGHNESS` taper between them — so about
 * 0.078 per unit of `warp`, and a `warp` past roughly 2.1 lets the worst case (every
 * harmonic aligned at once) put one ring through another. The states stop at 1.45,
 * which a 400-second sweep of all seven and their pairwise crossfades puts at a
 * worst-case gap of 0.27 units — about a pixel at `THROBBER_PX`, so the closest two
 * rings ever come is grazing. Nothing enforces the ceiling; if a state wants to look
 * busier, take it out of `flow`, which is where the perceived motion lives anyway.
 */
const HARMONICS = [
  { m: 1, amp: 0.05, lag: 0.2, freq: 0.31 },
  { m: 2, amp: 0.085, lag: 0.12, freq: 0.53 },
  { m: 3, amp: 0.055, lag: -0.18, freq: 0.71 },
  { m: 4, amp: 0.03, lag: 0.22, freq: 0.89 },
  { m: 5, amp: 0.022, lag: 0.3, freq: 1.13 },
  { m: 7, amp: 0.012, lag: -0.35, freq: 1.37 },
]

// Sample directions, and each harmonic's shape around the ring. All of it is fixed,
// so it is computed once at module load and every frame is multiply-and-add: the
// only trigonometry left per frame is one pair per harmonic per ring.
const DIR_X = new Float64Array(SAMPLES)
const DIR_Y = new Float64Array(SAMPLES)
const HARMONIC_SIN: Float64Array[] = HARMONICS.map(
  () => new Float64Array(SAMPLES)
)
const HARMONIC_COS: Float64Array[] = HARMONICS.map(
  () => new Float64Array(SAMPLES)
)

for (let j = 0; j < SAMPLES; j += 1) {
  const theta = (j / SAMPLES) * TAU
  DIR_X[j] = Math.cos(theta)
  DIR_Y[j] = Math.sin(theta)
  for (let h = 0; h < HARMONICS.length; h += 1) {
    HARMONIC_SIN[h][j] = Math.sin(HARMONICS[h].m * theta)
    HARMONIC_COS[h][j] = Math.cos(HARMONICS[h].m * theta)
  }
}

// How strongly a ring feels the field, by radius. Flat would make the inner rings
// proportionally far more crumpled than the outer ones — the deviation is in
// absolute units, and a 0.34-radius ring is small. Tapering it keeps the innermost
// loop recognisably a loop while leaving the outer ones as slack as they should be,
// and it is steep enough that the inner ring survives the largest `warp` any state
// asks for without pinching shut.
const ROUGHNESS = RING_RADII.map((r) => 0.3 + 0.7 * r)

// How far the travelling swell of `ripple` is spread across the rings, in cycles
// over the full radius. Just under one, so at any moment the wave is somewhere in
// the figure rather than everywhere in it.
const RIPPLE_SPAN = 0.85

// Inner rings are drawn a touch stronger than outer ones. Nothing here is lit or
// shaded — it is flat line work — but a uniform stack of five reads as a target,
// and the falloff gives it a centre.
const RING_ALPHA = RING_RADII.map((_, i) => 0.72 + 0.28 * (1 - i / (RINGS - 1)))

// Scratch for one ring's projected points, reused across rings and instances.
// Single-threaded and consumed before the next ring starts, so sharing it is safe
// and saves allocating 60 pairs per ring per frame.
const POINT_X = new Float64Array(SAMPLES)
const POINT_Y = new Float64Array(SAMPLES)

/**
 * Writes one frame straight to the DOM. Deliberately imperative: this runs at 60fps
 * while the chat is streaming, and re-rendering React for it would make the throbber
 * the most expensive thing on the page.
 */
function draw(
  look: Look,
  turn: number,
  flow: number,
  breath: number,
  ripplePhase: number,
  scanPhase: number,
  flashPhase: number,
  rings: (SVGPathElement | null)[],
  sparks: (SVGPathElement | null)[]
) {
  const angle = turn * TAU
  const ca = Math.cos(angle)
  const sa = Math.sin(angle)
  const breathe = 1 + look.breathe * Math.sin(TAU * breath)

  // Where the scan is, in ring indices. It runs from just outside the centre to
  // just past the outermost ring, so a sweep begins and ends off the figure rather
  // than appearing on top of a ring and vanishing off another.
  const width = Math.max(look.scanWidth, 0.001)
  const scanAt = fract(scanPhase) * (RINGS - 1 + 2 * width) - width
  // A flash is sharp at its peak and slow away from it, which is what makes it read
  // as an event rather than a fade in and out. Cubing a raised cosine gets that and
  // stays continuous where the cycle wraps.
  const flashAt =
    look.flash > 0
      ? look.flash * Math.pow(0.5 + 0.5 * Math.cos(TAU * fract(flashPhase)), 3)
      : 0

  for (let i = 0; i < RINGS; i += 1) {
    const node = rings[i]
    const sparkNode = sparks[i]
    if (!node) continue

    const base = RING_RADII[i]
    const amplitude = look.warp * ROUGHNESS[i]
    const scale =
      breathe *
      (1 + look.ripple * Math.sin(TAU * (ripplePhase - base * RIPPLE_SPAN)))

    // The whole of the per-frame trigonometry: one phase per harmonic for this
    // ring. Everything inside the sample loop below is multiply-and-add against the
    // tables built at module load.
    for (let h = 0; h < HARMONICS.length; h += 1) {
      const harmonic = HARMONICS[h]
      const phase = TAU * (flow * harmonic.freq + base * harmonic.lag)
      const cp = Math.cos(phase) * harmonic.amp * amplitude
      const sp = Math.sin(phase) * harmonic.amp * amplitude
      // sin(mθ + phase) expanded, so the sample loop never calls a trig function.
      for (let j = 0; j < SAMPLES; j += 1) {
        const term = HARMONIC_SIN[h][j] * cp + HARMONIC_COS[h][j] * sp
        // First harmonic seeds the accumulator; the rest add to it.
        POINT_X[j] = h === 0 ? term : POINT_X[j] + term
      }
    }

    for (let j = 0; j < SAMPLES; j += 1) {
      const r = (base + POINT_X[j]) * scale * RADIUS
      const x = r * DIR_X[j]
      const y = r * DIR_Y[j]
      POINT_X[j] = CENTER + x * ca - y * sa
      POINT_Y[j] = CENTER + x * sa + y * ca
    }

    const d = closedCurve(POINT_X, POINT_Y)
    node.setAttribute("d", d)

    // How much of the second colour this ring is showing: a band passing over it,
    // or every ring flashing at once. They are the same quantity, so a state can
    // hold both and the crossfade between such states is just a fade.
    const lit = Math.min(
      1,
      (look.scan > 0
        ? look.scan * smoothstep(width, 0, Math.abs(i - scanAt))
        : 0) + flashAt
    )

    const alpha = RING_ALPHA[i] * look.o
    // The accent dims as the spark comes up, so the ring *changes* colour rather
    // than acquiring a second one on top of the first.
    node.setAttribute("stroke-opacity", (alpha * (1 - 0.85 * lit)).toFixed(3))
    node.setAttribute("stroke-width", (1.1 * look.weight).toFixed(2))

    if (!sparkNode) continue
    if (lit > 0.004) {
      sparkNode.setAttribute("d", d)
      sparkNode.setAttribute("stroke-opacity", (alpha * lit).toFixed(3))
      sparkNode.setAttribute("stroke-width", (1.35 * look.weight).toFixed(2))
    } else {
      sparkNode.setAttribute("stroke-opacity", "0")
    }
  }
}

/**
 * A closed, smooth path through the sampled points.
 *
 * Quadratics between successive midpoints rather than a spline through the points
 * themselves: it is one control point per sample instead of two, it is continuous in
 * tangent everywhere including across the join, and at these sample densities the
 * curve it draws and the curve an interpolating spline would draw differ by less
 * than the line is wide. A polyline was the other option and shows its corners at
 * `THROBBER_PX` — the rings are the shape here, so their edges have to be curves.
 */
function closedCurve(xs: Float64Array, ys: Float64Array): string {
  const last = SAMPLES - 1
  let d = `M${((xs[last] + xs[0]) / 2).toFixed(2)},${((ys[last] + ys[0]) / 2).toFixed(2)}`
  for (let j = 0; j < SAMPLES; j += 1) {
    const next = j === last ? 0 : j + 1
    d +=
      `Q${xs[j].toFixed(2)},${ys[j].toFixed(2)} ` +
      `${((xs[j] + xs[next]) / 2).toFixed(2)},${((ys[j] + ys[next]) / 2).toFixed(2)}`
  }
  return `${d}Z`
}

// ── the states ──────────────────────────────────────────────────────────────

const LOOKS: Record<ThrobberMode, Look> = {
  // idle — the shape is held. `flow` is not quite zero, so over a minute the figure
  // is not the same figure, but nothing in it is moving at a speed the eye tracks.
  // Present rather than working: the mark is still there, which is what stops
  // "nothing is happening" from looking like "this component unmounted".
  idle: {
    spin: 0.005,
    flow: 0.02,
    warp: 0.9,
    breathe: 0,
    breatheRate: 0,
    ripple: 0,
    rippleRate: 0,
    scan: 0,
    scanRate: 0,
    scanWidth: 2,
    flash: 0,
    flashRate: 0,
    o: 0.45,
    weight: 0.85,
  },

  // loading — the whole figure turns and reshapes steadily. The plainest moving
  // state, and the floor for all of them: a fifth of a turn a second and a shape that
  // is visibly not the shape it was two seconds ago. Nothing is being decided or
  // produced yet, something is simply on its way — but "on its way" still has to look
  // like motion, not like a still frame with a slow drift on it.
  loading: {
    spin: 0.16,
    flow: 0.75,
    warp: 1.2,
    breathe: 0.035,
    breatheRate: 0.5,
    ripple: 0,
    rippleRate: 0,
    scan: 0,
    scanRate: 0,
    scanWidth: 2,
    flash: 0,
    flashRate: 0,
    o: 0.9,
    weight: 1,
  },

  // thinking — the *colour* carries this state: a band two rings wide travels from
  // the centre outward, in the one second hue the mark spends, fast enough to be
  // clearly a scan rather than a glow that comes and goes. Something happening inside
  // the shape rather than to it, which is the one thing movement alone cannot say —
  // hence the second colour, and hence the figure underneath turning slower than the
  // other working states so the scan is what the eye goes to.
  thinking: {
    spin: 0.1,
    flow: 0.6,
    warp: 1.3,
    breathe: 0.03,
    breatheRate: 0.5,
    ripple: 0,
    rippleRate: 0,
    scan: 1,
    scanRate: 0.9,
    scanWidth: 2,
    flash: 0,
    flashRate: 0,
    o: 0.85,
    weight: 1,
  },

  // generating — the fast one, by a wide margin: half a turn a second, the field
  // running under it four times quicker than `loading`'s, and a swell travelling
  // outward through the rings twice a second on top of that. It should look busy to
  // the point of being slightly urgent. The speed and the travelling relief together
  // are what make it read as output being produced rather than waited for, and this
  // is the state that has to be unmistakable at a glance, because it is the one the
  // user sees most.
  generating: {
    spin: 0.5,
    flow: 2.2,
    warp: 1.45,
    breathe: 0.03,
    breatheRate: 1.1,
    ripple: 0.07,
    rippleRate: 2,
    scan: 0,
    scanRate: 0,
    scanWidth: 2,
    flash: 0,
    flashRate: 0,
    o: 1,
    weight: 1.05,
  },

  // tool — every ring flashes to the second colour at once and falls back, better
  // than once a second, on a figure that is turning and reshaping under it. The only
  // state where all five do the same thing at the same instant, and the only one
  // meant to feel abrupt: a discrete operation, not progress. The flash is the loud
  // part, so the motion beneath it is deliberately less than `generating`'s — but it
  // is not still, or the mark looks frozen between flashes.
  tool: {
    spin: 0.2,
    flow: 0.9,
    warp: 1.25,
    breathe: 0.025,
    breatheRate: 0.8,
    ripple: 0,
    rippleRate: 0,
    scan: 0,
    scanRate: 0,
    scanWidth: 2,
    flash: 1,
    flashRate: 1.5,
    o: 1,
    weight: 1.1,
  },

  // training — the whole figure breathes, every ring in and out together, hard enough
  // that the outermost travels a tenth of its radius each way. Uniform motion is the
  // signature: unlike every other state nothing travels through the rings, because
  // nothing here is being handed along. It is bulk work on the whole thing at once,
  // and the breath is the largest single movement in the component.
  training: {
    spin: 0.15,
    flow: 0.55,
    warp: 1.3,
    breathe: 0.1,
    breatheRate: 0.7,
    ripple: 0,
    rippleRate: 0,
    scan: 0,
    scanRate: 0,
    scanWidth: 2,
    flash: 0,
    flashRate: 0,
    o: 0.95,
    weight: 1,
  },

  // benchmarking — one ring at a time takes the colour, holds it and hands it on, the
  // figure turning gently under it. The deliberate counterpart to `thinking`'s band:
  // the same outward scan, narrowed until it picks out a single ring, which reads as
  // taking one reading after another. Slower than `thinking` on purpose — each ring
  // has to be seen to be *held* — so the drift underneath carries the motion.
  benchmarking: {
    spin: 0.09,
    flow: 0.45,
    warp: 1.2,
    breathe: 0.02,
    breatheRate: 0.45,
    ripple: 0,
    rippleRate: 0,
    scan: 0.9,
    scanRate: 0.62,
    scanWidth: 0.5,
    flash: 0,
    flashRate: 0,
    o: 1,
    weight: 1,
  },
}

// ── maths ───────────────────────────────────────────────────────────────────

function mix(a: Look, b: Look, w: number): Look {
  const lerp = (x: number, y: number) => x + (y - x) * w
  return {
    spin: lerp(a.spin, b.spin),
    flow: lerp(a.flow, b.flow),
    warp: lerp(a.warp, b.warp),
    breathe: lerp(a.breathe, b.breathe),
    breatheRate: lerp(a.breatheRate, b.breatheRate),
    ripple: lerp(a.ripple, b.ripple),
    rippleRate: lerp(a.rippleRate, b.rippleRate),
    scan: lerp(a.scan, b.scan),
    scanRate: lerp(a.scanRate, b.scanRate),
    scanWidth: lerp(a.scanWidth, b.scanWidth),
    flash: lerp(a.flash, b.flash),
    flashRate: lerp(a.flashRate, b.flashRate),
    o: lerp(a.o, b.o),
    weight: lerp(a.weight, b.weight),
  }
}

function fract(x: number): number {
  return x - Math.floor(x)
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi)
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const u = clamp((x - edge0) / (edge1 - edge0), 0, 1)
  return u * u * (3 - 2 * u)
}
