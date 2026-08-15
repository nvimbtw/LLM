"use client"

import * as React from "react"

import { cn } from "@/lib/utils"

/**
 * The rail — one loading indicator for the whole site.
 *
 * A track, and a segment moving on it. That is the entire figure: a thin pill
 * lying at rest under a brighter one that moves, grows, fills or is cut. Nothing
 * else is ever drawn.
 *
 * The track is the change that let everything else get simpler. Before it, the
 * bar was always full and *hue alone* had to carry the state, which needed six
 * gradient stops, two stacked strokes and a fifteen-field state vector to stay
 * legible. With a track, a state is told apart by **where the segment is and how
 * much of the rail it covers** — position and extent, which read instantly and at
 * any size — and colour becomes the second signal it should always have been.
 *
 * Three rules survive from the design before it, because each was learned the
 * hard way:
 *
 *  1. **Geometry is built once at mount.** A frame writes two coordinates, two
 *     opacities and (in one state) a dash — never path data. An earlier version
 *     rebuilt five 60-segment paths per frame, and the browser re-parsing those
 *     is why it was smooth on one machine and stuttered on another.
 *  2. **The phase is integrated, never read off the clock.** Each frame adds
 *     `rate × dt`, so changing mode changes the *speed* the figure is moving at
 *     and it carries straight through. A state change can never jump it.
 *  3. **Two states mix by weight**, so one instance given a changing `mode`
 *     morphs instead of cutting. What is mixed here is the *output* — the four
 *     numbers a mode resolves to on this frame — rather than a vector of
 *     parameters feeding one shared formula. That is what lets each state be
 *     three lines of arithmetic with a name, instead of a row in a table of
 *     fifteen columns whose combinations nobody can picture.
 *
 * Idle is not a special case: its segment simply barely moves.
 */
export type ThrobberMode =
  | "idle"
  | "loading"
  | "thinking"
  | "generating"
  | "tool"
  | "training"
  | "benchmarking"
  | "tokenizing"

export interface ThrobberProps {
  mode?: ThrobberMode
  className?: string
  /**
   * The resting colour — the track, and the segment before it warms. Override
   * where the mark sits on a primary surface (e.g. a filled button) so it does
   * not vanish into it.
   */
  accent?: string
  /**
   * The warm colour a segment turns as it works.
   *
   * The pair has its own two tokens rather than borrowing `--primary`, and the
   * reason is measured: sharing the primary put the two ΔL* 8 apart in OKLab, so
   * they differed almost only in hue — the weakest way to tell two things apart
   * at a glance, and no way at all for some readers. The separation now runs
   * ΔL* 22 in light and 24 in dark, with the hues in near-perfect opposition, and
   * the pair still stands ~37 apart on lightness under simulated deuteranopia.
   * If you override one of these, override both, and check that gap.
   */
  spark?: string
}

/**
 * The mark's footprint, in CSS pixels — everywhere, without exception.
 *
 * It used to be sized by its caller, which meant a different figure in every row
 * it appeared in, and no two of them looked like the same object. Callers
 * position it; they never size it.
 */
export const THROBBER_W = 84
export const THROBBER_H = 22

const TAU = Math.PI * 2

// Geometry, in viewBox units — 1:1 with the CSS size, so a stroke width here is a
// stroke width on screen and the numbers below can be reasoned about directly.
const PAD = 5
const X1 = PAD
const X2 = THROBBER_W - PAD
const SPAN = X2 - X1
const MID = THROBBER_H / 2

// The rail's thickness. Round caps make the ends semicircles of half this, so it
// also sets how round the pill is — there is no separate radius to keep in step.
const STROKE = 10

// How many pieces the rail is cut into while tokenizing, and how much daylight
// there is between them.
//
// Both are sized against the round caps rather than against the rail. A cap adds
// half a stroke to each end of every dash, so a piece is drawn `STROKE` wider
// than its dash and a gap is `STROKE` narrower than its number: at four pieces
// the maths is a 14.5px pill with 4px of daylight, which is the smallest gap that
// still reads as a cut rather than as a seam.
const CUT_PIECES = 4
const CUT_PITCH = SPAN / CUT_PIECES
const CUT_GAP = STROKE + 4

// How far past each end of the rail a segment may be placed. Everything out
// there is hidden by the clip, and the point of putting it there is that a
// segment then *arrives* full-size from outside instead of growing out of the
// rail's end.
//
// Two distances, because two things travel: a plain segment is hidden once its
// round cap clears the end, while a cut one has to carry a whole piece and its
// gap past it before the pattern repeats cleanly.
const OVERSHOOT = STROKE / SPAN
const CUT_OVERSHOOT = (CUT_PITCH + STROKE) / SPAN

// How much of the resting colour the track keeps. Low enough to read as the
// unfilled part of something rather than as a second mark, high enough to give
// the segment somewhere to be.
const TRACK_ALPHA = 0.18

// How long a mode change takes to cross-fade: long enough to read as the mark
// changing pace, short enough that it never delays feedback.
const BLEND_MS = 380

// A frame this long was probably a background tab rather than a slow frame.
// Clamped so returning to the tab does not teleport the segment through a
// minute's worth of travel.
const MAX_FRAME_S = 1 / 15

const LABELS: Record<ThrobberMode, string> = {
  idle: "Idle",
  loading: "Loading",
  thinking: "Thinking",
  generating: "Generating",
  tool: "Running a tool",
  training: "Training",
  benchmarking: "Benchmarking",
  tokenizing: "Building a vocabulary",
}

/**
 * What one mode resolves to on one frame. Four numbers and a dash — everything
 * the renderer knows how to draw, and everything a crossfade has to mix.
 */
interface Frame {
  /** Where the segment starts, along the rail, 0–1. */
  from: number
  /** Where it ends. Always ≥ `from`. */
  to: number
  /** 0 = the resting colour, 1 = the warm one. */
  tint: number
  /** The segment's opacity. */
  alpha: number
  /** How far the segment is cut into pieces, 0 = whole. */
  cuts: number
}

/**
 * How fast each mode's phase advances, in cycles per second.
 *
 * This is the whole tempo of the component, kept in one place so the states can
 * be compared as what they are: generating is five times idle, and it should be.
 */
const RATES: Record<ThrobberMode, number> = {
  idle: 0.08,
  loading: 0.42,
  thinking: 0.26,
  generating: 1.1,
  tool: 0.85,
  training: 0.34,
  benchmarking: 0.3,
  // One piece-width of travel per cycle, so this is also how fast the cuts march:
  // 0.55 is ~10px a second, which is a walk rather than a slide.
  tokenizing: 0.55,
}

/**
 * Each state, as a function of its phase. One per mode, each a picture you can
 * hold in your head:
 *
 *   idle          a short segment resting at the centre, barely breathing
 *   loading       a wide segment crossing the rail and wrapping
 *   thinking      colour blooming out of the middle and drawing back in
 *   generating    a short segment running the rail, fast, warm
 *   tool          the whole rail lighting at once and falling back
 *   training      the whole rail held, pulsing warm and cool as one
 *   benchmarking  a fill advancing in discrete steps, then resetting
 *   tokenizing    a fill cut into pieces, the cuts travelling along it
 */
const STATES: Record<ThrobberMode, (phase: number) => Frame> = {
  // Present rather than working. The point of a resting state is that the mark is
  // still *there* — nothing happening must not look like the component unmounted.
  idle: (p) => {
    const breath = 0.5 - 0.5 * Math.cos(TAU * p)
    const half = 0.09 + 0.02 * breath
    return { from: 0.5 - half, to: 0.5 + half, tint: 0, alpha: 0.5, cuts: 0 }
  },

  // The plainest working state, and the floor for all of them: nothing is being
  // decided or produced yet, something is simply on its way. Wide and unhurried,
  // so it never competes with the states that mean real work.
  loading: (p) => ({ ...sweep(p, 0.4), tint: 0.35, alpha: 0.95, cuts: 0 }),

  // Something happening *inside* the figure rather than passing along it — the
  // distinction the mark has to make between reasoning and every other kind of
  // work. Slow, because the drawing back in is as much of the gesture as the
  // bloom, and a fast one reads as a pulse.
  thinking: (p) => {
    const bloom = 0.5 - 0.5 * Math.cos(TAU * p)
    const half = 0.06 + 0.44 * bloom
    return {
      from: 0.5 - half,
      to: 0.5 + half,
      // Warmest at full spread, so the colour arrives *with* the bloom instead of
      // being a separate thing happening at the same time.
      tint: 0.25 + 0.75 * bloom,
      alpha: 0.9,
      cuts: 0,
    }
  },

  // The fast one by a wide margin, and the state a user sees most, so it has to
  // be unmistakable at a glance. Short, warm, and moving at a speed nothing else
  // here comes close to.
  generating: (p) => ({ ...sweep(p, 0.17), tint: 1, alpha: 1, cuts: 0 }),

  // The only state where every part of the rail does the same thing at the same
  // instant, and the only one meant to feel abrupt: a discrete operation, not
  // progress. Sharp at the peak and slow away from it — cubing a raised cosine
  // gets that, and stays continuous where the cycle wraps.
  tool: (p) => {
    const beat = Math.pow(0.5 + 0.5 * Math.cos(TAU * fract(p)), 3)
    return { from: 0, to: 1, tint: beat, alpha: 0.55 + 0.45 * beat, cuts: 0 }
  },

  // Bulk work on the whole thing. Nothing travels, because nothing here is being
  // handed along: the rail is full and tints as one body, slowly, which is the
  // longest, calmest movement in the component.
  training: (p) => {
    const swell = 0.5 - 0.5 * Math.cos(TAU * p)
    return { from: 0, to: 1, tint: 0.2 + 0.6 * swell, alpha: 0.9, cuts: 0 }
  },

  // One reading taken after another and kept. The steps are the state: a smooth
  // fill would say the same thing as loading, and this is emphatically not that —
  // it is a countable number of discrete results accumulating.
  benchmarking: (p) => {
    const steps = 6
    const at = fract(p)
    // Each step eases into place and then holds, so the eye reads an arrival
    // rather than a slide. The hold is the two-thirds of each step that is flat.
    const index = Math.floor(at * steps)
    const within = smoothstep(0, 0.34, at * steps - index)
    return {
      from: 0,
      to: (index + within) / steps,
      tint: 1,
      alpha: 1,
      cuts: 0,
    }
  },

  // A continuous thing being divided up, which is the only honest picture of a
  // vocabulary build. The fill is calm on purpose — the cut is the signal, and a
  // busy figure under it would bury the one thing this state says.
  // The cut is held wide open and constant, and the *pieces travel* — a
  // continuous thing being divided and carried along. Nothing here pulses.
  //
  // It read as jank when the gaps breathed instead: a gap has to clear the whole
  // stroke width before any daylight shows between two round-capped pieces, so a
  // gap oscillating either side of that showed a solid rail for most of its cycle
  // and then snapped apart. Two of the three motions in a figure this small were
  // fighting each other, and the one that meant something — the travel — was the
  // one that was missing.
  // The cut rail runs past both ends, so the pieces march in from outside and out
  // the far side rather than being born at one end cap and dying at the other.
  tokenizing: () => ({
    from: -CUT_OVERSHOOT,
    to: 1 + CUT_OVERSHOOT,
    tint: 0.55,
    alpha: 0.95,
    cuts: 1,
  }),
}

/**
 * A segment of `length` crossing the rail once per cycle.
 *
 * It travels from entirely off one end to entirely off the other, and the rail
 * clips it — so a full-size pill slides in behind the track's rounded end,
 * crosses, and slides out behind the other. It used to be clamped to the rail
 * instead, which meant it *grew* out of the near end and *shrank* into the far
 * one: the segment appeared to be born a little way inside the track and die a
 * little way inside it, which is the one thing about the figure that looked
 * unresolved.
 *
 * `OVERSHOOT` — declared with the other geometry — is how far outside the rail is
 * far enough.
 */
function sweep(phase: number, length: number): Pick<Frame, "from" | "to"> {
  const travel = 1 + 2 * OVERSHOOT + length
  const head = -OVERSHOOT + fract(phase) * travel
  return { from: head - length, to: head }
}

const REDUCED_MOTION = "(prefers-reduced-motion: reduce)"

// Live-updating, SSR-safe read of the OS motion preference. A plain
// `matchMedia(...).matches` read is a snapshot: it never notices the setting
// changing, and on the server there is no `window` at all.
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
  accent = "var(--throbber-base)",
  spark = "var(--throbber-spark)",
}: ThrobberProps) {
  const reduced = usePrefersReducedMotion()
  // The clip is per-instance: several marks can be on a page at once, and a
  // duplicated id would leave every one of them pointing at whichever definition
  // the document happened to resolve — which breaks the moment the instance that
  // owned it unmounts.
  const clipId = `throbber-${React.useId().replace(/:/g, "")}`

  // The two segment layers: the same line in each colour, crossfaded by opacity.
  // Two elements rather than one whose colour is recomputed, because an opacity
  // is composited and a colour is a repaint — and because interpolating two CSS
  // colours by hand means doing colour arithmetic on values that are allowed to
  // be `var(--x)` or `currentColor`.
  const coolRef = React.useRef<SVGLineElement | null>(null)
  const warmRef = React.useRef<SVGLineElement | null>(null)

  // Mode crossfade. Held in refs so a transition never triggers a React render —
  // the whole animation runs outside the reconciler.
  const from = React.useRef(mode)
  const to = React.useRef(mode)
  const blendStart = React.useRef(0)

  // Bookkeeping in an effect rather than during render: a render can be discarded
  // or replayed under concurrent React, which would corrupt which mode we think
  // we are coming *from*. The rAF loop is the only reader, so starting the
  // crossfade a frame later than the render is imperceptible.
  React.useEffect(() => {
    if (to.current === mode) return
    from.current = to.current
    to.current = mode
    blendStart.current = performance.now()
  }, [mode])

  // The running phase. It persists across mode changes and across the effect
  // being torn down and rebuilt, which is the point: it describes a mark that has
  // been moving since the page loaded, not an animation that plays.
  const phase = React.useRef(0)
  const lastFrame = React.useRef(0)
  const staggered = React.useRef(false)
  const dashed = React.useRef(false)

  React.useEffect(() => {
    // Each instance starts at its own point in the cycle, so the several marks a
    // page can show at once are not one bar cloned. Claimed here rather than in
    // render because taking a number from a shared counter is a side effect, and
    // during render it would also make the first frame differ between server and
    // client for no benefit.
    if (!staggered.current) {
      staggered.current = true
      phase.current = nextStagger()
    }

    let frame = 0

    const tick = (now: number) => {
      const dt = clamp(
        (now - (lastFrame.current || now)) / 1000,
        0,
        MAX_FRAME_S
      )
      lastFrame.current = now

      const blend = smoothstep(0, 1, (now - blendStart.current) / BLEND_MS)
      // Reduced motion slows the mark rather than stopping it. Stopping is the
      // obvious reading of the preference and the wrong one — this is a status
      // indicator, and a motionless one reports nothing. The guidance targets
      // large, vestibular-triggering movement, and there is none here: the figure
      // is 84px wide and nothing in it travels far.
      const damp = reduced ? 0.35 : 1
      phase.current +=
        lerp(RATES[from.current], RATES[to.current], blend) * damp * dt

      const shown =
        blend >= 1
          ? STATES[to.current](phase.current)
          : mixFrames(
              STATES[from.current](phase.current),
              STATES[to.current](phase.current),
              blend
            )

      draw(shown, phase.current, coolRef.current, warmRef.current, dashed)
      frame = requestAnimationFrame(tick)
    }

    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [reduced])

  return (
    <span
      role="status"
      aria-label={LABELS[mode]}
      // The size is the component's, not the caller's — see `THROBBER_W`. An
      // inline style rather than a utility class, so the constants above are the
      // only place it is written down. `className` is still honoured for
      // everything a caller legitimately controls: margin, placement, colour.
      style={{ width: THROBBER_W, height: THROBBER_H }}
      className={cn(
        "inline-flex shrink-0 items-center justify-center",
        className
      )}
    >
      <svg
        viewBox={`0 0 ${THROBBER_W} ${THROBBER_H}`}
        className="block h-full w-full"
        aria-hidden
      >
        {/* The track. Never written to after mount — it is the one part of the
            figure that does not move, and that is what makes the part that does
            legible. */}
        <line
          x1={X1}
          y1={MID}
          x2={X2}
          y2={MID}
          stroke={accent}
          strokeWidth={STROKE}
          strokeLinecap="round"
          opacity={TRACK_ALPHA}
        />
        {/* The rail as a shape, so a segment travelling past either end is cut by
            it instead of stopping at it. Written as the rect the track's stroke
            covers — the same pill, so the cut edge follows the track's own
            rounded end exactly and a segment reads as passing behind it. */}
        <defs>
          <clipPath id={clipId}>
            <rect
              x={X1 - STROKE / 2}
              y={MID - STROKE / 2}
              width={SPAN + STROKE}
              height={STROKE}
              rx={STROKE / 2}
            />
          </clipPath>
        </defs>

        {/* The two segment layers, drawn empty. Both coordinates and both
            opacities are written every frame, but they have to *start* at
            nothing: an unset `x1`/`x2` is 0, and with round caps a zero-length
            line at full opacity is a full-width dot — so every mount, and every
            server render, would flash a bead at the left edge before the first
            frame landed. */}
        <g clipPath={`url(#${clipId})`}>
          <line
            ref={coolRef}
            x1={X1}
            x2={X1}
            opacity={0}
            y1={MID}
            y2={MID}
            stroke={accent}
            strokeWidth={STROKE}
            strokeLinecap="round"
          />
          <line
            ref={warmRef}
            x1={X1}
            x2={X1}
            opacity={0}
            y1={MID}
            y2={MID}
            stroke={spark}
            strokeWidth={STROKE}
            strokeLinecap="round"
          />
        </g>
      </svg>
    </span>
  )
}

/**
 * Writes one frame straight to the DOM. Deliberately imperative: this runs at
 * 60fps while a chat is streaming, and re-rendering React for it would make the
 * throbber the most expensive thing on the page.
 */
function draw(
  frame: Frame,
  phase: number,
  cool: SVGLineElement | null,
  warm: SVGLineElement | null,
  dashed: { current: boolean }
) {
  if (!cool || !warm) return

  // Clamped to the *travel*, not to the rail: a segment is meant to sit outside
  // the rail at each end of its cycle, and the clip is what hides it. The bound
  // is the widest any state reaches, so nothing here decides what a state may do.
  const lo = -CUT_OVERSHOOT
  const hi = 1 + CUT_OVERSHOOT
  const x1 = X1 + SPAN * clamp(frame.from, lo, hi)
  const x2 = X1 + SPAN * clamp(Math.max(frame.from, frame.to), lo, hi)
  // Round caps mean a zero-length line still paints a full-width dot, so a
  // segment shrinking into the end of the rail would leave a bead sitting there
  // and then blink out. Fading it over the last few units is that, without a
  // threshold to pop across.
  const emerging = smoothstep(0, STROKE * 0.6, x2 - x1)

  // The cut. `cuts` opens the gap from nothing to `CUT_GAP` — which exists only
  // so the state can be morphed into and out of — and the pattern then slides
  // one whole piece per cycle, which is the entire motion of that state.
  const cutting = frame.cuts > 0.004
  if (cutting) {
    const gap = frame.cuts * CUT_GAP
    const dash = Math.max(0.5, CUT_PITCH - gap)
    const offset = -fract(phase) * CUT_PITCH
    setDash(cool, dash, gap, offset)
    setDash(warm, dash, gap, offset)
  } else if (dashed.current) {
    cool.removeAttribute("stroke-dasharray")
    warm.removeAttribute("stroke-dasharray")
    cool.removeAttribute("stroke-dashoffset")
    warm.removeAttribute("stroke-dashoffset")
  }
  dashed.current = cutting

  const a = frame.alpha * emerging
  for (const [line, share] of [
    [cool, 1 - frame.tint],
    [warm, frame.tint],
  ] as const) {
    line.setAttribute("x1", x1.toFixed(2))
    line.setAttribute("x2", x2.toFixed(2))
    line.setAttribute("opacity", (a * share).toFixed(3))
  }
}

function setDash(
  line: SVGLineElement,
  dash: number,
  gap: number,
  offset: number
) {
  line.setAttribute("stroke-dasharray", `${dash.toFixed(2)} ${gap.toFixed(2)}`)
  line.setAttribute("stroke-dashoffset", offset.toFixed(2))
}

// Successive starting offsets into the cycle. Stepping by the golden ratio
// spreads any number of instances about as evenly as they can be spread, without
// needing to know in advance how many there will be.
let stagger = 0
function nextStagger(): number {
  stagger = fract(stagger + 0.6180339887)
  return stagger
}

// ── maths ───────────────────────────────────────────────────────────────────

function mixFrames(a: Frame, b: Frame, w: number): Frame {
  return {
    from: lerp(a.from, b.from, w),
    to: lerp(a.to, b.to, w),
    tint: lerp(a.tint, b.tint, w),
    alpha: lerp(a.alpha, b.alpha, w),
    cuts: lerp(a.cuts, b.cuts, w),
  }
}

function lerp(a: number, b: number, w: number): number {
  return a + (b - a) * w
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
