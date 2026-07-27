"use client"

import * as React from "react"

import { cn } from "@/lib/utils"

/**
 * Ledger ticks — one loading indicator for the whole site.
 *
 * Three hairline graduations standing on an invisible baseline, in a fixed square
 * footprint. The state vocabulary is entirely in how the three heights move: they
 * stand still at rest, breathe irregularly to reason, pass a peak along to
 * generate, snap the middle one up for a tool call, rise together to train, and
 * light one at a time to benchmark.
 *
 * Three decisions shape it:
 *
 *  1. **Fixed footprint.** It never spans its container. The previous version was
 *     a full-width rule, which made it as wide as whatever held it and put it in
 *     competition with the page instead of beside a label. Sizing is by height
 *     (`h-4`, `h-5`); the width follows the aspect and the mark centres itself in
 *     whatever box it's given.
 *  2. **Every state renders the same three ticks** and differs only in a vector of
 *     six numbers (`Marks`). Because a state is just numbers, any two can be mixed
 *     by weight — so a mode change crossfades and the ticks travel to their new
 *     behaviour rather than cutting to it. Give one instance a changing `mode`
 *     rather than swapping instances, or there's nothing to morph between.
 *  3. **States are told apart by motion, not hue.** One accent, and opacity only
 *     ever reinforces what a height is already saying.
 *
 * Strokes are `non-scaling-stroke`, so the hairline stays a hairline at any size
 * instead of fattening as the mark grows.
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
   * Colour of the ticks. Override where the mark already sits on a primary surface
   * (e.g. the send button) so it doesn't vanish.
   */
  accent?: string
}

const TAU = Math.PI * 2

// Geometry, in viewBox units. A 14×14 box: three ticks standing on a baseline just
// inside the bottom edge, growing upward.
//
// A tick at full height very nearly fills the box. There's no interior padding to
// spare at this size — every unit held back is one the mark doesn't get to use, and
// the whole point of the fixed footprint is that the footprint stays small.
const VIEW = 14
const TICK_X = [2, 7, 12]
// The ticks stand on the very bottom edge of the box, with nothing held back below
// them. That's what makes the mark's own bottom edge *be* its baseline — so a caller
// that puts it in a `items-baseline` row gets the ticks sitting on the same line as
// the text beside them, with no magic offset to maintain. (An inline-flex box whose
// content is a replaced element takes its baseline from its bottom margin edge.)
const BASE_Y = VIEW
// MIN_H is the resting height, so it also sets how present the idle state is. Too low
// and three ticks at rest read as leftover dots rather than as a mark that is
// deliberately holding still.
const MIN_H = 4
const MAX_H = 12

// How long a mode change takes to crossfade. Long enough to read as travel, short
// enough that it never delays feedback.
const BLEND_MS = 380

// The interpolable state vector: one height and one opacity per tick. Every field is
// a plain number precisely so that any two modes can be mixed by weight.
interface Marks {
  /** Tick heights in viewBox units, MIN_H..MAX_H. */
  h: [number, number, number]
  o: [number, number, number]
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
}: ThrobberProps) {
  const reduced = usePrefersReducedMotion()
  const ticksRef = React.useRef<(SVGLineElement | null)[]>([])

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

  React.useEffect(() => {
    let frame = 0
    const start = performance.now()

    const tick = (now: number) => {
      // Reduced motion slows the clock rather than stopping it. Freezing was the
      // obvious reading of the preference and the wrong one: this is a status
      // indicator, and a motionless one reports nothing. The guidance targets large,
      // vestibular-triggering movement, so damping toward rest (below) honours the
      // intent while the mark still indicates.
      const elapsed = ((now - start) / 1000) * (reduced ? 0.45 : 1)
      const blend = Math.min((now - blendStart.current) / BLEND_MS, 1)

      let marks =
        blend >= 1
          ? marksFor(currentMode.current, elapsed)
          : mix(
              marksFor(previousMode.current, elapsed),
              marksFor(currentMode.current, elapsed),
              smoothstep(0, 1, blend)
            )
      // Halves every excursion, so the ticks still move but only just.
      if (reduced) marks = mix(marks, RESTING, 0.5)

      draw(marks, ticksRef.current)
      frame = requestAnimationFrame(tick)
    }

    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [reduced])

  return (
    <span
      role="status"
      aria-label={LABELS[mode]}
      // Centres itself in whatever box it's given, so a caller that hands it a wide
      // slot gets a centred mark rather than a stretched one.
      className={cn("inline-flex h-5 items-center justify-center", className)}
    >
      <svg
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        className="block h-full w-auto overflow-visible"
        fill="none"
        aria-hidden
      >
        {TICK_X.map((x, i) => (
          <line
            key={x}
            ref={(node) => {
              ticksRef.current[i] = node
            }}
            x1={x}
            x2={x}
            y1={BASE_Y}
            y2={BASE_Y - MIN_H}
            stroke={accent}
            strokeWidth={1.5}
            strokeLinecap="round"
            // Keeps the hairline a hairline: without this the stroke scales with the
            // box and an h-10 mark would be drawn in fat strokes.
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
    </span>
  )
}

// Writes one frame straight to the DOM. Deliberately imperative: this runs at 60fps
// while the chat is streaming, and re-rendering React for it would make the throbber
// the most expensive thing on the page.
function draw(marks: Marks, ticks: (SVGLineElement | null)[]) {
  for (let i = 0; i < 3; i += 1) {
    const node = ticks[i]
    if (!node) continue
    node.setAttribute("y2", `${BASE_Y - clamp(marks.h[i], MIN_H, MAX_H)}`)
    node.setAttribute("stroke-opacity", `${clamp(marks.o[i], 0, 1)}`)
  }
}

// ── the states ──────────────────────────────────────────────────────────────

// Each mode is a pure function of elapsed seconds to the state vector, which is what
// lets any two of them be mixed. Nothing here reads the DOM or the clock.
function marksFor(mode: ThrobberMode, t: number): Marks {
  switch (mode) {
    case "loading":
      return loading(t)
    case "thinking":
      return thinking(t)
    case "generating":
      return generating(t)
    case "tool":
      return tool(t)
    case "training":
      return training(t)
    case "benchmarking":
      return benchmarking(t)
    default:
      return RESTING
  }
}

const RESTING: Marks = { h: [MIN_H, MIN_H, MIN_H], o: [0.42, 0.42, 0.42] }

// loading — a low wave passing along the three ticks. The gentlest of the moving
// states: nothing is being decided or produced yet, something is simply on its way.
function loading(t: number): Marks {
  const p = TAU * phase(t, 1.15)
  // Centred so the trough clears the floor. A wave that clamps at MIN_H holds every
  // tick flat at the bottom of its swing, which reads as sticking, not waving.
  const at = (i: number) => level(0.42 + 0.32 * Math.sin(p - (i * TAU) / 3))
  return { h: [at(0), at(1), at(2)], o: [0.72, 0.72, 0.72] }
}

// thinking — irregular breathing. Two incommensurate periods per tick, offset from
// each other, so the pattern never visibly repeats: undecided, not cyclic.
function thinking(t: number): Marks {
  // Same reasoning as `loading`: the two amplitudes together must not reach the floor,
  // or the breath flattens out at the bottom.
  const at = (i: number) =>
    level(
      0.42 +
        0.17 * Math.sin((TAU * t) / 1.9 + i * 1.1) +
        0.09 * Math.sin((TAU * t) / 1.15 + i * 2.3)
    )
  return { h: [at(0), at(1), at(2)], o: [0.85, 0.85, 0.85] }
}

// generating — a peak handed along the ticks, wrapping. Faster and far higher in
// contrast than `loading`: one tick clearly leads, which is what makes it read as
// output being produced rather than waited for.
function generating(t: number): Marks {
  const head = phase(t, 0.78) * 3
  const at = (i: number) => weight(head, i)
  const w = [at(0), at(1), at(2)]
  return {
    h: [lift(w[0], 1.5), lift(w[1], 1.5), lift(w[2], 1.5)],
    o: [0.4 + 0.55 * w[0], 0.4 + 0.55 * w[1], 0.4 + 0.55 * w[2]],
  }
}

// tool — the middle tick snaps up, holds, snaps back. A quintic ease-out over a
// fraction of the dwell gives the arrival a mechanical hardness; the outer two stay
// down, so the whole mark reads as one discrete operation rather than progress.
function tool(t: number): Marks {
  const c = phase(t, 1.3)
  const up =
    c < 0.5
      ? easeOutQuint(Math.min(c / 0.12, 1))
      : 1 - easeOutQuint(Math.min((c - 0.5) / 0.12, 1))
  return {
    h: [MIN_H, MIN_H + (MAX_H - MIN_H) * up, MIN_H],
    o: [0.35, 0.5 + 0.5 * up, 0.35],
  }
}

// training — all three rising and falling as one, the amplitude building across
// eight-odd breaths and then relaxing. Uniform motion is the signature: unlike every
// other state nothing travels, because nothing here is being handed along — it's
// bulk work on the whole thing at once.
function training(t: number): Marks {
  const breath = 0.5 - 0.5 * Math.cos(TAU * phase(t, 1.05))
  const cycle = phase(t, 11)
  const build = cycle < 0.5 ? cycle * 2 : (1 - cycle) * 2
  const h = MIN_H + (MAX_H - MIN_H) * breath * (0.45 + 0.55 * build)
  const o = 0.55 + 0.35 * breath
  return { h: [h, h, h], o: [o, o, o] }
}

// benchmarking — one tick at full height at a time, eased into and then dwelt on, as
// if taking a reading. The slow, high-contrast counterpart to `generating`'s sweep:
// it ping-pongs rather than wrapping, and the pause on each tick is the point.
function benchmarking(t: number): Marks {
  const stops = [0, 1, 2, 1]
  const dwell = 0.62
  const step = Math.floor(t / dwell)
  const local = (t / dwell) % 1
  const from = stops[mod(step, stops.length)]
  const to = stops[mod(step + 1, stops.length)]
  const head = from + (to - from) * easeInOut(Math.min(local / 0.42, 1))
  const w = [weight(head, 0), weight(head, 1), weight(head, 2)]
  return {
    h: [lift(w[0], 2), lift(w[1], 2), lift(w[2], 2)],
    o: [0.3 + 0.7 * w[0], 0.3 + 0.7 * w[1], 0.3 + 0.7 * w[2]],
  }
}

// ── maths ───────────────────────────────────────────────────────────────────

/**
 * How much tick `i` belongs to a peak centred at `head` (in tick units). Triangular,
 * and measured the short way around the three ticks so a peak leaving the last one
 * arrives at the first without a jump.
 */
function weight(head: number, i: number): number {
  const raw = Math.abs(head - i)
  return Math.max(0, 1 - Math.min(raw, 3 - raw))
}

/** A 0..1 fraction of the tick's travel as a height in viewBox units. */
function level(f: number): number {
  return MIN_H + (MAX_H - MIN_H) * f
}

/** A 0..1 weight as a tick height, `sharpness` steepening the peak's falloff. */
function lift(w: number, sharpness: number): number {
  return level(Math.pow(w, sharpness))
}

function mix(a: Marks, b: Marks, w: number): Marks {
  const lerp = (x: number, y: number) => x + (y - x) * w
  return {
    h: [lerp(a.h[0], b.h[0]), lerp(a.h[1], b.h[1]), lerp(a.h[2], b.h[2])],
    o: [lerp(a.o[0], b.o[0]), lerp(a.o[1], b.o[1]), lerp(a.o[2], b.o[2])],
  }
}

/** Position within a repeating period, 0..1. */
function phase(t: number, period: number): number {
  return (((t % period) + period) % period) / period
}

function mod(n: number, m: number): number {
  return ((n % m) + m) % m
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi)
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const u = clamp((x - edge0) / (edge1 - edge0), 0, 1)
  return u * u * (3 - 2 * u)
}

function easeOutQuint(u: number): number {
  return 1 - Math.pow(1 - u, 5)
}

function easeInOut(u: number): number {
  return u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2
}
