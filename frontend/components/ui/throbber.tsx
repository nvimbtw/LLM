"use client"

import * as React from "react"

import { cn } from "@/lib/utils"

/**
 * The Carriage — one loading indicator for the whole site.
 *
 * A hairline rule that spans the full width of its container and *never moves*.
 * A single accent mark — the carriage — travels along it, and the motion
 * signature of that one mark is the entire state vocabulary: it sweeps to load,
 * hovers to think, advances and inks to generate, snaps between stops for a tool
 * call, sweeps repeatedly to train, and steps between graduations to benchmark.
 *
 * Two decisions make this one component rather than seven drawings:
 *
 *  1. **Every state renders the identical elements** — rule, graduations, trail,
 *     carriage — and differs only in a vector of numbers (`Marks`). The previous
 *     implementation gave each mode its own SVG subtree, so React unmounted one
 *     and mounted another and nothing could transition. Here a mode change
 *     crossfades the numbers, so the carriage physically travels to its new
 *     behaviour instead of teleporting.
 *  2. **States are told apart by motion, not hue.** One ink colour
 *     (`currentColor`) and one accent (`--primary`), as before.
 *
 * Sizing comes from layout: give it a height and let it take the width it's in
 * (`h-6 w-full`, `h-5 flex-1`). The viewBox tracks the measured pixel box, so
 * 1 user unit = 1 px and the carriage never distorts at any aspect ratio.
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
   * Colour of the carriage and its ink trail. Override where the mark already
   * sits on a primary surface (e.g. the send button) so it doesn't vanish.
   */
  accent?: string
}

const TAU = Math.PI * 2
// Where the graduation ticks sit, and — for `benchmarking` — the stops the
// carriage takes its readings at. The two must match or the mark would appear to
// measure between the marks on its own scale.
const GRADUATIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
// How long a mode change takes to crossfade. Long enough to read as travel,
// short enough that it never delays feedback.
const BLEND_MS = 420

// The interpolable state vector. Every field is a plain number precisely so that
// any two modes can be mixed by weight — this is what makes morphing possible.
interface Marks {
  /** Carriage position along the rule, 0..1. */
  carriage: number
  /** Carriage tick half-height, as a fraction of the box height. */
  carriageScale: number
  carriageOpacity: number
  /** Inked span of the rule, 0..1. */
  trailFrom: number
  trailTo: number
  trailOpacity: number
  /** Multiplier on the rule's stroke weight. */
  ruleWeight: number
  /** Graduation tick height, 0 = absent. */
  gradScale: number
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

export function Throbber({ mode = "idle", className, accent = "var(--primary)" }: ThrobberProps) {
  const reduced = usePrefersReducedMotion()
  const hostRef = React.useRef<HTMLSpanElement>(null)
  const ruleRef = React.useRef<SVGLineElement>(null)
  const trailRef = React.useRef<SVGLineElement>(null)
  const carriageRef = React.useRef<SVGLineElement>(null)
  const gradsRef = React.useRef<(SVGLineElement | null)[]>([])

  // Measured box, held twice on purpose. The state drives the viewBox, which is a
  // rendered attribute; the ref is what the animation loop reads. Keeping the loop
  // off the state is what stops a resize from restarting it — see the paint effect.
  const [box, setBox] = React.useState({ w: 0, h: 0 })
  const boxRef = React.useRef({ w: 0, h: 0 })

  // Mode crossfade. Held in refs so a transition never triggers a React render —
  // the whole animation runs outside the reconciler.
  const currentMode = React.useRef(mode)
  const previousMode = React.useRef(mode)
  const blendStart = React.useRef(0)

  // Bookkeeping happens in an effect rather than during render: a render can be
  // discarded or replayed under concurrent React, which would corrupt which mode
  // we think we're coming *from*. The rAF loop is the only reader, so starting the
  // crossfade a frame later than the render is imperceptible.
  React.useEffect(() => {
    if (currentMode.current === mode) return
    previousMode.current = currentMode.current
    currentMode.current = mode
    blendStart.current = performance.now()
  }, [mode])

  React.useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      // The ref updates every time so the next frame draws at the true size; the
      // state only updates on a real change, to avoid a pointless re-render.
      boxRef.current = { w: width, h: height }
      setBox((prev) =>
        Math.abs(prev.w - width) < 0.5 && Math.abs(prev.h - height) < 0.5
          ? prev
          : { w: width, h: height }
      )
    })
    observer.observe(host)
    return () => observer.disconnect()
  }, [])

  // The animation clock must not depend on layout. It used to run off `box`
  // state, so anything that resized the mark — most visibly the cycling
  // "Thinking… / Smelting…" label beside it in the chat — tore the loop down and
  // restarted it, resetting `elapsed` to 0 and snapping the carriage back to its
  // t=0 position. Reading the size from a ref keeps one continuous clock for the
  // component's whole life, and a resize just changes where the next frame lands.
  React.useEffect(() => {
    const paint = (elapsed: number, w: number, h: number) => {
      // Reduced motion slows the clock rather than stopping it. Freezing was the
      // obvious reading of the preference and the wrong one: this is a status
      // indicator, and a motionless one reports nothing. The guidance targets
      // large, vestibular-triggering movement, so damping the travel toward rest
      // (below) honours the intent while the mark still indicates.
      const clock = reduced ? elapsed * 0.45 : elapsed
      const now = typeof performance !== "undefined" ? performance.now() : 0
      const blend = Math.min((now - blendStart.current) / BLEND_MS, 1)
      let marks =
        blend >= 1
          ? marksFor(currentMode.current, clock)
          : mix(
              marksFor(previousMode.current, clock),
              marksFor(currentMode.current, clock),
              smoothstep(0, 1, blend)
            )
      // Halves every excursion — sweep amplitude, tick height, the opacity
      // envelopes that would otherwise blank the carriage mid-pass.
      if (reduced) marks = mix(marks, RESTING, 0.5)
      draw(marks, w, h, { ruleRef, trailRef, carriageRef, gradsRef })
    }

    let frame = 0
    const start = performance.now()
    const tick = (now: number) => {
      const { w, h } = boxRef.current
      // Before the first measurement there is nothing to draw into, but the loop
      // keeps running so the clock stays continuous from mount.
      if (w > 0 && h > 0) paint((now - start) / 1000, w, h)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [reduced])

  const strokeWidth = Math.min(Math.max(box.h * 0.055, 1), 2)

  return (
    <span
      ref={hostRef}
      role="status"
      aria-label="Loading"
      className={cn("relative block w-full text-foreground", className)}
    >
      {box.w > 0 && (
        <svg
          viewBox={`0 0 ${box.w} ${box.h}`}
          width={box.w}
          height={box.h}
          className="block overflow-visible"
          fill="none"
          aria-hidden
        >
          <line ref={ruleRef} stroke="currentColor" strokeOpacity={0.28} strokeLinecap="round" />
          {GRADUATIONS.map((_, i) => (
            <line
              key={i}
              ref={(node) => {
                gradsRef.current[i] = node
              }}
              stroke="currentColor"
              strokeWidth={Math.max(strokeWidth * 0.8, 1)}
              strokeLinecap="butt"
            />
          ))}
          <line ref={trailRef} stroke={accent} strokeLinecap="round" />
          <line ref={carriageRef} stroke={accent} strokeLinecap="round" />
        </svg>
      )}
    </span>
  )
}

// ── drawing ─────────────────────────────────────────────────────────────────

// Writes one frame straight to the DOM. Deliberately imperative: this runs at
// 60fps while the chat is streaming, and re-rendering React for it would make the
// throbber the most expensive thing on the page.
function draw(
  marks: Marks,
  w: number,
  h: number,
  refs: {
    ruleRef: React.RefObject<SVGLineElement | null>
    trailRef: React.RefObject<SVGLineElement | null>
    carriageRef: React.RefObject<SVGLineElement | null>
    gradsRef: React.RefObject<(SVGLineElement | null)[]>
  }
) {
  // Inset so a round cap at either extreme isn't clipped by the viewBox.
  const inset = Math.min(Math.max(w * 0.015, 2), 10)
  const span = w - inset * 2
  const x = (u: number) => inset + u * span
  const midY = h / 2
  const weight = Math.min(Math.max(h * 0.055, 1), 2)

  const rule = refs.ruleRef.current
  if (rule) {
    rule.setAttribute("x1", `${inset}`)
    rule.setAttribute("y1", `${midY}`)
    rule.setAttribute("x2", `${inset + span}`)
    rule.setAttribute("y2", `${midY}`)
    rule.setAttribute("stroke-width", `${weight * marks.ruleWeight}`)
  }

  const grads = refs.gradsRef.current
  if (grads) {
    const height = marks.gradScale * h * 0.26
    for (let i = 0; i < GRADUATIONS.length; i += 1) {
      const node = grads[i]
      if (!node) continue
      const gx = x(GRADUATIONS[i])
      node.setAttribute("x1", `${gx}`)
      node.setAttribute("y1", `${midY}`)
      node.setAttribute("x2", `${gx}`)
      node.setAttribute("y2", `${midY + height}`)
      node.setAttribute("stroke-opacity", `${0.3 * marks.gradScale}`)
    }
  }

  const trail = refs.trailRef.current
  if (trail) {
    trail.setAttribute("x1", `${x(marks.trailFrom)}`)
    trail.setAttribute("y1", `${midY}`)
    trail.setAttribute("x2", `${x(marks.trailTo)}`)
    trail.setAttribute("y2", `${midY}`)
    trail.setAttribute("stroke-width", `${weight * 1.35}`)
    trail.setAttribute("stroke-opacity", `${marks.trailOpacity}`)
  }

  const carriage = refs.carriageRef.current
  if (carriage) {
    const cx = x(marks.carriage)
    const half = marks.carriageScale * h * 0.9
    carriage.setAttribute("x1", `${cx}`)
    carriage.setAttribute("y1", `${midY - half}`)
    carriage.setAttribute("x2", `${cx}`)
    carriage.setAttribute("y2", `${midY + half}`)
    carriage.setAttribute("stroke-width", `${weight * 1.5}`)
    carriage.setAttribute("stroke-opacity", `${marks.carriageOpacity}`)
  }
}

// ── the states ──────────────────────────────────────────────────────────────

// Each mode is a pure function of elapsed seconds to the state vector, which is
// what lets any two of them be mixed. Nothing here reads the DOM or the clock.
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
      return idle(t)
  }
}

const RESTING: Marks = {
  carriage: 0.5,
  carriageScale: 0.16,
  carriageOpacity: 0.5,
  trailFrom: 0.5,
  trailTo: 0.5,
  trailOpacity: 0,
  ruleWeight: 1,
  gradScale: 0,
}

// idle — the carriage parked at centre, breathing so slowly it reads as present
// rather than active.
function idle(t: number): Marks {
  return { ...RESTING, carriageOpacity: 0.42 + 0.14 * Math.sin((TAU * t) / 3.6) }
}

// loading — a full-width sweep. Driven by a cosine so it decelerates into each
// turn and accelerates out, with no discontinuity at the ends.
function loading(t: number): Marks {
  const u = 0.5 - 0.5 * Math.cos(TAU * phase(t, 2.6))
  return { ...RESTING, carriage: 0.03 + u * 0.94, carriageScale: 0.18, carriageOpacity: 0.95 }
}

// thinking — hovering near the centre, drifting. Two incommensurate sines so the
// wander never visibly repeats: undecided, not cyclic.
function thinking(t: number): Marks {
  const drift = 0.055 * Math.sin((TAU * t) / 1.9) + 0.028 * Math.sin((TAU * t) / 1.15 + 1.4)
  return {
    ...RESTING,
    carriage: 0.5 + drift,
    carriageScale: 0.17 + 0.035 * Math.sin((TAU * t) / 0.9),
    carriageOpacity: 0.9,
  }
}

// generating — advancing steadily and inking the rule behind it. The envelope
// fades the pass in and out at the wrap so the reset is never seen as a jump.
function generating(t: number): Marks {
  const p = phase(t, 2.4)
  const carriage = 0.03 + p * 0.94
  const envelope = smoothstep(0, 0.07, p) * smoothstep(0, 0.12, 1 - p)
  return {
    ...RESTING,
    carriage,
    carriageScale: 0.2,
    carriageOpacity: envelope,
    trailFrom: 0.03,
    trailTo: carriage,
    trailOpacity: 0.75 * envelope,
  }
}

// tool — snapping between fixed stops and holding. Quintic ease-out over the
// first fifth of each dwell gives the arrival a mechanical hardness; the stops
// ping-pong so the carriage never makes a long jump back to the start.
function tool(t: number): Marks {
  const stops = [0.12, 0.38, 0.62, 0.88, 0.62, 0.38]
  const dwell = 0.42
  const step = Math.floor(t / dwell)
  const local = (t / dwell) % 1
  const from = stops[mod(step, stops.length)]
  const to = stops[mod(step + 1, stops.length)]
  return {
    ...RESTING,
    carriage: from + (to - from) * easeOutQuint(Math.min(local / 0.2, 1)),
    carriageScale: 0.26,
    carriageOpacity: 1,
  }
}

// training — repeated passes that leave the rule progressively heavier, then
// release. The weight follows a slow triangle across eight passes so it builds
// and relaxes continuously rather than resetting.
function training(t: number): Marks {
  const p = phase(t, 1.6)
  const carriage = 0.03 + p * 0.94
  const envelope = smoothstep(0, 0.06, p) * smoothstep(0, 0.1, 1 - p)
  const cycle = phase(t, 12.8)
  const build = cycle < 0.5 ? cycle * 2 : (1 - cycle) * 2
  return {
    ...RESTING,
    carriage,
    carriageScale: 0.22,
    carriageOpacity: envelope,
    trailFrom: 0.03,
    trailTo: carriage,
    trailOpacity: (0.3 + 0.45 * build) * envelope,
    ruleWeight: 1 + build * 1.5,
  }
}

// benchmarking — stepping between the graduations and dwelling on each, as if
// taking a reading. Deliberately the slow, eased counterpart to `tool`'s snap,
// and the only state that raises the scale it measures against.
function benchmarking(t: number): Marks {
  const stops = [...GRADUATIONS, 0.7, 0.5, 0.3]
  const dwell = 0.78
  const step = Math.floor(t / dwell)
  const local = (t / dwell) % 1
  const from = stops[mod(step, stops.length)]
  const to = stops[mod(step + 1, stops.length)]
  return {
    ...RESTING,
    carriage: from + (to - from) * easeInOut(Math.min(local / 0.45, 1)),
    carriageScale: 0.28,
    carriageOpacity: 1,
    gradScale: 1,
  }
}

// ── maths ───────────────────────────────────────────────────────────────────

function mix(a: Marks, b: Marks, w: number): Marks {
  const lerp = (x: number, y: number) => x + (y - x) * w
  return {
    carriage: lerp(a.carriage, b.carriage),
    carriageScale: lerp(a.carriageScale, b.carriageScale),
    carriageOpacity: lerp(a.carriageOpacity, b.carriageOpacity),
    trailFrom: lerp(a.trailFrom, b.trailFrom),
    trailTo: lerp(a.trailTo, b.trailTo),
    trailOpacity: lerp(a.trailOpacity, b.trailOpacity),
    ruleWeight: lerp(a.ruleWeight, b.ruleWeight),
    gradScale: lerp(a.gradScale, b.gradScale),
  }
}

/** Position within a repeating period, 0..1. */
function phase(t: number, period: number): number {
  return ((t % period) + period) % period / period
}

function mod(n: number, m: number): number {
  return ((n % m) + m) % m
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const u = Math.min(Math.max((x - edge0) / (edge1 - edge0), 0), 1)
  return u * u * (3 - 2 * u)
}

function easeOutQuint(u: number): number {
  return 1 - Math.pow(1 - u, 5)
}

function easeInOut(u: number): number {
  return u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2
}
