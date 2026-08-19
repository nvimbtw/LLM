"use client"

import * as React from "react"

import { cn } from "@/lib/utils"

/**
 * The trellis — one loading indicator for the whole site.
 *
 * A hexagonal field of nineteen dots, and a tree grown across it. Ink leaves
 * the centre, runs along lattice edges and branches; every dot it reaches
 * lights and swells a little, while the dots it has not reached go on breathing
 * underneath. That is the entire figure.
 *
 * **The tree is grown, not drawn.** A fresh one is grown by random walk at the
 * start of every pass, so the mark is never the same twice. What stays fixed is
 * the *character* of the growth, and that is what carries the state: one long
 * wandering route is a token being sampled, a broad shallow bush is several
 * paths held open at once, the full spanning net is every parameter moving
 * together. Eight states are eight recognisably different pictures rather than
 * one picture at eight speeds, which is what every earlier version of this
 * component failed at.
 *
 * Three rules survive from those earlier versions, each learned the hard way:
 *
 *  1. **Geometry is built once at mount.** Every lattice edge is rendered up
 *     front and a tree is merely a choice of which are visible this pass, so a
 *     frame writes opacities and one dash length and never path data. An
 *     earlier version rebuilt paths per frame, and the browser re-parsing those
 *     is why it was smooth on one machine and stuttered on another.
 *  2. **The phase is integrated, never read off the clock.** Each frame adds
 *     `rate × dt`, and a mode change blends the *rate*, so the figure changes
 *     pace rather than jumping.
 *  3. **Growing allocates, so it happens on a pass boundary** — about once a
 *     second — and never inside a frame.
 *
 * One rule is deliberately *not* inherited. The old rule-and-segment mark
 * promised it would never re-start; this one re-starts on purpose, because
 * repeated growth is its whole language. A state change grows a new tree
 * immediately rather than waiting out the current one.
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
   * The resting colour — the field, and ink before it warms. Override where the
   * mark sits on a primary surface (e.g. a filled button) so it does not vanish
   * into it.
   */
  accent?: string
  /**
   * The warm colour ink turns as the work gets real.
   *
   * The pair has its own two tokens rather than borrowing `--primary`, and the
   * reason is measured: sharing the primary put the two ΔL* 8 apart in OKLab,
   * so they differed almost only in hue — the weakest way to tell two things
   * apart, and no way at all for some readers. If you override one, override
   * both, and keep the lightness gap.
   */
  spark?: string
}

const TAU = Math.PI * 2

/* ── The field ───────────────────────────────────────────────────────────────
   A hexagonal patch of a *triangular* lattice: three rings, nineteen dots.

   Triangular rather than square for two reasons that are not decoration. A
   square grid of dots is a figure other products already own. And a triangular
   lattice offers six directions out of every dot where a square offers four,
   which is what makes a line grown on it look grown rather than plotted.
*/

const RING = 2
const SPACING = 6
const DOT_R = 1.15
const ROW_H = Math.sqrt(3) / 2

interface Cell {
  q: number
  r: number
  x: number
  y: number
}

const place = (q: number, r: number): Cell => ({
  q,
  r,
  x: SPACING * (q + r / 2),
  y: SPACING * r * ROW_H,
})

const FIELD: Cell[] = (() => {
  const out: Cell[] = []
  for (let q = -RING; q <= RING; q++) {
    for (
      let r = Math.max(-RING, -RING - q);
      r <= Math.min(RING, RING - q);
      r++
    ) {
      out.push(place(q, r))
    }
  }
  return out
})()

const at = (q: number, r: number) =>
  FIELD.findIndex((cell) => cell.q === q && cell.r === r)

const ROOT = at(0, 0)

const DIRS: [number, number][] = [
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, 0],
  [-1, 1],
  [0, 1],
]

/** Six-way adjacency, by field index. */
const NEIGHBOURS: number[][] = FIELD.map((cell) =>
  DIRS.map(([dq, dr]) => at(cell.q + dq, cell.r + dr)).filter((i) => i >= 0)
)

/** Every lattice edge, once. Rendered at mount; reused by every tree. */
const EDGES: { a: number; b: number }[] = (() => {
  const out: { a: number; b: number }[] = []
  NEIGHBOURS.forEach((near, a) => {
    near.forEach((b) => {
      if (a < b) out.push({ a, b })
    })
  })
  return out
})()

const EDGE_INDEX = new Map(EDGES.map((edge, i) => [`${edge.a}-${edge.b}`, i]))
const edgeBetween = (a: number, b: number) =>
  EDGE_INDEX.get(a < b ? `${a}-${b}` : `${b}-${a}`) ?? -1

/**
 * The mark's footprint, in CSS pixels — everywhere, without exception. Callers
 * position it; they never size it.
 */
export const THROBBER_W = Math.round(SPACING * RING * 2 + DOT_R * 2 + 2)
export const THROBBER_H = Math.round(SPACING * RING * 2 * ROW_H + DOT_R * 2 + 2)
const CX = THROBBER_W / 2
const CY = THROBBER_H / 2

/** The field's own life: every dot breathing, out of phase by distance. */
function restingLevel(cell: Cell, phase: number): number {
  const d = Math.hypot(cell.x, cell.y) / (SPACING * RING)
  return 0.1 + 0.05 * (0.5 - 0.5 * Math.cos(TAU * (phase - d * 0.35)))
}

/* ── The states ──────────────────────────────────────────────────────────── */

/**
 * How fast each mode's phase advances, in passes per second. One place, so the
 * states can be compared as what they are: generating is five times idle, and
 * it should be.
 */
const RATES: Record<ThrobberMode, number> = {
  idle: 0.08,
  loading: 0.42,
  thinking: 0.26,
  generating: 1.1,
  tool: 0.85,
  training: 0.34,
  benchmarking: 0.3,
  tokenizing: 0.55,
}

/** How warm each mode runs. 0 is the resting colour, 1 the spark. */
const TINT: Record<ThrobberMode, number> = {
  idle: 0,
  loading: 0.35,
  thinking: 0.8,
  generating: 1,
  tool: 0.9,
  training: 0.5,
  benchmarking: 1,
  tokenizing: 0.55,
}

/**
 * How each state grows. `limbs` is how many directions leave the root, `branch`
 * how many an inner dot may split into, `depth` how far the ink travels —
 * three numbers that between them give a long thin route, a low bush, or a full
 * net.
 */
interface Growth {
  limbs: number
  branch: number
  depth: number
}

const GROWTH: Record<ThrobberMode, Growth> = {
  // Nothing grown: the root dot alone, breathing. A resting state has to look
  // present, not unmounted.
  idle: { limbs: 0, branch: 0, depth: 0 },
  loading: { limbs: 3, branch: 1, depth: 3 },
  // Broad and shallow, grown and then withdrawn — several possibilities held
  // open, none of them taken.
  thinking: { limbs: 4, branch: 2, depth: 2 },
  // One route, as long as the field allows: the closest this figure comes to
  // drawing a token being chosen.
  generating: { limbs: 1, branch: 1, depth: 5 },
  // Everything at once, for the two states that are not a path through
  // anything — a discrete operation, and bulk work on the whole body.
  tool: { limbs: 6, branch: 6, depth: 4 },
  training: { limbs: 6, branch: 6, depth: 4 },
  benchmarking: { limbs: 5, branch: 1, depth: 2 },
  tokenizing: { limbs: 3, branch: 2, depth: 2 },
}

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

/** Deterministic PRNG — a tree should be re-growable, not merely random. */
function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface Tree {
  /** Depth at which each edge is reached, or -1 if this tree omits it. */
  edgeDepth: Int8Array
  /** Depth at which each dot is reached, or -1. */
  dotDepth: Int8Array
  /** The deepest step any ink has to travel, for normalising the advance. */
  span: number
}

/** Grow one tree from the root by random walk. Runs on a pass boundary. */
function grow(rand: () => number, how: Growth): Tree {
  const edgeDepth = new Int8Array(EDGES.length).fill(-1)
  const dotDepth = new Int8Array(FIELD.length).fill(-1)
  dotDepth[ROOT] = 0

  let frontier = [ROOT]
  let span = 0

  for (let depth = 0; depth < how.depth && frontier.length > 0; depth++) {
    const next: number[] = []
    const width = depth === 0 ? how.limbs : how.branch

    for (const from of frontier) {
      // Shuffle this dot's free neighbours and take the first few. Shuffling
      // rather than picking in order is what stops the same limb winning every
      // pass, which is the whole reason the figure stays alive to look at.
      const options = NEIGHBOURS[from].filter((n) => dotDepth[n] < 0)
      for (let i = options.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1))
        ;[options[i], options[j]] = [options[j], options[i]]
      }
      for (const to of options.slice(0, width)) {
        // Re-check: a sibling in this same round may have claimed it.
        if (dotDepth[to] >= 0) continue
        dotDepth[to] = depth + 1
        edgeDepth[edgeBetween(from, to)] = depth
        next.push(to)
        span = depth + 1
      }
    }
    frontier = next
  }

  return { edgeDepth, dotDepth, span }
}

const REDUCED_MOTION = "(prefers-reduced-motion: reduce)"

// A frame this long was a background tab rather than a slow frame. Clamped so
// returning to the tab does not race the figure through a minute of growth.
const MAX_FRAME_S = 1 / 15

// How long a mode change takes to reach its new pace.
const BLEND_MS = 380

/**
 * Live-updating, SSR-safe read of the OS motion preference. A plain
 * `matchMedia(...).matches` is a snapshot: it never notices the setting change,
 * and on the server there is no `window` at all.
 */
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
  const edges = React.useRef<(SVGLineElement | null)[]>([])
  const dots = React.useRef<(SVGCircleElement | null)[]>([])

  // Everything the animation needs, kept in refs: the loop runs at 60fps while
  // a chat streams, and re-rendering React for it would make the throbber the
  // most expensive thing on the page.
  const tree = React.useRef<Tree | null>(null)
  const passStart = React.useRef(0)
  const passIndex = React.useRef(0)
  const grownFor = React.useRef<ThrobberMode | null>(null)
  const seed = React.useRef(0)

  const phase = React.useRef(0)
  const lastFrame = React.useRef(0)
  const rateFrom = React.useRef(RATES[mode])
  const rateTo = React.useRef(RATES[mode])
  const blendStart = React.useRef(0)
  const modeRef = React.useRef(mode)
  const colours = React.useRef({ accent, spark })

  // Bookkeeping in an effect rather than during render: a render can be
  // discarded or replayed under concurrent React, which would corrupt which
  // rate we think we are coming *from*.
  React.useEffect(() => {
    colours.current = { accent, spark }
  }, [accent, spark])

  React.useEffect(() => {
    if (modeRef.current === mode) return
    rateFrom.current = currentRate(rateFrom, rateTo, blendStart)
    rateTo.current = RATES[mode]
    blendStart.current = performance.now()
    modeRef.current = mode
    // Grow the new state's character immediately rather than waiting out the
    // pass in progress — feedback that lags a second is not feedback.
    grownFor.current = null
  }, [mode])

  React.useEffect(() => {
    let frame = 0

    const tick = (now: number) => {
      const dt = clamp(
        (now - (lastFrame.current || now)) / 1000,
        0,
        MAX_FRAME_S
      )
      lastFrame.current = now

      // Reduced motion slows the mark rather than stopping it. Stopping is the
      // obvious reading of the preference and the wrong one — this is a status
      // indicator, and a motionless one reports nothing. The guidance targets
      // large, vestibular-triggering movement; nothing here travels 30px.
      const damp = reduced ? 0.35 : 1
      phase.current += currentRate(rateFrom, rateTo, blendStart) * damp * dt

      const active = modeRef.current
      if (!seed.current) seed.current = (Math.random() * 2 ** 31) | 0

      // A new tree at the start of every pass, and on a state change.
      const local = phase.current - passStart.current
      if (grownFor.current !== active || local >= 1) {
        passStart.current = phase.current
        passIndex.current += 1
        grownFor.current = active
        tree.current = grow(
          mulberry32(seed.current ^ (passIndex.current * 2654435761)),
          GROWTH[active]
        )
      }

      draw(
        tree.current,
        clamp(phase.current - passStart.current, 0, 1),
        phase.current,
        active,
        colours.current,
        edges.current,
        dots.current
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
        {EDGES.map((edge, i) => (
          <line
            key={`${edge.a}-${edge.b}`}
            ref={(el) => {
              edges.current[i] = el
            }}
            x1={CX + FIELD[edge.a].x}
            y1={CY + FIELD[edge.a].y}
            x2={CX + FIELD[edge.b].x}
            y2={CY + FIELD[edge.b].y}
            // Dash lengths are then fractions of the edge, whatever its real
            // length is — and on a triangular lattice they are not all equal.
            pathLength={1}
            stroke={accent}
            strokeWidth={1.2}
            strokeLinecap="round"
            opacity={0}
          />
        ))}
        {FIELD.map((cell, i) => (
          <circle
            key={`${cell.q},${cell.r}`}
            ref={(el) => {
              dots.current[i] = el
            }}
            cx={CX + cell.x}
            cy={CY + cell.y}
            r={DOT_R}
            fill={accent}
            opacity={0.12}
          />
        ))}
      </svg>
    </span>
  )
}

/** Writes one frame straight to the DOM. */
function draw(
  tree: Tree | null,
  local: number,
  phase: number,
  mode: ThrobberMode,
  colours: { accent: string; spark: string },
  edges: (SVGLineElement | null)[],
  dots: (SVGCircleElement | null)[]
) {
  if (!tree) return

  const tint = TINT[mode]
  const colour = tint > 0.5 ? colours.spark : colours.accent
  const warmth = tint > 0.5 ? 1 : 0.9

  const full = tree.span + 1
  let reach: number
  let body = 1

  switch (mode) {
    case "idle":
      reach = 0
      break
    case "thinking":
      // Out and back: a shape considered rather than one built.
      reach = (0.5 - 0.5 * Math.cos(TAU * local)) * full
      break
    case "tool":
      reach = full
      body = Math.pow(0.5 + 0.5 * Math.cos(TAU * local), 3)
      break
    case "training":
      reach = full
      body = 0.4 + 0.5 * (0.5 - 0.5 * Math.cos(TAU * local))
      break
    case "benchmarking":
      // Whole limbs arriving one at a time, and held.
      reach = Math.floor(local * (full + 1))
      break
    default:
      // Grown across the pass, then cleared so the next one can differ.
      reach = local * (full + 0.6)
      body = 1 - smoothstep((local - 0.86) / 0.14)
  }

  for (let i = 0; i < EDGES.length; i++) {
    const el = edges[i]
    if (!el) continue
    const depth = tree.edgeDepth[i]
    if (depth < 0) {
      el.setAttribute("opacity", "0")
      continue
    }
    const drawn = clamp(reach - depth, 0, 1)
    el.setAttribute("stroke", colour)
    if (mode === "tokenizing") {
      // The grown tree, then cut: pieces travelling out along every twig.
      el.setAttribute("stroke-dasharray", "0.18 0.14")
      el.setAttribute("stroke-dashoffset", (-local * 0.32).toFixed(3))
      el.setAttribute("opacity", (drawn * warmth).toFixed(3))
    } else {
      el.setAttribute("stroke-dasharray", `${drawn.toFixed(3)} 1`)
      el.setAttribute("stroke-dashoffset", "0")
      el.setAttribute(
        "opacity",
        (drawn > 0 ? drawn * body * warmth : 0).toFixed(3)
      )
    }
  }

  for (let i = 0; i < FIELD.length; i++) {
    const el = dots[i]
    if (!el) continue
    const depth = tree.dotDepth[i]
    let lit =
      depth >= 0 && reach >= depth ? clamp(reach - depth + 1, 0, 1) * body : 0
    if (mode === "idle" && i === ROOT) {
      lit = 0.55 + 0.45 * (0.5 - 0.5 * Math.cos(TAU * local))
    }
    el.setAttribute("fill", lit > 0.02 ? colour : colours.accent)
    el.setAttribute("r", (DOT_R * (1 + 0.3 * lit)).toFixed(2))
    el.setAttribute(
      "opacity",
      (lit > 0.02 ? lit * warmth : restingLevel(FIELD[i], phase)).toFixed(3)
    )
  }
}

/** The rate right now, mid-blend between the mode we left and the one we are in. */
function currentRate(
  from: React.RefObject<number>,
  to: React.RefObject<number>,
  since: React.RefObject<number>
): number {
  const t = smoothstep((performance.now() - since.current) / BLEND_MS)
  return from.current + (to.current - from.current) * t
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi)
}

function smoothstep(x: number): number {
  const u = clamp(x, 0, 1)
  return u * u * (3 - 2 * u)
}
