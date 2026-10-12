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
 * the *character* of the growth, and that is what carries the state: one route
 * walked without stopping is a token being sampled, a broad shallow bush is
 * several paths held open at once, the full spanning net is every parameter
 * moving together. Eight states are eight recognisably different pictures
 * rather than one picture at eight speeds, which is what every earlier version
 * of this component failed at.
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
 *
 * **One state is not a tree.** `generating` is a *snake* — see the section of
 * that name. It grows nothing, never re-starts, and is the only state that
 * answers a mode change late.
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

/** Rings out from the centre — how far a dot is from home, in steps. */
const RADIUS: number[] = FIELD.map(
  ({ q, r }) => (Math.abs(q) + Math.abs(r) + Math.abs(q + r)) / 2
)

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
  // Not grown at all — see "The snake". `depth` is the only one of the three
  // numbers it still uses: the length of the body, in edges.
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

/* ── The snake ───────────────────────────────────────────────────────────────
   `generating` is the one state that grows nothing. A tree grown from the root
   and then cleared has a seam in it — the moment it starts over — and a token
   stream has no such moment: it is one act continuing. So this state walks. The
   head steps to a neighbouring dot, the tail lets the oldest one go, and the
   figure is the six dots in between, travelling.

   Stopping is the other half of it. A walk that vanished the instant a reply
   ended would cut; instead the head is sent home to the centre and the body
   drains in behind it, which is why this is the only state whose figure outlives
   its mode by half a second. What is left standing when it has drained is a lit
   dot at the centre — which is where every other state's first frame begins, so
   the figure that replaces it grows out of the light the snake left rather than
   striking up out of a dark field.
*/

/** The body, in edges — five, so six dots are lit. */
const BODY = GROWTH.generating.depth

/**
 * How many dots the head covers in one pass. Ties the walk to
 * `RATES.generating` instead of giving it a second rate to keep in sync: the
 * head travels its own length in the time every other state grows and clears a
 * whole tree.
 */
const STEPS_PER_PASS = BODY

/** How far ahead the walk is planned — over a pass of travel, plus the body. */
const AHEAD = STEPS_PER_PASS + BODY + 2

/** The dimmest a dot burns before the tail lets it go. */
const TAIL_LEVEL = 0.35

/**
 * The pace the run home settles at, in dots per second, and deliberately not
 * the blended rate: this is a full stop rather than a change of speed, and
 * blending toward `idle`'s would leave it crawling home for twenty seconds.
 * It is still eased into over `ACCELERATE` — rule 2 holds here too, it is only
 * the rate being blended *to* that is the snake's own.
 */
const HOMING_STEPS_PER_S = 18

/** Dots the run home spends getting up to that pace. */
const ACCELERATE = 1.2

interface Snake {
  /** Dots walked, oldest first. Trimmed behind and planned ahead per pass. */
  route: number[]
  /** Where the head is along that route — fractional, between two dots. */
  head: number
  /** Route index of the centre once it is heading home; -1 while crawling. */
  home: number
  /** Where the head was when the stop began, in the same units, so the run
   * home can ease out of the pace it was crawling at. */
  stoppedAt: number
}

const newSnake = (): Snake => ({
  route: [ROOT],
  head: 0,
  home: -1,
  stoppedAt: 0,
})

const homingPace = (snake: Snake) => {
  const crawl = RATES.generating * STEPS_PER_PASS
  const gathered = smoothstep((snake.head - snake.stoppedAt) / ACCELERATE)
  return crawl + (HOMING_STEPS_PER_S - crawl) * gathered
}

/** True once the last of the body has been drawn into the centre. */
const swallowed = (snake: Snake) =>
  snake.home >= 0 && snake.head - snake.home >= BODY

/**
 * The next dot the head walks into: adjacent, and not one its own body will
 * still be standing on when it arrives.
 */
function step(route: number[], rand: () => number): number {
  const from = route[route.length - 1]
  const body = route.slice(-BODY)
  const free = NEIGHBOURS[from].filter((cell) => !body.includes(cell))
  for (let i = free.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[free[i], free[j]] = [free[j], free[i]]
  }
  // A dot with no free neighbour of its own is a dead end one step early: take
  // it and the snake has to walk through itself on the step after this one.
  const open = free.find((cell) =>
    NEIGHBOURS[cell].some((n) => n !== from && !body.includes(n))
  )
  // Boxed in anyway — reachable at the rim, where a dot has three neighbours
  // rather than six. Clip the corner the body has stood on longest.
  return open ?? free[0] ?? stalest(NEIGHBOURS[from], route)
}

function stalest(near: number[], route: number[]): number {
  return near.reduce((a, b) =>
    route.lastIndexOf(b) < route.lastIndexOf(a) ? b : a
  )
}

/**
 * Drop what the tail has left behind and plan far enough ahead that no frame
 * has to. Both halves allocate, which is why this is a pass boundary's work.
 */
function plan(snake: Snake, rand: () => number) {
  const spent = Math.floor(snake.head) - BODY
  if (spent > 0) {
    snake.route.splice(0, spent)
    snake.head -= spent
  }
  while (snake.route.length < Math.floor(snake.head) + AHEAD) {
    snake.route.push(step(snake.route, rand))
  }
}

/**
 * Point the head at the centre. Everything the walk had planned past the dot it
 * is entering is dropped: this is a stop, not a detour.
 */
function sendHome(snake: Snake) {
  const entering = Math.min(Math.ceil(snake.head), snake.route.length - 1)
  snake.route.length = entering + 1

  let cell = snake.route[entering]
  let behind = snake.route[entering - 1] ?? -1
  while (cell !== ROOT) {
    const closer = NEIGHBOURS[cell].filter((n) => RADIUS[n] < RADIUS[cell])
    // Reversing into its own neck reads as a stumble; any other step that still
    // closes on the centre does not.
    const next = closer.find((n) => n !== behind) ?? closer[0]
    snake.route.push(next)
    behind = cell
    cell = next
  }
  snake.home = snake.route.length - 1
  snake.stoppedAt = snake.head
}

/** What is on screen this frame: where the head is, and how much body is still
 * behind it. */
interface Body {
  head: number
  span: number
}

function bodyOf(snake: Snake): Body {
  if (snake.home < 0) return { head: snake.head, span: BODY }
  // Home: the head stops at the centre but the walk does not, and the distance
  // between the two is the body being drawn in after it.
  const over = Math.max(0, snake.head - snake.home)
  return {
    head: Math.min(snake.head, snake.home),
    span: clamp(BODY - over, 0, BODY),
  }
}

/**
 * How brightly a dot burns, by how far behind the head it is. Bright at the
 * head and falling away behind, so a still frame says which way the snake is
 * travelling; the exponent holds the head end up and spends most of the fall on
 * the last two dots, where it reads as a tail rather than a fade.
 *
 * Either end is fractional: `arriving` is the dot the head is walking into,
 * `leaving` the one the tail is walking off. Only the tail end dims that way —
 * a body draining into the centre gets shorter from behind, and the head holds
 * its light until the state that replaces it takes the dot over.
 */
function bodyLevel(age: number, span: number): number {
  if (age < -1 || age > span) return 0
  const arriving = clamp(age + 1, 0, 1)
  const leaving = age <= 0 ? 1 : clamp(span - age, 0, 1)
  const taper =
    1 - (1 - TAIL_LEVEL) * Math.pow(clamp(age / (BODY - 1), 0, 1), 1.4)
  return arriving * taper * leaving
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

export function TrellisThrobber({
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
  // Non-null exactly while a snake is on screen, which outlasts `generating`
  // itself by the run home.
  const snake = React.useRef<Snake | null>(null)
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

    // The snake is the exception, in both directions: a stream that has stopped
    // is put away rather than cut, and one that resumes mid-way home simply
    // turns around. Nudging `passStart` back a whole pass makes the next frame
    // a boundary, which is where the route it now needs gets planned.
    if (!snake.current) return
    if (mode === "generating") {
      snake.current.home = -1
      passStart.current = phase.current - 1
    } else if (snake.current.home < 0) {
      sendHome(snake.current)
    }
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
      const rate = currentRate(rateFrom, rateTo, blendStart)
      phase.current += rate * damp * dt

      const active = modeRef.current
      if (!seed.current) seed.current = (Math.random() * 2 ** 31) | 0

      if (active === "generating" && !snake.current) {
        snake.current = newSnake()
        // A fresh walk is one dot with nowhere to go: make this frame a
        // boundary so it is planned now rather than a pass from now.
        passStart.current = phase.current - 1
      }

      const walking = snake.current
      if (walking) {
        // Crawling takes the blended rate, so the walk gathers pace as the
        // state arrives. The run home does not — see `HOMING_STEPS_PER_S`.
        const pace =
          walking.home < 0 ? rate * STEPS_PER_PASS : homingPace(walking)
        walking.head += pace * damp * dt

        if (walking.home < 0 && phase.current - passStart.current >= 1) {
          passStart.current = phase.current
          passIndex.current += 1
          plan(
            walking,
            mulberry32(seed.current ^ (passIndex.current * 2654435761))
          )
        }
        if (swallowed(walking)) snake.current = null
      }

      if (snake.current) {
        crawl(
          snake.current,
          bodyOf(snake.current),
          phase.current,
          colours.current,
          edges.current,
          dots.current
        )
      } else {
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
      }
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
      // `loading` and `tokenizing`: grown across the pass, then cleared so the
      // next one can differ. `generating` never arrives here — it crawls.
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

/** Writes one frame of the snake straight to the DOM. */
function crawl(
  snake: Snake,
  body: Body,
  phase: number,
  colours: { accent: string; spark: string },
  edges: (SVGLineElement | null)[],
  dots: (SVGCircleElement | null)[]
) {
  restField(phase, colours.accent, edges, dots)
  lightSnake(snake, body, colours.spark, edges, dots)
}

/** Everything dark but the field's own breathing — the snake is written over
 * the top of this, which is cheaper than tracking what it lit last frame. */
function restField(
  phase: number,
  accent: string,
  edges: (SVGLineElement | null)[],
  dots: (SVGCircleElement | null)[]
) {
  for (const el of edges) el?.setAttribute("opacity", "0")
  for (let i = 0; i < FIELD.length; i++) {
    const el = dots[i]
    if (!el) continue
    el.setAttribute("fill", accent)
    el.setAttribute("r", DOT_R.toFixed(2))
    el.setAttribute("opacity", restingLevel(FIELD[i], phase).toFixed(3))
  }
}

/** The lit dots, and the edges between them. `generating` is the warmest state
 * in `TINT`, so the snake is always ink rather than field colour. */
function lightSnake(
  snake: Snake,
  body: Body,
  spark: string,
  edges: (SVGLineElement | null)[],
  dots: (SVGCircleElement | null)[]
) {
  const tail = body.head - body.span
  const first = Math.max(0, Math.floor(tail))
  const last = Math.min(Math.ceil(body.head), snake.route.length - 1)

  for (let i = first; i <= last; i++) {
    const el = dots[snake.route[i]]
    const lit = bodyLevel(body.head - i, body.span)
    if (!el || lit <= 0.02) continue
    el.setAttribute("fill", spark)
    el.setAttribute("r", (DOT_R * (1 + 0.3 * lit)).toFixed(2))
    el.setAttribute("opacity", lit.toFixed(3))
  }

  for (let i = first; i < last; i++) {
    // How much of this edge the body covers, from the dot it was walked from.
    const lo = clamp(tail - i, 0, 1)
    const hi = clamp(body.head - i, 0, 1)
    if (hi - lo < 0.001) continue

    const index = edgeBetween(snake.route[i], snake.route[i + 1])
    const el = edges[index]
    if (!el) continue
    const lit = bodyLevel(body.head - i - (lo + hi) / 2, body.span)
    el.setAttribute("stroke", spark)
    el.setAttribute("stroke-dasharray", `${(hi - lo).toFixed(3)} 1`)
    // A line is drawn between its two dots in field order, which is not
    // necessarily the direction the snake crossed it: reversed, the same
    // stretch of body has to be measured from the far end.
    el.setAttribute(
      "stroke-dashoffset",
      (EDGES[index].a === snake.route[i] ? -lo : hi - 1).toFixed(3)
    )
    el.setAttribute("opacity", lit.toFixed(3))
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
