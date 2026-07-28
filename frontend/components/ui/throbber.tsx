"use client"

import * as React from "react"

import { cn } from "@/lib/utils"

/**
 * The polyhedron — one loading indicator for the whole site.
 *
 * A faceted ball turning in place: an icosphere, shaded by which way each triangle
 * faces. States differ in how fast it turns and in what the surface does — nothing,
 * a ripple running over it, the whole thing breathing, single faces stabbing
 * outward into spikes, or a charge running the seams of the mesh.
 *
 * This is the third design and the first one that can't stutter, which is the
 * reason for it. The two before — three ticks whose heights cycled, then an arc
 * travelling a ring — both animated a *pattern with a period*, and a period has an
 * end: however carefully the seam is made continuous, the eye tracks the motion,
 * notices it arrive back where it began, and reads that as the animation cutting
 * off and restarting. A rotating body has no such moment. It is at a different
 * orientation every frame it will ever be drawn at, so there is nothing to restart.
 *
 * Four things follow from that and are worth keeping:
 *
 *  1. **Angles are integrated, never computed from the clock.** Each frame adds
 *     `rate × dt` to the running yaw and pitch. A mode change therefore changes the
 *     *speed* the ball turns at, and the orientation carries straight through it —
 *     it is not possible for a state change to move the body, only to change how it
 *     is moving. `mode` swaps are a crossfade of the rates, so the ball spins up or
 *     slows down into its new behaviour rather than cutting.
 *  2. **Every state is the same body**, differing only in a vector of eight
 *     numbers (`Look`). Because a state is just numbers, any two mix by weight.
 *     Give one instance a changing `mode` rather than swapping instances.
 *  3. **States are told apart by motion and relief, not hue** — with one exception.
 *     `thinking` spends a second colour — purple — on a charge travelling the
 *     seams, because it is the one state where the message is that something is
 *     happening *inside* the body rather than to its surface, and no amount of
 *     surface movement says that. Everything else is one accent and a single light
 *     direction.
 *  4. **The facets are a constant size on screen, not a constant count.** The mark
 *     is drawn anywhere from 16px in a queue row to 120px on the showcase, and a
 *     fixed face count cannot serve both: twenty triangles are right at 16px and
 *     coarse at 120px, while 320 are an icosphere at 120px and an undifferentiated
 *     grey disc at 16px. So the body is subdivided to whichever level puts its
 *     edges nearest `TARGET_FACET_PX`, measured from the element's actual height.
 *     Growing the mark adds triangles rather than enlarging them, which is what
 *     makes it read as the same object seen closer.
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
   * Colour of the ball. Override where the mark already sits on a primary surface
   * (e.g. the send button) so it doesn't vanish.
   */
  accent?: string
  /**
   * Colour of the discharge that runs across the seams. The one place a second hue
   * is spent, so it has to be a hue nothing else in the mark uses: the default is
   * the purple `--throbber-spark`, against the blue-grey primary.
   */
  spark?: string
}

const TAU = Math.PI * 2

// Geometry, in viewBox units. The body leaves room around itself for `tool`'s
// spikes, which reach half again its radius.
const VIEW = 16
const CENTER = VIEW / 2
const RADIUS = 5

// A resting tilt, in turns, so the ball is never seen pole-on and the rotation is
// legible from the first frame.
const BASE_TILT = 0.07

// Direction the light comes from, in view space (+z is out of the screen). Slightly
// up and to the left, which is where every reader's mental light source is.
const LIGHT: Vec3 = normalize({ x: -0.35, y: 0.55, z: 0.76 })

// How long a mode change takes to cross-fade. Long enough to read as the ball
// changing pace, short enough that it never delays feedback.
const BLEND_MS = 420

// A frame long enough that it was probably a background tab rather than a slow
// frame. Clamped so returning to the tab doesn't teleport the ball through a
// minute's worth of rotation.
const MAX_FRAME_S = 1 / 15

/**
 * How big the mark is, in CSS pixels — everywhere, without exception.
 *
 * The mark used to be sized by its caller, which meant the ball had a different
 * radius and a different facet count in every row it appeared in, and no two of
 * them looked like the same object. One number instead: the body has one radius
 * and one facet size wherever it is drawn, and callers position it but never size
 * it. Change this and the whole app changes with it.
 *
 * 64px is where two constraints meet. It is small enough to sit in the chat
 * thread's status row beside a 14px label without the row becoming a panel, and
 * large enough that the surface still reads — a ripple, a breath and a spike are
 * all distinguishable at this size, which they are not much below it. It also puts
 * the facets almost exactly on `TARGET_FACET_PX`, so the body is at the density the
 * subdivision was chosen for rather than at the edge of a rounding decision.
 */
export const THROBBER_PX = 64

// Which subdivision `THROBBER_PX` calls for: the level whose facet edges land
// nearest this, in CSS pixels. Fixed rather than measured, now that the size is.
const TARGET_FACET_PX = 5.5

interface Vec3 {
  x: number
  y: number
  z: number
}

/**
 * The interpolable state vector. Every field is a plain number precisely so that
 * any two modes can be mixed by weight — and the two rates are *rates*, which is
 * what lets a mode change alter the motion without moving the body.
 */
interface Look {
  /** Turns per second about the vertical axis. */
  spin: number
  /** Turns per second about the horizontal axis, so it never reads as a barrel. */
  tumble: number
  /** How fast the surface pattern advances, in cycles per second. */
  rate: number
  /** Outward displacement of every face at once, as a fraction of the radius. */
  swell: number
  /** Outward displacement of the faces the pattern is currently touching. */
  spike: number
  /** Share of the surface the pattern touches at once — narrow = one face at a time. */
  width: number
  /** Overall opacity of the body. */
  o: number
  /** Brightness of the discharge running across the seams, 0 = none. */
  seam: number
  /** How fast that discharge travels, in cycles per second. */
  seamRate: number
  /** How much of the mesh it lights at once — narrow = a spark, wide = a glow. */
  seamWidth: number
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
  const facesRef = React.useRef<(SVGPolygonElement | null)[]>([])
  const seamsRef = React.useRef<(SVGPolygonElement | null)[]>([])
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

  // The body's running orientation and surface phase. These persist across mode
  // changes and across the effect being torn down and rebuilt, which is the point:
  // they are the state of a thing that has been turning since the page loaded, not
  // the progress of an animation that plays.
  const yaw = React.useRef(0)
  const pitch = React.useRef(0)
  const pulse = React.useRef(0)
  const seamPhase = React.useRef(0)
  const lastFrame = React.useRef(0)
  const staggered = React.useRef(false)

  React.useEffect(() => {
    // Each instance starts at its own orientation, so the several marks a page can
    // show at once (the sidebar, the queue's now-band, a job header) don't turn in
    // lockstep and read as one animation cloned. Claimed here rather than at render
    // because taking a number from a shared counter is a side effect, and doing it
    // during render would also make the first frame differ between server and
    // client for no benefit — nothing about the stagger reaches the markup.
    if (!staggered.current) {
      staggered.current = true
      yaw.current = nextStagger()
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
      // targets large, vestibular-triggering movement, so a slow turn honours the
      // intent while the mark still indicates.
      if (reduced) look = mix(look, LOOKS.idle, 0.65)

      yaw.current += look.spin * dt
      pitch.current += look.tumble * dt
      pulse.current += look.rate * dt
      seamPhase.current += look.seamRate * dt

      draw(
        look,
        yaw.current,
        pitch.current,
        pulse.current,
        seamPhase.current,
        facesRef.current,
        seamsRef.current
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
        {FACES.map((_, i) => (
          <polygon
            key={i}
            ref={(node) => {
              facesRef.current[i] = node
            }}
            fill={accent}
            fillOpacity={0}
            // Each facet is filled *and* outlined in the accent. Drawing the seam
            // per face rather than as one deduplicated wireframe means a spike
            // carries its own outline out with it — and it is what the discharge
            // layer below needs, since that has to light one facet's seams and not
            // its neighbours'.
            stroke={accent}
            strokeWidth={0.75}
            strokeOpacity={0}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        ))}
        {/* The discharge, over the top of the body so a lit seam is never buried
            under the fill of the facet in front of it. Its own layer rather than a
            recolouring of the strokes above, because a seam that is *both* structure
            and signal would have to switch hue mid-fade, and the switch is visible
            at the sizes this is drawn at. */}
        {FACES.map((_, i) => (
          <polygon
            key={i}
            ref={(node) => {
              seamsRef.current[i] = node
            }}
            fill="none"
            stroke={spark}
            strokeWidth={1}
            strokeOpacity={0}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
    </span>
  )
}

// Successive starting orientations, in turns. Stepping by the golden ratio spreads
// any number of instances about as evenly as they can be spread, without needing to
// know in advance how many there will be.
let stagger = 0
function nextStagger(): number {
  stagger = fract(stagger + 0.6180339887)
  return stagger
}

// ── the body ────────────────────────────────────────────────────────────────

// The body, subdivided to whatever `THROBBER_PX` calls for. Built once at module
// load and shared by every instance: the geometry is the same ball everywhere, and
// only the orientation differs.
//
// Two levels of subdivision — 320 faces — is the ceiling on purpose. The next
// would be 1,280 polygons updated per frame per instance, and several pages show
// three of these at once.
const MAX_SUBDIVISIONS = 2
const FACES: Vec3[][] = buildBody()

function buildBody(): Vec3[][] {
  let faces = buildIcosahedron()
  // The body's radius on screen: the viewBox is `VIEW` tall and the ball `RADIUS`.
  const radiusPx = (THROBBER_PX * RADIUS) / VIEW
  for (let level = 0; level < MAX_SUBDIVISIONS; level += 1) {
    // Subdivide while it takes the facets *closer* to the target, not merely while
    // they are above it — halving an edge from 7px to 3.5px against a 5.5px target
    // overshoots by more than stopping would.
    const edge = distance(faces[0][0], faces[0][1]) * radiusPx
    if (Math.abs(edge / 2 - TARGET_FACET_PX) >= Math.abs(edge - TARGET_FACET_PX)) {
      break
    }
    faces = subdivide(faces)
  }
  return faces
}

// Each face's outward normal, which for a body inscribed in a sphere is just its
// centroid direction. Precomputed alongside the faces: it is needed twice a frame
// (culling, then shading) and never changes in body space.
const NORMALS: Vec3[] = FACES.map((face) => normalize(centroid(face)))

// Splits every triangle into four and pushes the new midpoints out onto the unit
// sphere. Winding is preserved, so the outward normals stay outward.
function subdivide(faces: Vec3[][]): Vec3[][] {
  const out: Vec3[][] = []
  for (const [a, b, c] of faces) {
    const ab = normalize(midpoint(a, b))
    const bc = normalize(midpoint(b, c))
    const ca = normalize(midpoint(c, a))
    out.push([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca])
  }
  return out
}

function buildIcosahedron(): Vec3[][] {
  const g = (1 + Math.sqrt(5)) / 2
  const v: Vec3[] = [
    { x: -1, y: g, z: 0 },
    { x: 1, y: g, z: 0 },
    { x: -1, y: -g, z: 0 },
    { x: 1, y: -g, z: 0 },
    { x: 0, y: -1, z: g },
    { x: 0, y: 1, z: g },
    { x: 0, y: -1, z: -g },
    { x: 0, y: 1, z: -g },
    { x: g, y: 0, z: -1 },
    { x: g, y: 0, z: 1 },
    { x: -g, y: 0, z: -1 },
    { x: -g, y: 0, z: 1 },
  ].map(normalize)

  // Wound counter-clockwise seen from outside, so a face's normal points away from
  // the centre and back-face culling is a sign test.
  const indices = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ]
  return indices.map((face) => face.map((i) => v[i]))
}

/**
 * Writes one frame straight to the DOM. Deliberately imperative: this runs at 60fps
 * while the chat is streaming, and re-rendering React for it would make the throbber
 * the most expensive thing on the page.
 *
 * Faces are drawn where they fall, without depth sorting. For a convex body under
 * an orthographic projection, front-facing polygons cannot overlap each other —
 * culling is the whole of the hidden-surface problem here.
 */
function draw(
  look: Look,
  yaw: number,
  pitch: number,
  pulse: number,
  seamPhase: number,
  nodes: (SVGPolygonElement | null)[],
  seams: (SVGPolygonElement | null)[]
) {
  const a = yaw * TAU
  const b = (BASE_TILT + pitch) * TAU
  const swell = look.swell * (0.5 + 0.5 * Math.sin(TAU * pulse))

  for (let i = 0; i < FACES.length; i += 1) {
    const node = nodes[i]
    const seam = seams[i]
    if (!node) continue

    const normal = rotate(NORMALS[i], a, b)
    // Back-facing, so it is on the far side of the body and nothing about it is
    // visible. The small positive threshold drops faces exactly edge-on, which
    // project to a line and read as a stray hairline.
    if (normal.z <= 0.04) {
      node.setAttribute("fill-opacity", "0")
      node.setAttribute("stroke-opacity", "0")
      seam?.setAttribute("stroke-opacity", "0")
      continue
    }

    const lift = 1 + swell + look.spike * relief(NORMALS[i], i, pulse, look.width)
    let points = ""
    for (const vertex of FACES[i]) {
      const p = rotate(vertex, a, b)
      const x = CENTER + p.x * RADIUS * lift
      const y = CENTER - p.y * RADIUS * lift
      points += `${x.toFixed(2)},${y.toFixed(2)} `
    }
    node.setAttribute("points", points)

    // One light, one term. The facing already varies from face to face, so this is
    // all the shading needed for the relief to read.
    const lambert = Math.max(0, dot(normal, LIGHT))
    const shade = 0.05 + 0.42 * Math.pow(lambert, 1.3)
    node.setAttribute("fill-opacity", `${(shade * look.o).toFixed(3)}`)
    // The edge is what keeps the facets crisp once the fills get faint near the
    // silhouette. Drawn per face, so a spike carries its own outline out with it.
    node.setAttribute(
      "stroke-opacity",
      `${(0.28 * look.o * (0.35 + 0.65 * normal.z)).toFixed(3)}`
    )

    if (!seam) continue
    // The discharge shares the facet's projected outline, so it lights that
    // triangle's three seams exactly. Faded toward the silhouette like the seam
    // under it, or a facet turning away flares as brightly as one facing us.
    const charge =
      look.seam > 0
        ? relief(NORMALS[i], i, seamPhase, look.seamWidth) *
          look.seam *
          (0.45 + 0.55 * normal.z)
        : 0
    if (charge > 0.004) {
      seam.setAttribute("points", points)
      seam.setAttribute("stroke-opacity", charge.toFixed(3))
    } else {
      seam.setAttribute("stroke-opacity", "0")
    }
  }
}

/**
 * How far face `i` is currently pushed out, 0..1.
 *
 * The pattern is a band sweeping the body from pole to pole, scattered a little so
 * the faces at one latitude don't all fire in unison. `width` is what makes the
 * same function serve every state: wide, and most of the surface swells at once
 * into a ripple; narrow, and one face at a time stabs out and withdraws.
 */
function relief(
  normal: Vec3,
  i: number,
  pulse: number,
  width: number
): number {
  if (width <= 0) return 0
  // Body-space latitude, so the band is fixed to the ball and turns with it rather
  // than sweeping the screen independently of the rotation.
  const latitude = normal.y
  // The golden ratio spreads the faces evenly over the interval without them
  // landing in any order the eye can anticipate.
  const scatter = ((i * 0.6180339887) % 1) * 0.3
  const p = fract(pulse - latitude * 0.5 + scatter)
  const distance = Math.min(p, 1 - p)
  return smoothstep(width, 0, distance)
}

// ── the states ──────────────────────────────────────────────────────────────

const LOOKS: Record<ThrobberMode, Look> = {
  // idle — barely turning, no relief, dim. Present rather than working: the ball
  // is still there and still moving, which is what stops "nothing is happening"
  // from looking like "this component unmounted".
  idle: {
    spin: 0.02, tumble: 0.006, rate: 0, swell: 0, spike: 0, width: 0,
    o: 0.5, seam: 0, seamRate: 0, seamWidth: 0,
  },

  // loading — a steady, unhurried turn and a smooth surface. The plainest moving
  // state and the one that has to be instantly legible: nothing is being decided or
  // produced yet, something is simply on its way.
  loading: {
    spin: 0.13, tumble: 0.02, rate: 0, swell: 0, spike: 0, width: 0,
    o: 0.9, seam: 0, seamRate: 0, seamWidth: 0,
  },

  // thinking — the body is nearly still and the *seams* carry the state: a charge
  // runs the mesh, lighting a facet's three edges and passing to the next, in the
  // one second colour the mark spends. Neurons firing, not a ball turning — which
  // is why the relief is almost nothing here and the rotation is the slowest of any
  // working state. Something is happening inside it rather than to it.
  thinking: {
    spin: 0.045, tumble: 0.03, rate: 0.28, swell: 0.015, spike: 0.03, width: 0.34,
    o: 0.85, seam: 1, seamRate: 0.75, seamWidth: 0.055,
  },

  // generating — fast, with a tight ripple running over the body. The speed and the
  // travelling relief together are what make it read as output being produced
  // rather than waited for.
  generating: {
    spin: 0.34, tumble: 0.03, rate: 1.5, swell: 0.02, spike: 0.11, width: 0.16,
    o: 1, seam: 0, seamRate: 0, seamWidth: 0,
  },

  // tool — single faces stab out hard and withdraw, on a body that is otherwise
  // turning slowly. The one state with real spikes, and the only one whose relief
  // is meant to feel abrupt: a discrete operation, not progress.
  tool: {
    spin: 0.09, tumble: 0.02, rate: 0.85, swell: 0, spike: 0.45, width: 0.055,
    o: 1, seam: 0, seamRate: 0, seamWidth: 0,
  },

  // training — the whole ball breathes, every face out and back together, while it
  // turns steadily. Uniform relief is the signature: unlike every other state
  // nothing travels across the surface, because nothing here is being handed along.
  // It is bulk work on the whole thing at once.
  training: {
    spin: 0.15, tumble: 0.05, rate: 0.3, swell: 0.16, spike: 0.03, width: 0.5,
    o: 0.95, seam: 0, seamRate: 0, seamWidth: 0,
  },

  // benchmarking — one face at a time lifts, holds and settles, the body barely
  // turning under it. The deliberate counterpart to `generating`'s ripple: the same
  // travelling band, narrowed until it picks out a single facet, which reads as
  // taking one reading after another.
  benchmarking: {
    spin: 0.06, tumble: 0.015, rate: 0.5, swell: 0, spike: 0.34, width: 0.04,
    o: 1, seam: 0, seamRate: 0, seamWidth: 0,
  },
}

// ── maths ───────────────────────────────────────────────────────────────────

// Yaw about the vertical axis, then pitch about the horizontal one. Two rotations
// rather than a matrix: there are only ever two angles, and the matrix would be
// rebuilt every frame to save four multiplications.
function rotate(v: Vec3, yaw: number, pitch: number): Vec3 {
  const cy = Math.cos(yaw)
  const sy = Math.sin(yaw)
  const x = v.x * cy + v.z * sy
  const z1 = -v.x * sy + v.z * cy

  const cp = Math.cos(pitch)
  const sp = Math.sin(pitch)
  return { x, y: v.y * cp - z1 * sp, z: v.y * sp + z1 * cp }
}

function mix(a: Look, b: Look, w: number): Look {
  const lerp = (x: number, y: number) => x + (y - x) * w
  return {
    spin: lerp(a.spin, b.spin),
    tumble: lerp(a.tumble, b.tumble),
    rate: lerp(a.rate, b.rate),
    swell: lerp(a.swell, b.swell),
    spike: lerp(a.spike, b.spike),
    width: lerp(a.width, b.width),
    o: lerp(a.o, b.o),
    seam: lerp(a.seam, b.seam),
    seamRate: lerp(a.seamRate, b.seamRate),
    seamWidth: lerp(a.seamWidth, b.seamWidth),
  }
}

function centroid(face: Vec3[]): Vec3 {
  return face.reduce(
    (sum, v) => ({
      x: sum.x + v.x / face.length,
      y: sum.y + v.y / face.length,
      z: sum.z + v.z / face.length,
    }),
    { x: 0, y: 0, z: 0 }
  )
}

function midpoint(a: Vec3, b: Vec3): Vec3 {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 }
}

function distance(a: Vec3, b: Vec3): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)
}

function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(v.x, v.y, v.z) || 1
  return { x: v.x / length, y: v.y / length, z: v.z / length }
}

function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z
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
