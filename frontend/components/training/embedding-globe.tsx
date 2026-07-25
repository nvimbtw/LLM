"use client"

import * as React from "react"
import { Pause, Play } from "lucide-react"

import { Button } from "@/components/ui/button"
import { EMB_COLOR, embRankColor } from "@/components/training/chart-theme"
import { FigurePlaceholder } from "@/components/training/stat-run"

type Vec3 = [number, number, number]
type Face = [number, number, number]

// Convex hull of the point set as a triangle list, each triple oriented so its
// winding is counter-clockwise seen from *outside* the body. Brute force: a triangle
// (i,j,k) is a hull face iff every other point lies on one side of its plane. N is
// tiny (≤ ANISOTROPY_TOKENS = 32) and this only reruns when a new report lands, so
// the O(N⁴) sweep is nothing. Coplanar faces yield a few overlapping triangles —
// harmless under the translucent fill. Returns [] when there's no volume (N < 4 or
// everything collapses onto a plane), and the caller falls back to points + edges.
function convexHull(pts: Vec3[]): Face[] {
  const n = pts.length
  if (n < 4) return []

  // Centroid, for orienting each face's normal outward.
  const c: Vec3 = [0, 0, 0]
  for (const p of pts) {
    c[0] += p[0]
    c[1] += p[1]
    c[2] += p[2]
  }
  c[0] /= n
  c[1] /= n
  c[2] /= n

  // Tolerance scaled to the cloud's extent so near-coplanar points don't spuriously
  // split a face (or reject one) on floating-point noise.
  let ext = 0
  for (const p of pts) ext = Math.max(ext, Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]))
  const tol = (ext || 1) * 1e-3

  const faces: Face[] = []
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        const ux = pts[j][0] - pts[i][0]
        const uy = pts[j][1] - pts[i][1]
        const uz = pts[j][2] - pts[i][2]
        const vx = pts[k][0] - pts[i][0]
        const vy = pts[k][1] - pts[i][1]
        const vz = pts[k][2] - pts[i][2]
        let nx = uy * vz - uz * vy
        let ny = uz * vx - ux * vz
        let nz = ux * vy - uy * vx
        const nl = Math.hypot(nx, ny, nz)
        if (nl < tol) continue // collinear triple — no plane
        nx /= nl
        ny /= nl
        nz /= nl

        let pos = false
        let neg = false
        for (let m = 0; m < n; m++) {
          if (m === i || m === j || m === k) continue
          const d =
            nx * (pts[m][0] - pts[i][0]) +
            ny * (pts[m][1] - pts[i][1]) +
            nz * (pts[m][2] - pts[i][2])
          if (d > tol) pos = true
          else if (d < -tol) neg = true
          if (pos && neg) break
        }
        if (pos && neg) continue // points straddle the plane — interior triangle

        // Orient outward: flip winding when the normal points toward the centroid.
        const outward =
          nx * (pts[i][0] - c[0]) + ny * (pts[i][1] - c[1]) + nz * (pts[i][2] - c[2])
        faces.push(outward < 0 ? [i, k, j] : [i, j, k])
      }
    }
  }
  return faces
}

const normalize3 = (v: Vec3): Vec3 => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / l, v[1] / l, v[2] / l]
}

// Orientation is a 3×3 rotation matrix (row-major, column-vector convention: the
// screen point is M·p). A matrix — not Euler yaw/pitch — is what makes the drag feel
// right from every angle: each drag is composed as a rotation about the *screen* axes
// and pre-multiplied onto M, so "drag right" always turns the visible face right, even
// after the body has been tumbled upside down (where Euler angles reverse).
type Mat3 = [number, number, number, number, number, number, number, number, number]

const applyMat = (m: Mat3, p: Vec3): Vec3 => [
  m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
  m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
  m[6] * p[0] + m[7] * p[1] + m[8] * p[2],
]

// a·b
const matMul = (a: Mat3, b: Mat3): Mat3 => {
  const out = [0, 0, 0, 0, 0, 0, 0, 0, 0] as Mat3
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c]
  return out
}

// Rotation about the screen's vertical (y) and horizontal (x) axes.
const rotY = (a: number): Mat3 => {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [c, 0, s, 0, 1, 0, -s, 0, c]
}
const rotX = (a: number): Mat3 => {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [1, 0, 0, 0, c, -s, 0, s, c]
}

// A spherical mesh ready to draw: a unit direction and a display radius per vertex,
// plus the triangle list. Vertices 0..tokenCount-1 are the original tokens (they get
// a coloured dot); anything past that is a subdivision midpoint — geometry only.
// `tokenEdges` are the coarse token-to-token connections (the pre-subdivision hull
// edges): the only lines actually worth drawing, so the surface stays legible instead
// of a mess of subdivision seams.
interface Mesh {
  dirs: Vec3[]
  radii: number[]
  faces: Face[]
  tokenEdges: [number, number][]
  tokenCount: number
}

// Turn the token cloud into an icosphere. Each token becomes a point on the unit
// sphere in the direction its (whitened) embedding points, so what shows is the
// *directional* spread of the frequent tokens — a genuine ball rather than the raw
// PCA sliver. Because every point lies on the sphere, the convex hull is a full
// spherical triangulation (no token is buried inside), and subdividing each face
// toward the sphere rounds the flats into the familiar faceted icosphere. A gentle
// per-vertex radius bump keeps the surface from being perfectly featureless: a token
// that sits far from the average bulges out slightly.
function buildIcosphere(pts: Vec3[], subdivisions: number): Mesh {
  const tokenCount = pts.length
  const mags = pts.map((p) => Math.hypot(p[0], p[1], p[2]))
  const maxMag = Math.max(1e-6, ...mags)
  const dirs: Vec3[] = pts.map(normalize3)
  const radii: number[] = mags.map((m) => 0.85 + 0.15 * (m / maxMag))

  let faces = convexHull(dirs)
  if (faces.length === 0) return { dirs, radii, faces, tokenEdges: [], tokenCount }

  // The coarse hull's edges, deduped — every vertex here is still a token, so these
  // are exactly the token-to-token links we'll draw over the smooth surface.
  const tokenEdges: [number, number][] = []
  const seenEdge = new Set<number>()
  for (const [a, b, c] of faces) {
    for (const [i, j] of [
      [a, b],
      [b, c],
      [c, a],
    ]) {
      const lo = Math.min(i, j)
      const hi = Math.max(i, j)
      const key = lo * 1e6 + hi
      if (!seenEdge.has(key)) {
        seenEdge.add(key)
        tokenEdges.push([lo, hi])
      }
    }
  }

  // Each pass splits every triangle into four; new corners are edge midpoints pushed
  // back onto the sphere (radius averaged), deduped per shared edge so the mesh stays
  // watertight.
  for (let s = 0; s < subdivisions; s++) {
    const mid = new Map<number, number>()
    const edge = (a: number, b: number): number => {
      const key = a < b ? a * 1e6 + b : b * 1e6 + a
      const hit = mid.get(key)
      if (hit != null) return hit
      const idx = dirs.length
      dirs.push(normalize3([dirs[a][0] + dirs[b][0], dirs[a][1] + dirs[b][1], dirs[a][2] + dirs[b][2]]))
      radii.push((radii[a] + radii[b]) / 2)
      mid.set(key, idx)
      return idx
    }
    const next: Face[] = []
    for (const [a, b, c] of faces) {
      const ab = edge(a, b)
      const bc = edge(b, c)
      const ca = edge(c, a)
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca])
    }
    faces = next
  }
  return { dirs, radii, faces, tokenEdges, tokenCount }
}

// Respect the OS "reduce motion" setting: the globe holds still (no auto-spin) and
// the viewer can still nudge it by hand or press play.
function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState(false)
  React.useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)")
    const sync = () => setReduced(mq.matches)
    sync()
    mq.addEventListener("change", sync)
    return () => mq.removeEventListener("change", sync)
  }, [])
  return reduced
}

export interface EmbeddingGlobeProps {
  // Per-report 3-D PCA projections; the latest frame's geometry is the body shown.
  frames: Vec3[][]
  // `full` is the bordered, interactive square beside the biplot; `mini` is the
  // small quietly-spinning ornament in the hero (no chrome, no controls).
  variant?: "full" | "mini"
  className?: string
}

// The embedding globe: the frequent tokens' embeddings projected to the top-3
// principal axes and laid onto a sphere by direction — a rotating icosphere whose
// surface vertices are the tokens. It's the biplot's third dimension made literal,
// spun so you can read how the frequent tokens' directions spread (tokens that point
// alike sit close on the sphere). Hand-rolled SVG with a painter's-order face sort;
// the body auto-rotates and the full variant can be grabbed to orbit.
export function EmbeddingGlobe({ frames, variant = "full", className }: EmbeddingGlobeProps) {
  const mini = variant === "mini"
  // Memoized so the `?? []` fallback doesn't hand downstream memos a fresh array each render.
  const latest = React.useMemo(() => frames[frames.length - 1] ?? [], [frames])
  const n = latest.length

  // PCA scales each axis by its singular value, so PC1 dwarfs PC3 and the raw cloud is
  // a thin sliver, not a body. Whiten it — divide each axis by its spread (RMS) across
  // the whole run — so the three axes contribute equally and the hull reads as a
  // rounded, roughly-spherical solid. Real clustering still shows as a lopsided ball;
  // only the artificial PCA elongation is removed. RMS is taken over every frame so the
  // normalization is stable as the embeddings grow rather than rescaling each report.
  const axisScale = React.useMemo(() => {
    let sx = 0
    let sy = 0
    let sz = 0
    let count = 0
    for (const f of frames)
      for (const [x, y, z] of f) {
        sx += x * x
        sy += y * y
        sz += z * z
        count++
      }
    const rms = (s: number) => Math.sqrt(s / Math.max(1, count)) || 1
    return [rms(sx), rms(sy), rms(sz)] as const
  }, [frames])

  // The whitened vertices of the latest frame — the geometry everything else reads.
  const pts = React.useMemo(
    () =>
      latest.map(
        ([x, y, z]) =>
          [x / axisScale[0], y / axisScale[1], z / axisScale[2]] as Vec3
      ),
    [latest, axisScale]
  )

  // The icosphere mesh (rebuilt only when the geometry changes, not every spin tick).
  // Both variants get one subdivision so the small hero globe reads as the same
  // rounded body as the large one — it's just drawn smaller.
  const mesh = React.useMemo(() => buildIcosphere(pts, 1), [pts])

  // Vertex positions in the normalized sphere space (|v| ≤ 1). Depend only on the
  // mesh, so rotation doesn't rebuild them.
  const verts = React.useMemo(
    () => mesh.dirs.map((d, i): Vec3 => [d[0] * mesh.radii[i], d[1] * mesh.radii[i], d[2] * mesh.radii[i]]),
    [mesh]
  )

  const reduced = usePrefersReducedMotion()
  const [playing, setPlaying] = React.useState(!reduced)

  // Orientation is state (a rotation matrix) so the render stays a pure function of it;
  // the spin is just setState in a rAF loop. Seeded to a pleasant three-quarter tilt.
  const [orient, setOrient] = React.useState<Mat3>(() =>
    matMul(rotX(-0.42), rotY(0.6))
  )
  const [dragging, setDragging] = React.useState(false)

  // Angular velocity (rad/s) about the screen vertical (velY, the turntable spin) and
  // horizontal (velX, the top-over-bottom tumble). A drag injects both; friction then
  // bleeds velX back to rest and velY back to the base rate, so a flick in any
  // direction coasts and eases off instead of stopping dead.
  const velY = React.useRef(mini ? 0.55 : 0.32)
  const velX = React.useRef(0)
  const drag = React.useRef<{ x: number; y: number; t: number } | null>(null)
  // Read inside the animation loop (which owns no deps), kept in sync here.
  const playingRef = React.useRef(playing)
  React.useEffect(() => {
    playingRef.current = playing
  }, [playing])

  React.useEffect(() => {
    const spin = mini ? 0.55 : 0.32 // base rate — a slow, showroom turn
    let raf = 0
    let prev = performance.now()
    const step = (now: number) => {
      const dt = Math.min(0.05, (now - prev) / 1000)
      prev = now
      // The hand drives the body while grabbed; otherwise coast, decaying the current
      // velocity toward the base rate (0 when paused) so a fling eases back smoothly.
      if (!drag.current) {
        const base = playingRef.current ? spin : 0
        const k = 1 - Math.exp(-2.2 * dt)
        velY.current += (base - velY.current) * k // toward the turntable spin
        velX.current += (0 - velX.current) * k // toward rest (no vertical base spin)
        if (Math.abs(velY.current) > 1e-4 || Math.abs(velX.current) > 1e-4) {
          // Both axes in screen space, so the coast reads the same as the drag did.
          const delta = matMul(rotX(velX.current * dt), rotY(velY.current * dt))
          setOrient((m) => matMul(delta, m))
        }
      }
      raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [mini])

  // Drag-to-orbit (full variant): grab the body and swing it. Releasing hands the
  // swing's velocity to the coast loop above, so it keeps spinning and eases back to
  // the natural turn instead of stopping dead.
  const onPointerDown = (e: React.PointerEvent) => {
    if (mini) return
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    drag.current = { x: e.clientX, y: e.clientY, t: performance.now() }
    velY.current = 0
    velX.current = 0
    setDragging(true)
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const dx = (e.clientX - d.x) * 0.01
    const dy = (e.clientY - d.y) * 0.01
    const now = performance.now()
    const dtMove = (now - d.t) / 1000
    d.x = e.clientX
    d.y = e.clientY
    d.t = now
    // Track both fling velocities (smoothed) so a flick in any direction carries.
    if (dtMove > 0) {
      velY.current = 0.6 * velY.current + 0.4 * (dx / dtMove)
      velX.current = 0.6 * velX.current + 0.4 * (dy / dtMove)
    }
    // Compose the drag as screen-space rotations (about the view's x/y axes) and
    // pre-multiply — this is what keeps "right is right" and "up is up" at every
    // orientation, and it tumbles freely with no wall at the poles.
    const delta = matMul(rotX(dy), rotY(dx))
    setOrient((m) => matMul(delta, m))
  }
  const onPointerUp = () => {
    const d = drag.current
    drag.current = null
    setDragging(false)
    // A pause before release means no fling; otherwise clamp the flick to a sane speed.
    const clamp = (v: number) => Math.max(-8, Math.min(8, v))
    if (!d || performance.now() - d.t > 100) {
      velY.current = 0
      velX.current = 0
    } else {
      velY.current = clamp(velY.current)
      velX.current = clamp(velX.current)
    }
  }

  if (n < 2) {
    return (
      <FigurePlaceholder
        className={className ?? "mx-auto aspect-square w-full max-w-[440px]"}
      >
        {mini ? "—" : "Waiting for the embedding projection…"}
      </FigurePlaceholder>
    )
  }

  // ── project the current orientation ──────────────────────────────────────────
  const S = 100
  const C = S / 2
  // Positions live in the unit ball (|v| ≤ 1), so the fit scale is a constant.
  const scale = (mini ? 44 : 46) / 1.02
  const rp = verts.map((p) => applyMat(orient, p)) // rotated 3-D (z = depth, larger = nearer)
  const sp = rp.map(([x, y]) => [C + x * scale, C - y * scale] as const)

  // Painter's order: far faces first. Facing (rotated normal · viewer) drives the
  // fill/stroke opacity so the lit front reads solid and the far side stays a ghost.
  const drawn = mesh.faces
    .map(([a, b, c]) => {
      const ux = rp[b][0] - rp[a][0]
      const uy = rp[b][1] - rp[a][1]
      const uz = rp[b][2] - rp[a][2]
      const vx = rp[c][0] - rp[a][0]
      const vy = rp[c][1] - rp[a][1]
      const vz = rp[c][2] - rp[a][2]
      const nx = uy * vz - uz * vy
      const ny = uz * vx - ux * vz
      const nz = ux * vy - uy * vx
      const nl = Math.hypot(nx, ny, nz) || 1
      const facing = Math.max(0, nz / nl)
      const depth = (rp[a][2] + rp[b][2] + rp[c][2]) / 3
      return { a, b, c, depth, facing }
    })
    .sort((p, q) => p.depth - q.depth)

  const svg = (
    <svg
      viewBox={`0 0 ${S} ${S}`}
      preserveAspectRatio="xMidYMid meet"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      className={
        mini
          ? "block size-full touch-none select-none"
          : "block aspect-square w-full touch-none select-none rounded-md border" +
            (dragging ? " cursor-grabbing" : " cursor-grab")
      }
    >
      {drawn.map(({ a, b, c, facing }, i) => {
        // Fill only. The stroke matches the fill so adjacent facets close their seams
        // without drawing a visible mesh — the smoothness comes from shading, not lines.
        const op = 0.05 + 0.2 * facing
        return (
          <path
            key={i}
            d={`M ${sp[a][0]} ${sp[a][1]} L ${sp[b][0]} ${sp[b][1]} L ${sp[c][0]} ${sp[c][1]} Z`}
            fill={EMB_COLOR}
            fillOpacity={op}
            stroke={EMB_COLOR}
            strokeOpacity={op}
            strokeWidth={0.3}
            strokeLinejoin="round"
          />
        )
      })}
      {/* The readable structure: only the token-to-token links, each faded by depth so
          the near side reads and the far side recedes. */}
      {mesh.tokenEdges.map(([a, b], i) => {
        const depth = (rp[a][2] + rp[b][2]) / 2
        const t = Math.max(0, Math.min(1, (depth + 1) / 2))
        return (
          <line
            key={i}
            x1={sp[a][0]}
            y1={sp[a][1]}
            x2={sp[b][0]}
            y2={sp[b][1]}
            stroke={EMB_COLOR}
            strokeOpacity={0.06 + 0.34 * t}
            strokeWidth={mini ? 0.3 : 0.35}
            strokeLinecap="round"
          />
        )
      })}
      {Array.from({ length: mesh.tokenCount }, (_, i) => {
        const [x, y] = sp[i]
        // Near vertices sit brighter; far ones sink back — a cheap depth cue.
        const t = (rp[i][2] + 1) / 2
        return (
          <circle
            key={i}
            cx={x}
            cy={y}
            r={mini ? 0.9 : 1.05}
            fill={embRankColor(i, n)}
            fillOpacity={0.45 + 0.55 * Math.max(0, Math.min(1, t))}
          />
        )
      })}
    </svg>
  )

  if (mini) {
    return <div className={className ?? "size-20"}>{svg}</div>
  }

  return (
    <div className={className ?? "mx-auto w-full max-w-[440px] space-y-3"}>
      {svg}
      <div className="flex items-center gap-3">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="size-7 shrink-0"
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? "Pause rotation" : "Play rotation"}
        >
          {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
        </Button>
        <span className="text-xs text-muted-foreground">
          Drag to orbit · {n} frequent tokens on the top-3 principal axes
        </span>
      </div>
    </div>
  )
}
