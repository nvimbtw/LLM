"use client"

import * as React from "react"

import { cn } from "@/lib/utils"

export type ThrobberMode = "thinking" | "loading" | "idle" | "training"

type Motif = "sun" | "wave" | "streak"
type Dir = "normal" | "reverse"

interface Scene {
  motif: Motif
  variant: number
  colorA: string
  colorB: string
  colorDuration: number // seconds, for the A<->B color swap
  direction: Dir
  duration: number // seconds, for the motif's motion
}

interface Preset {
  /** Motion duration range (seconds). */
  duration: [number, number]
  /** Time between scene switches (ms). */
  interval: [number, number]
  /** Color A<->B swap period range (seconds). */
  color: [number, number]
}

const PRESETS: Record<ThrobberMode, Preset> = {
  thinking: { duration: [0.6, 1.2], interval: [1400, 3200], color: [1, 1.8] },
  loading: { duration: [1.6, 2.8], interval: [2800, 5200], color: [1.6, 2.8] },
  idle: { duration: [6, 10], interval: [5500, 10000], color: [3, 5] },
  training: { duration: [3.5, 6], interval: [9000, 16000], color: [4, 7] },
}

const MOTIFS: Motif[] = ["sun", "wave", "streak"]
const VARIANTS = 3

// Theme color tokens (raw CSS vars so we can animate `color` between them).
const C = {
  yellow: "var(--chart-1)",
  green: "var(--chart-2)",
  blue: "var(--chart-3)",
  rose: "var(--chart-4)",
  purple: "var(--chart-5)",
  red: "var(--destructive)",
}

// 2 colors per scheme, 2–3 schemes per motif. A scene picks one scheme and
// swaps between its two colors periodically.
const MOTIF_SCHEMES: Record<Motif, [string, string][]> = {
  streak: [
    [C.blue, C.green],
    [C.purple, C.blue],
  ],
  wave: [
    [C.yellow, C.green],
    [C.green, C.blue],
  ],
  sun: [
    [C.yellow, C.red],
    [C.yellow, C.rose],
    [C.rose, C.red],
  ],
}

// Deterministic first scene so SSR and the first client render match; the real
// randomization only kicks in after mount (see effect below).
const INITIAL_SCENE: Scene = {
  motif: "wave",
  variant: 0,
  colorA: C.yellow,
  colorB: C.green,
  colorDuration: 2,
  direction: "normal",
  duration: 2,
}

function rand(min: number, max: number) {
  return min + Math.random() * (max - min)
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

function pickDifferent<T>(arr: T[], prev: T): T {
  if (arr.length < 2) return arr[0]
  let next = prev
  while (next === prev) next = pick(arr)
  return next
}

function rollScene(
  preset: Preset,
  colorsOverride: string[] | undefined,
  forceDirection: Dir | "alternate" | "random" | undefined,
  speed: number,
  prev: Scene
): Scene {
  const motif = pickDifferent(MOTIFS, prev.motif)

  // Pick a 2-color scheme, avoiding an exact repeat of the previous pair.
  const schemes = MOTIF_SCHEMES[motif]
  let scheme = pick(schemes)
  if (schemes.length > 1) {
    while (scheme[0] === prev.colorA && scheme[1] === prev.colorB) {
      scheme = pick(schemes)
    }
  }
  let [colorA, colorB] =
    colorsOverride && colorsOverride.length >= 2
      ? [colorsOverride[0], colorsOverride[1]]
      : scheme
  if (Math.random() < 0.5) [colorA, colorB] = [colorB, colorA]

  const direction: Dir =
    forceDirection === "normal" || forceDirection === "reverse"
      ? forceDirection
      : forceDirection === "alternate"
        ? prev.direction === "normal"
          ? "reverse"
          : "normal"
        : Math.random() < 0.5
          ? "normal"
          : "reverse"

  return {
    motif,
    variant: Math.floor(Math.random() * VARIANTS),
    colorA,
    colorB,
    colorDuration: rand(preset.color[0], preset.color[1]),
    direction,
    duration: rand(preset.duration[0], preset.duration[1]) / speed,
  }
}

export function FluxThrobber({
  mode = "idle",
  size,
  width,
  height,
  className,
  speed = 1,
  direction,
  colors,
}: {
  mode?: ThrobberMode
  size?: number
  width?: number
  height?: number
  className?: string
  speed?: number
  direction?: "forward" | "reverse" | "alternate" | "random"
  colors?: string[]
}) {
  const [scene, setScene] = React.useState<Scene>(INITIAL_SCENE)
  const [sceneId, setSceneId] = React.useState(0)

  const sceneRef = React.useRef(scene)
  sceneRef.current = scene

  const forceDir =
    direction === "forward"
      ? "normal"
      : direction === "reverse"
        ? "reverse"
        : direction

  React.useEffect(() => {
    const preset = PRESETS[mode]

    const next = rollScene(preset, colors, forceDir, speed, sceneRef.current)
    setScene(next)
    setSceneId((id) => id + 1)

    // Idle intentionally holds a single motif + scheme (no scene switching).
    // NOTE: we deliberately do NOT honor prefers-reduced-motion here. On some
    // environments (notably WSLg, where GNOME's "enable animations" setting is
    // unset) Chromium/Brave falsely report reduced-motion, which previously
    // froze the entire throbber. The throbber is decorative, so we always run.
    if (mode === "idle") return

    let timer: ReturnType<typeof setTimeout>
    const tick = () => {
      const delay = rand(preset.interval[0], preset.interval[1])
      timer = setTimeout(() => {
        setScene((prev) => rollScene(preset, colors, forceDir, speed, prev))
        setSceneId((id) => id + 1)
        tick()
      }, delay)
    }
    tick()

    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, speed, forceDir])

  return (
    <span
      role="status"
      aria-label="Loading"
      className={cn(
        "relative inline-block overflow-hidden rounded-md bg-muted/40",
        className
      )}
      style={{ width: width ?? size, height: height ?? size }}
    >
      <style>{KEYFRAMES}</style>
      <svg
        data-flux
        viewBox="0 0 400 140"
        preserveAspectRatio="xMidYMid slice"
        className="h-full w-full"
        fill="none"
        aria-hidden
        style={
          {
            "--flux-a": scene.colorA,
            "--flux-b": scene.colorB,
            color: "var(--flux-a)",
            animation: `flux-hue ${scene.colorDuration}s ease-in-out infinite`,
          } as React.CSSProperties
        }
      >
        <g key={sceneId} style={{ animation: "flux-enter 0.5s ease-out both" }}>
          {scene.motif === "sun" && <Sun scene={scene} />}
          {scene.motif === "wave" && <Wave scene={scene} />}
          {scene.motif === "streak" && <Streak scene={scene} />}
        </g>
      </svg>
    </span>
  )
}

function anim(name: string, scene: Scene): React.CSSProperties {
  return {
    animation: `${name} ${scene.duration}s linear infinite`,
    animationDirection: scene.direction,
  }
}

// Rising sun: a fixed dot at the bottom centre; concentric rings radiate
// outward (or inward, when reversed) at constant speed — evenly spaced, fading
// in and out at the edges. The radius itself animates (not a scale), so the
// spacing between rings stays uniform as they travel.
function Sun({ scene }: { scene: Scene }) {
  const rings = 5 + (scene.variant % 3)

  return (
    <g stroke="currentColor" strokeWidth="2" fill="none">
      {Array.from({ length: rings }).map((_, i) => (
        <circle
          key={i}
          cx="200"
          cy="140"
          r={6}
          // Animate via transform: scale() rather than the `r` CSS property.
          // Animating SVG geometry props (r/cx/cy) in CSS is Blink-only and
          // unreliable across GPU/render paths (breaks on WSL/Brave, Safari,
          // Firefox). scale() is universally supported and GPU-accelerated;
          // fill-box origin keeps it centred on the circle and
          // non-scaling-stroke keeps the stroke a constant 2px.
          vectorEffect="non-scaling-stroke"
          style={{
            ...anim("flux-radiate", scene),
            transformBox: "fill-box",
            transformOrigin: "center",
            animationDelay: `${(i / rings) * scene.duration}s`,
          }}
        />
      ))}
      {/* The sun itself — anchored at the bottom, does not move. */}
      <circle cx="200" cy="140" r="12" fill="currentColor" stroke="none" />
    </g>
  )
}

// Build a smooth squiggle spanning [x0, x1] around baseline `y`; one period =
// `w`, so a translate of `w` loops seamlessly. The vertical baseline is baked
// into the geometry so the only animated transform is horizontal.
function wavePath(amp: number, w: number, x0: number, x1: number, y: number) {
  let d = `M ${x0} ${y}`
  let up = -1
  for (let x = x0; x < x1; x += w / 2) {
    d += ` Q ${x + w / 4} ${y + up * amp} ${x + w / 2} ${y}`
    up *= -1
  }
  return d
}

// Stacked squiggles scrolling horizontally (left or right by direction). Each
// line gets its own speed, phase, and amplitude and is spread across the height,
// so they drift against each other instead of clumping into one band.
const WAVE_SPEEDS = [1, 1.55, 0.78, 1.28, 0.95]

function Wave({ scene }: { scene: Scene }) {
  const baseAmp = [16, 22, 12][scene.variant % 3]
  const lines = 4 + (scene.variant % 2) // 4 or 5

  return (
    <g stroke="currentColor" strokeWidth="2" fill="none">
      {Array.from({ length: lines }).map((_, i) => {
        const amp = baseAmp * (0.75 + (i % 3) * 0.2)
        const y = ((i + 1) / (lines + 1)) * 140
        return (
          <path
            key={i}
            d={wavePath(amp, 160, -320, 720, y)}
            strokeOpacity={0.4 + (i % 2) * 0.18}
            style={{
              animation: `flux-scroll-x ${scene.duration * WAVE_SPEEDS[i % WAVE_SPEEDS.length]}s linear infinite`,
              animationDirection: scene.direction,
              animationDelay: `${-i * 0.7}s`,
            }}
          />
        )
      })}
    </g>
  )
}

// Diagonal beams sliding horizontally (left or right by direction).
function Streak({ scene }: { scene: Scene }) {
  const shallow = scene.variant === 1
  const count = 12 + (scene.variant % 3) * 2
  const gap = 56
  const dx = shallow ? 120 : 70
  const weight = scene.variant === 2 ? 3 : 2

  return (
    <g
      stroke="currentColor"
      strokeWidth={weight}
      strokeLinecap="round"
      style={anim("flux-march", scene)}
    >
      {Array.from({ length: count }).map((_, i) => (
        <line
          key={i}
          x1={i * gap - 160}
          y1="160"
          x2={i * gap - 160 + dx}
          y2="-20"
          strokeOpacity={i % 2 === 0 ? 0.45 : 0.24}
        />
      ))}
    </g>
  )
}

const KEYFRAMES = `
@keyframes flux-enter {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes flux-hue {
  0%, 40% { color: var(--flux-a); }
  50%, 90% { color: var(--flux-b); }
  100% { color: var(--flux-a); }
}
@keyframes flux-radiate {
  /* base r=6 scaled to 28.667 == r 172, kept linear so ring spacing is uniform */
  0% { transform: scale(1); opacity: 0; }
  18% { opacity: 0.5; }
  82% { opacity: 0.5; }
  100% { transform: scale(28.667); opacity: 0; }
}
@keyframes flux-scroll-x {
  to { transform: translateX(-160px); }
}
@keyframes flux-march {
  to { transform: translateX(56px); }
}
`
