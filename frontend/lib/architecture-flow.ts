export type ArchitectureView = "training" | "generation"
export type ArchitectureStep =
  | "input"
  | "embedding"
  | "attention"
  | "feedforward"
  | "norm"
  | "output"
  | "sample"
  | "append"
  | "loss"
  | "backward"
  | "optimizer"
  | "update"
  | "batch"

export interface ArchitectureFrame {
  step: ArchitectureStep
  progress: number
  cycle: number
}

const FORWARD_STEPS: ArchitectureStep[] = [
  "input",
  "embedding",
  "attention",
  "feedforward",
  "norm",
  "output",
]
const GENERATION_STEPS: ArchitectureStep[] = [
  ...FORWARD_STEPS,
  "sample",
  "append",
]
const TRAINING_STEPS: ArchitectureStep[] = [
  ...FORWARD_STEPS,
  "loss",
  "backward",
  "optimizer",
  "update",
  "batch",
]
export const ARCHITECTURE_STEP_MS = 1800
export const ARCHITECTURE_RETURN_MS = 3000
const TOKEN_WIDTHS = [24, 36, 20, 30, 42, 26, 34]

export interface DiagramPoint {
  x: number
  y: number
}
export interface DiagramBox extends DiagramPoint {
  width: number
  height: number
}
export type Operation = Exclude<
  ArchitectureStep,
  "input" | "append" | "update" | "batch"
>

/** Derives the explanatory phase from active elapsed time. Each generation cycle appends one token; training advances batches only after weight updates. */
export function architectureFrame(
  view: ArchitectureView,
  elapsedMs: number
): ArchitectureFrame {
  const steps = view === "generation" ? GENERATION_STEPS : TRAINING_STEPS
  const cycleMs =
    (steps.length - 1) * ARCHITECTURE_STEP_MS + ARCHITECTURE_RETURN_MS
  const elapsed = Math.max(0, Number.isFinite(elapsedMs) ? elapsedMs : 0)
  const cycle = Math.floor(elapsed / cycleMs)
  let remaining = elapsed % cycleMs
  for (const step of steps) {
    const duration =
      step === "append" || step === "update"
        ? ARCHITECTURE_RETURN_MS
        : ARCHITECTURE_STEP_MS
    if (remaining < duration)
      return { step, progress: remaining / duration, cycle }
    remaining -= duration
  }
  return { step: "input", progress: 0, cycle: cycle + 1 }
}

/** Clamps an animation fraction to its closed unit interval. */
export function unit(value: number): number {
  return Math.max(0, Math.min(1, value))
}

/** Eases a bounded fraction with zero endpoint velocity, so completed motion settles instead of cutting off. */
export function smooth(value: number): number {
  const t = unit(value)
  return t * t * (3 - 2 * t)
}

/** Maps a named stage interval to a smooth, clamped transition. */
export function phaseProgress(
  progress: number,
  start: number,
  end: number
): number {
  return smooth((progress - start) / (end - start))
}

/** Fades stage emphasis in and out independently of geometry; pausing freezes both on the same clock. */
export function stageEmphasis(
  frame: ArchitectureFrame | null,
  step: ArchitectureStep
): number {
  if (!frame || frame.step !== step) return 0
  return (
    phaseProgress(frame.progress, 0.12, 0.3) *
    (1 - phaseProgress(frame.progress, 0.82, 1))
  )
}

/** Completes processing before the stage ends and retains its result until the next input. The next cycle gently resets the miniature instead of snapping back to a halfway illustration. */
export function operationProgress(
  frame: ArchitectureFrame | null,
  operation: Operation,
  view: ArchitectureView
): number {
  if (!frame) return 1
  if (frame.step === "input")
    return frame.cycle > 0 ? 1 - phaseProgress(frame.progress, 0, 0.7) : 0
  const steps = view === "generation" ? GENERATION_STEPS : TRAINING_STEPS
  const current = steps.indexOf(frame.step)
  const target = steps.indexOf(operation)
  if (current < target) return 0
  if (current > target) return 1
  return phaseProgress(frame.progress, 0.3, 0.82)
}

/** Gives illustrative tokens stable varied widths without allocating by model context or generated sequence length. */
export function tokenWidth(index: number): number {
  return TOKEN_WIDTHS[index % TOKEN_WIDTHS.length]
}

/** Returns a bounded window of token identities. Overflow scrolls during append, preserving the same token positions across the cycle boundary. */
export function promptTokens(frame: ArchitectureFrame, view: ArchitectureView) {
  const completed = view === "generation" ? frame.cycle : 0
  const total = 4 + completed
  const first =
    view === "generation"
      ? Math.max(0, total - 6)
      : 4 *
        (frame.cycle +
          (frame.step === "batch" && frame.progress >= 0.5 ? 1 : 0))
  const scroll =
    view === "generation" && frame.step === "append" && total >= 6
      ? smooth(frame.progress / 0.3) * (tokenWidth(first) + 6)
      : 0
  let x = 0
  const tokens = Array.from({ length: Math.min(total, 6) }, (_, offset) => {
    const index = first + offset
    const token = { index, x: x - scroll, width: tokenWidth(index) }
    x += token.width + 6
    return token
  })
  return {
    tokens,
    next: { index: total, x: x - scroll, width: tokenWidth(total) },
    clipped: view === "generation" && (first > 0 || scroll > 0),
  }
}

/** Converts a shared orthogonal route into SVG path data, used for both the line and its moving mark. */
export function diagramPath(points: DiagramPoint[]): string {
  return roundedRoute(points)
    .map((point, index) => `${index ? "L" : "M"}${point.x} ${point.y}`)
    .join(" ")
}

/** Locates a mark by distance along the exact connection polyline, including bends and zero-length segments. */
export function pointOnRoute(
  points: DiagramPoint[],
  progress: number
): DiagramPoint {
  points = roundedRoute(points)
  const lengths = points
    .slice(1)
    .map((point, index) =>
      Math.hypot(point.x - points[index].x, point.y - points[index].y)
    )
  let remaining =
    lengths.reduce((sum, length) => sum + length, 0) * unit(progress)
  for (let index = 0; index < lengths.length; index++) {
    const length = lengths[index]
    if (remaining <= length && length > 0) {
      const fraction = remaining / length
      return {
        x: points[index].x + (points[index + 1].x - points[index].x) * fraction,
        y: points[index].y + (points[index + 1].y - points[index].y) * fraction,
      }
    }
    remaining -= length
  }
  return points.at(-1) ?? { x: 0, y: 0 }
}

/** Softens route corners with a small sampled quadratic. Drawing and travel share these exact segments, including compact branches; endpoints remain fixed. */
function roundedRoute(points: DiagramPoint[]): DiagramPoint[] {
  if (points.length < 3) return points
  const route = [points[0]]
  for (let index = 1; index < points.length - 1; index++) {
    const before = points[index - 1]
    const corner = points[index]
    const after = points[index + 1]
    const incoming = Math.hypot(corner.x - before.x, corner.y - before.y)
    const outgoing = Math.hypot(after.x - corner.x, after.y - corner.y)
    const radius = Math.min(8, incoming / 2, outgoing / 2)
    if (!radius) {
      route.push(corner)
      continue
    }
    const entry = {
      x: corner.x + ((before.x - corner.x) * radius) / incoming,
      y: corner.y + ((before.y - corner.y) * radius) / incoming,
    }
    const exit = {
      x: corner.x + ((after.x - corner.x) * radius) / outgoing,
      y: corner.y + ((after.y - corner.y) * radius) / outgoing,
    }
    route.push(entry)
    for (let segment = 1; segment <= 8; segment++) {
      const t = segment / 8
      route.push({
        x:
          (1 - t) ** 2 * entry.x + 2 * (1 - t) * t * corner.x + t ** 2 * exit.x,
        y:
          (1 - t) ** 2 * entry.y + 2 * (1 - t) * t * corner.y + t ** 2 * exit.y,
      })
    }
  }
  route.push(points[points.length - 1])
  return route
}

/** Provides fixed, bounded operation geometry for a horizontal or narrow-column diagram. */
export function architectureLayout(vertical: boolean) {
  const positions: Record<Operation, DiagramPoint> = vertical
    ? {
        embedding: { x: 132, y: 112 },
        attention: { x: 132, y: 282 },
        feedforward: { x: 132, y: 422 },
        norm: { x: 132, y: 572 },
        output: { x: 132, y: 702 },
        sample: { x: 132, y: 842 },
        loss: { x: 132, y: 842 },
        backward: { x: 132, y: 982 },
        optimizer: { x: 132, y: 1122 },
      }
    : {
        embedding: { x: 24, y: 150 },
        attention: { x: 206, y: 150 },
        feedforward: { x: 398, y: 150 },
        norm: { x: 590, y: 150 },
        output: { x: 756, y: 150 },
        sample: { x: 756, y: 328 },
        loss: { x: 756, y: 340 },
        backward: { x: 398, y: 340 },
        optimizer: { x: 206, y: 340 },
      }
  const nodes = Object.fromEntries(
    Object.entries(positions).map(([key, position]) => [
      key,
      {
        ...position,
        width: vertical ? 152 : 112,
        height: 88,
      },
    ])
  ) as Record<Operation, DiagramBox>
  return {
    nodes,
    prompt: { x: vertical ? 36 : 24, y: 40 },
    stack: vertical
      ? { x: 92, y: 228, width: 244, height: 328 }
      : { x: 176, y: 82, width: 394, height: 184 },
    width: vertical ? 388 : 916,
  }
}

/** Returns a named port on an operation so all arrows meet the same exact boundary. */
export function port(
  box: DiagramBox,
  side: "top" | "bottom" | "left" | "right"
): DiagramPoint {
  return {
    x:
      side === "left"
        ? box.x
        : side === "right"
          ? box.x + box.width
          : box.x + box.width / 2,
    y:
      side === "top"
        ? box.y
        : side === "bottom"
          ? box.y + box.height
          : box.y + box.height / 2,
  }
}
