import { expression, HEADER_FLOATS, PARAMETER_FLOATS } from "./program"
import type { ArtStream } from "./stream"
import type { ThrobberMode } from "@/components/ui/trellis-throbber"
import type { Gpu, Renderer } from "./generated/renderer"

const FRAME_MS = 1000 / 24
const MAX_SIDE = 192
const TARGET_MS = 100
const MORPH_SECONDS = 0.42
const SETTLED = 0.001
const RATES: Record<ThrobberMode, number> = {
  idle: 0,
  loading: 0.28,
  thinking: 0.35,
  generating: 0.6,
  tool: 0.2,
  training: 0.5,
  benchmarking: 0.4,
  tokenizing: 0.45,
}

let gpu: Promise<Gpu> | undefined

async function device(): Promise<Gpu> {
  if (!window.isSecureContext || !("gpu" in navigator))
    throw new Error("WebGPU unavailable")
  gpu ??= import("./generated/renderer").then(
    async ({ default: init, Gpu }) => {
      await init({ module_or_path: "/live-throbber/renderer_bg.wasm" })
      return Gpu.create()
    }
  )
  return gpu
}

export interface CanvasOptions {
  canvas: HTMLCanvasElement
  stream: ArtStream
  mode: () => ThrobberMode
  reduced: boolean
  signal: AbortSignal
  ready: () => void
  failed: () => void
}

/** Owns GPU resources and all observers; its returned cleanup is safe after async cancellation. */
export async function animateArt(options: CanvasOptions): Promise<() => void> {
  const engine = await device()
  if (options.signal.aborted) return () => {}
  const renderer = await engine.renderer(options.canvas)
  if (options.signal.aborted) {
    renderer.free()
    return () => {}
  }
  try {
    return run(options, renderer)
  } catch (error) {
    renderer.free()
    throw error
  }
}

function run(options: CanvasOptions, renderer: Renderer): () => void {
  const { canvas, stream, reduced } = options
  const parameters = new Float32Array(PARAMETER_FLOATS)
  let current = expression(stream.snapshot.seed, stream.snapshot.structure)
  let target = current.slice()
  let seenSeed = stream.snapshot.seed
  let seenStructure = stream.snapshot.structure
  let phase = 0
  let pace = 0
  let previous = 0
  let lastTarget = 0
  let frame = 0
  let timer: ReturnType<typeof setTimeout> | undefined
  let visible = true
  let disposed = false
  let ready = false

  const schedule = () => {
    if (disposed || frame || timer || !visible || document.hidden) return
    frame = requestAnimationFrame(tick)
  }
  const fail = () => {
    cleanup()
    options.failed()
  }
  const tick = (now: number) => {
    frame = 0
    if (disposed || !visible || document.hidden) return
    const interval = reduced ? 500 : FRAME_MS
    if (previous && now - previous < interval) {
      timer = setTimeout(
        () => {
          timer = undefined
          schedule()
        },
        interval - (now - previous)
      )
      return
    }
    const dt = Math.min((now - (previous || now)) / 1000, 0.064)
    previous = now
    const snapshot = stream.snapshot
    if (snapshot.structure !== seenStructure) {
      seenStructure = snapshot.structure
      current = expression(snapshot.seed, seenStructure)
      target = current.slice()
      seenSeed = snapshot.seed
    }
    if (snapshot.seed !== seenSeed && now - lastTarget >= TARGET_MS) {
      target = expression(snapshot.seed, seenStructure)
      seenSeed = snapshot.seed
      lastTarget = now
    }
    const blend = reduced ? 1 : 1 - Math.exp(-dt / MORPH_SECONDS)
    let distance = 0
    for (let i = 0; i < current.length; i++) {
      const delta = target[i] - current[i]
      distance = Math.max(distance, Math.abs(delta))
      current[i] += delta * blend
    }
    const rate = reduced ? 0 : RATES[options.mode()]
    pace += (rate - pace) * (1 - Math.exp(-dt / 0.35))
    phase += pace * dt
    parameters[0] = phase
    parameters.set(current, HEADER_FLOATS)
    let submitted: boolean
    try {
      submitted = renderer.render(parameters)
      if (submitted && !ready) {
        ready = true
        options.ready()
      }
    } catch {
      fail()
      return
    }
    if (
      (!reduced && (rate > 0 || pace > SETTLED || distance > SETTLED)) ||
      snapshot.seed !== seenSeed ||
      !submitted
    ) {
      schedule()
    }
  }

  const resize = new ResizeObserver(() => {
    const side = Math.max(
      1,
      Math.min(
        MAX_SIDE,
        Math.round(canvas.clientWidth * Math.min(devicePixelRatio, 2))
      )
    )
    canvas.width = canvas.height = side
    try {
      renderer.resize(side)
      schedule()
    } catch {
      fail()
    }
  })
  const intersection = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting
    if (visible) {
      previous = 0
      schedule()
    } else if (frame) {
      cancelAnimationFrame(frame)
      frame = 0
    }
  })
  const theme = new MutationObserver(() => {
    readPalette(canvas, parameters)
    schedule()
  })
  const unsubscribe = stream.subscribe(schedule)
  const visibility = () => {
    if (document.hidden) {
      cancelAnimationFrame(frame)
      frame = 0
      clearTimeout(timer)
      timer = undefined
    } else {
      previous = 0
      schedule()
    }
  }
  function cleanup() {
    if (disposed) return
    disposed = true
    cancelAnimationFrame(frame)
    clearTimeout(timer)
    resize.disconnect()
    intersection.disconnect()
    theme.disconnect()
    document.removeEventListener("visibilitychange", visibility)
    unsubscribe()
    renderer.free()
  }
  readPalette(canvas, parameters)
  resize.observe(canvas)
  intersection.observe(canvas)
  theme.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class", "style"],
  })
  document.addEventListener("visibilitychange", visibility)
  schedule()
  return cleanup
}

function readPalette(canvas: HTMLCanvasElement, parameters: Float32Array) {
  const style = getComputedStyle(canvas)
  const colours = [
    "--throbber-base",
    "--throbber-spark",
    "--primary",
    "--chart-1",
  ]
  colours.forEach((token, index) => {
    const value = style.getPropertyValue(token).trim()
    if (!/^#[\da-f]{6}$/i.test(value))
      throw new Error(`Unsupported colour: ${token}`)
    const rgb = Number.parseInt(value.slice(1), 16)
    parameters.set(
      [(rgb >>> 16) / 255, ((rgb >>> 8) & 255) / 255, (rgb & 255) / 255, 1],
      4 + index * 4
    )
  })
}
