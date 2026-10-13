import { expression, HEADER_FLOATS, PARAMETER_FLOATS } from "./program"
import type { ArtStream } from "./stream"
import type { ThrobberMode } from "@/components/ui/trellis-throbber"
import type { Gpu, Renderer } from "./generated/renderer"
import { ArtSampler } from "./sampler"
import { ArtMorph } from "./morph"
import { motionPose, motionRate, POSE_FLOATS, POSE_SECONDS } from "./motion"
import { FrameClock, FRAME_MS, REDUCED_FRAME_MS } from "./frame-clock"

const MAX_SIDE = 192
const SETTLED = 0.001

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
  const morph = new ArtMorph(
    expression(stream.snapshot.seed, stream.snapshot.structure)
  )
  const sampler = new ArtSampler(stream.snapshot)
  const clock = new FrameClock(reduced ? REDUCED_FRAME_MS : FRAME_MS)
  const pose = new Float32Array(POSE_FLOATS)
  const desiredPose = new Float32Array(POSE_FLOATS)
  pose[2] = 1
  let seenMode = options.mode()
  let modeBegan = performance.now()
  let phase = 0
  let pace = 0
  let previous = 0
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
    if (!clock.take(now)) {
      if (reduced) {
        timer = setTimeout(() => {
          timer = undefined
          schedule()
        }, clock.remaining(now))
      } else {
        schedule()
      }
      return
    }
    const dt = Math.min((now - (previous || now)) / 1000, 0.064)
    previous = now
    const snapshot = stream.snapshot
    const reset =
      snapshot.structure !== sampler.snapshot.structure ||
      snapshot.count < sampler.snapshot.count
    if (sampler.sample(snapshot, now)) {
      const target = expression(snapshot.seed, snapshot.structure)
      if (reset || reduced) morph.reset(target)
      else morph.retarget(target, now)
    }
    const morphing = morph.advance(now)
    let distance = 0
    const mode = options.mode()
    if (mode !== seenMode) {
      seenMode = mode
      modeBegan = now
    }
    const rate = reduced ? 0 : motionRate(mode, (now - modeBegan) / 1000)
    pace += (rate - pace) * (1 - Math.exp(-dt / 0.35))
    phase += pace * dt
    motionPose(reduced ? "idle" : mode, phase, desiredPose)
    const poseBlend = reduced ? 1 : 1 - Math.exp(-dt / POSE_SECONDS)
    for (let i = 0; i < POSE_FLOATS; i++) {
      // Ripple phase wraps at 2π; interpolating it across the seam reverses time.
      if (i === 6) {
        pose[i] = desiredPose[i]
        continue
      }
      const delta = desiredPose[i] - pose[i]
      distance = Math.max(distance, Math.abs(delta))
      pose[i] += delta * poseBlend
    }
    parameters.set(pose)
    parameters.set(morph.values, HEADER_FLOATS)
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
      (!reduced &&
        (rate > 0 || pace > SETTLED || distance > SETTLED || morphing)) ||
      sampler.pending(snapshot) ||
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
    // Edge contact or a collapsed element can intersect without a visible pixel.
    visible =
      entry.isIntersecting &&
      entry.intersectionRect.width > 0 &&
      entry.intersectionRect.height > 0
    if (visible) {
      previous = 0
      clock.reset()
      schedule()
    } else {
      cancelAnimationFrame(frame)
      frame = 0
      clearTimeout(timer)
      timer = undefined
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
      clock.reset()
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
      POSE_FLOATS + index * 4
    )
  })
}
