import type { ThrobberMode } from "@/components/ui/trellis-throbber"

export const POSE_FLOATS = 8
export const POSE_SECONDS = 0.6
const TAU = Math.PI * 2
const RATES: Record<ThrobberMode, number> = {
  idle: 0,
  loading: 0.9,
  thinking: 0.6,
  generating: 0.25,
  tool: 0.7,
  training: 0.55,
  benchmarking: 0.65,
  tokenizing: 1.1,
}

export function motionRate(mode: ThrobberMode, ageSeconds: number): number {
  return mode === "loading"
    ? Math.max(
        0.25,
        RATES.loading / (1 + Math.log1p(Math.max(0, ageSeconds)) / 4)
      )
    : RATES[mode]
}

/** Offsets, scale, rotation, then scan position/strength and ripple phase/strength. */
export function motionPose(
  mode: ThrobberMode,
  phase: number,
  out: Float32Array
): void {
  out.fill(0)
  out[2] = 1
  // Keep phase continuous while a training ripple fades into another state.
  out[6] = phase % TAU
  switch (mode) {
    case "loading":
      out[2] += Math.sin(phase) * 0.07
      break
    case "thinking":
      out[0] = Math.sin(phase * 0.7) * 0.12
      out[1] = Math.cos(phase * 0.47) * 0.1
      out[3] = Math.sin(phase * 0.3) * 0.07
      break
    case "generating":
      out[0] = Math.sin(phase * 0.47) * 0.025
      out[1] = Math.cos(phase * 0.31) * 0.02
      break
    case "tool":
      out[2] += Math.sin(phase * 0.7) * 0.025
      out[3] = Math.sin(phase * 0.35) * 0.035
      break
    case "training":
      out[0] = Math.sin(phase * 0.4) * 0.06
      out[3] = Math.sin(phase * 0.2) * 0.08
      out[7] = 0.09
      break
    case "benchmarking":
      out[4] = Math.sin(phase * 1.4) * 2.4
      out[5] = -0.12
      break
    case "tokenizing":
      out[4] = Math.sin(phase * 1.4) * 2.4
      out[5] = 0.16
      break
  }
}
