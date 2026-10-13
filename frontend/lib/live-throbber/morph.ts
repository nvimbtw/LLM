import { TARGET_INTERVAL_MS } from "./sampler"

/** Linear segments share the target cadence; retargeting keeps the current image. */
export class ArtMorph {
  readonly values: Float32Array
  private readonly from: Float32Array
  private target: Float32Array
  private beganAt = 0
  private active = false

  constructor(initial: Float32Array) {
    this.values = initial.slice()
    this.from = initial.slice()
    this.target = initial
  }

  reset(target: Float32Array): void {
    this.values.set(target)
    this.from.set(target)
    this.target = target
    this.active = false
  }

  retarget(target: Float32Array, now: number): void {
    this.advance(now)
    this.from.set(this.values)
    this.target = target
    this.beganAt = now
    this.active = true
  }

  advance(now: number): boolean {
    if (!this.active) return false
    const progress = Math.min(
      1,
      Math.max(0, (now - this.beganAt) / TARGET_INTERVAL_MS)
    )
    for (let i = 0; i < this.values.length; i++) {
      this.values[i] = this.from[i] + (this.target[i] - this.from[i]) * progress
    }
    this.active = progress < 1
    return this.active
  }
}
