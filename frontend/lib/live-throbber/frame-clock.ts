export const FRAME_MS = 1000 / 60
export const REDUCED_FRAME_MS = 500
// rAF timestamps round and 59.94 Hz displays fall just short of an exact 60 Hz.
const FRAME_TOLERANCE_MS = 1

/** Keeps the target cadence across refresh rates, without catching up missed frames. */
export class FrameClock {
  private nextAt = 0

  constructor(private readonly interval: number) {}

  take(now: number): boolean {
    if (now + FRAME_TOLERANCE_MS < this.nextAt) return false
    const late = Math.max(0, now - this.nextAt + FRAME_TOLERANCE_MS)
    this.nextAt += (Math.floor(late / this.interval) + 1) * this.interval
    return true
  }

  remaining(now: number): number {
    return Math.max(0, this.nextAt - now)
  }

  reset(): void {
    this.nextAt = 0
  }
}
