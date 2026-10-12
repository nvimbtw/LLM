import type { ArtSnapshot } from "./stream"

export const VISUAL_TARGETS_PER_SECOND = 1.25
export const TARGET_INTERVAL_MS = 1000 / VISUAL_TARGETS_PER_SECOND
export const MORPH_SECONDS = 0.9

/** Samples the newest accumulated input on a visual clock, with no catch-up queue. */
export class ArtSampler {
  snapshot: ArtSnapshot
  private nextAt = 0

  constructor(initial: ArtSnapshot) {
    this.snapshot = initial
  }

  pending(latest: ArtSnapshot): boolean {
    return (
      latest.seed !== this.snapshot.seed ||
      latest.structure !== this.snapshot.structure
    )
  }

  sample(latest: ArtSnapshot, now: number): ArtSnapshot | undefined {
    const reset =
      latest.structure !== this.snapshot.structure ||
      latest.count < this.snapshot.count
    if (!this.pending(latest) || (!reset && now < this.nextAt)) return
    this.snapshot = latest
    this.nextAt = now + TARGET_INTERVAL_MS
    return latest
  }
}
