/**
 * Holds superseding values until the next animation frame.
 *
 * A streaming source can notify far more often than a screen can change — one
 * token per socket frame, sixty frames a second — and each value replaces the
 * last one rather than adding to it. Coalescing them here keeps the work
 * proportional to frames instead of to notifications, and leaves the source
 * free to be a plain synchronous callback.
 *
 * `flush` is for values that must not be overtaken: a final message, a history
 * read, a new turn. It applies anything pending immediately, in order, so a
 * notification that arrives after the last token is never shown before it.
 */
export class FrameCoalescer<T> {
  private pending: { value: T } | null = null
  private frame: number | null = null

  /**
   * Creates a coalescer that delivers accepted values through `apply`.
   * `apply` runs outside React's render, once per frame at most.
   */
  constructor(private readonly apply: (value: T) => void) {}

  /** Replaces any value not yet applied and schedules delivery on the next frame. */
  push(value: T): void {
    this.pending = { value }
    if (this.frame === null)
      this.frame = requestAnimationFrame(() => {
        this.frame = null
        this.drain()
      })
  }

  /** Delivers the pending value now, cancelling the scheduled frame; a no-op when nothing is pending. */
  flush(): void {
    if (this.frame !== null) {
      cancelAnimationFrame(this.frame)
      this.frame = null
    }
    this.drain()
  }

  /** Drops anything pending — for an unmount or a conversation being left — without applying it. */
  dispose(): void {
    if (this.frame !== null) {
      cancelAnimationFrame(this.frame)
      this.frame = null
    }
    this.pending = null
  }

  /** Applies the buffered value once, clearing it first so a re-entrant push is not lost. */
  private drain(): void {
    const next = this.pending
    this.pending = null
    if (next) this.apply(next.value)
  }
}
