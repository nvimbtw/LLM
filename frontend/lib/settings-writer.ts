import type { Settings } from "./settings"

export const SETTINGS_WRITE_DEBOUNCE_MS = 500

/**
 * Owns one settings write timer, coalescing slider/text changes into the last complete value.
 * Flush writes once immediately; discard cancels pending work when the session is gone.
 */
export function createSettingsWriter(
  write: (settings: Settings) => void,
  debounceMs = SETTINGS_WRITE_DEBOUNCE_MS
) {
  let timer: ReturnType<typeof setTimeout> | null = null
  let pending: Settings | null = null

  /** Cancels the owned timer without changing the pending value. */
  function clearTimer(): void {
    if (timer !== null) clearTimeout(timer)
    timer = null
  }

  /** Writes the last scheduled value once and clears both pending state and the timer. */
  function flush(): void {
    clearTimer()
    if (!pending) return
    const settings = pending
    pending = null
    write(settings)
  }

  /** Replaces pending preferences and restarts the debounce window from this change. */
  function schedule(settings: Settings): void {
    pending = settings
    clearTimer()
    timer = setTimeout(flush, debounceMs)
  }

  /** Drops work belonging to a session that can no longer authorize the write. */
  function discard(): void {
    clearTimer()
    pending = null
  }

  return { schedule, flush, discard }
}
