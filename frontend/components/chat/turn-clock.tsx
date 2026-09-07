"use client"

import * as React from "react"

/**
 * How long the turn in progress has been running, at the head of the thread's
 * status row.
 *
 * Mounted only while a turn is in flight, and it starts its clock from its own
 * mount rather than being handed a start time. That is not a shortcut — it is
 * what keeps the second hand out of `ChatThread`: a clock owned up there would
 * re-render every message in the conversation once a second for the length of
 * the reply, to move two characters.
 *
 * It measures the wait, not the model: a turn queued behind three jobs on the
 * GPU is counted from the moment the question was asked, because that is the
 * number the person watching it is actually keeping.
 */
export function TurnClock() {
  const [secs, setSecs] = React.useState(0)

  React.useEffect(() => {
    const started = Date.now()
    // Faster than the second it displays, so the number turns over within
    // ~200ms of the real boundary rather than drifting a whole second off it
    // over a long reply. React bails out of the re-render when the second has
    // not changed, so the extra ticks cost a comparison.
    const id = setInterval(
      () => setSecs(Math.floor((Date.now() - started) / 1000)),
      TICK_MS
    )
    return () => clearInterval(id)
  }, [])

  return (
    <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
      {clock(secs)}
    </span>
  )
}

const TICK_MS = 200

const MINUTE = 60
const HOUR = 60 * MINUTE

/** The running form — short, and the same width for whole stretches of it, so a
 * clock ticking beside the count does not shove it about. */
function clock(secs: number): string {
  if (secs < MINUTE) return `${secs}s`
  if (secs < HOUR) {
    return `${Math.floor(secs / MINUTE)}m ${pad(secs % MINUTE)}s`
  }
  return `${Math.floor(secs / HOUR)}h ${pad(Math.floor((secs % HOUR) / MINUTE))}m`
}

const pad = (n: number) => String(n).padStart(2, "0")

/**
 * The settled form, for the row once the reply is in: `Worked for 42 seconds`.
 *
 * Spelled out rather than `42s`, because it is now a sentence about something
 * finished rather than a readout of something happening — and the two states
 * sit in the same row a second apart, so they had better not look alike.
 * Coarsened as it grows: the seconds inside an eleven minute turn are noise,
 * and nobody reads a number they cannot use.
 */
export function workedFor(secs: number): string {
  const whole = Math.max(0, Math.round(secs))
  if (whole < MINUTE) return `Worked for ${plural(whole, "second")}`

  if (whole < HOUR) {
    const rest = whole % MINUTE
    const mins = plural(Math.floor(whole / MINUTE), "minute")
    return rest === 0
      ? `Worked for ${mins}`
      : `Worked for ${mins} ${plural(rest, "second")}`
  }

  const rest = Math.floor((whole % HOUR) / MINUTE)
  const hours = plural(Math.floor(whole / HOUR), "hour")
  return rest === 0
    ? `Worked for ${hours}`
    : `Worked for ${hours} ${plural(rest, "minute")}`
}

const plural = (n: number, unit: string) => `${n} ${unit}${n === 1 ? "" : "s"}`
