"use client"

import * as React from "react"

/**
 * Follows a rapidly changing number at a fixed maximum rate.
 *
 * A turn's token count changes on every streamed token, but a figure a reader
 * is watching does not need to change that often — and everything downstream of
 * it (a footer, a context meter, a whole composer) would otherwise redraw at
 * token rate. The latest value is held in a ref and published on a trailing
 * timer, so a burst costs one timer per interval rather than one per change and
 * the value that ends the burst is always the one that lands.
 *
 * Returns the last published value; the first interval after a change is the
 * only lag, and it is bounded by `intervalMs`.
 */
export function useThrottledNumber(value: number, intervalMs: number): number {
  const [published, setPublished] = React.useState(value)
  const latest = React.useRef(value)
  const timer = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  React.useEffect(() => {
    latest.current = value
    if (timer.current !== null) return
    timer.current = setTimeout(() => {
      timer.current = null
      setPublished(latest.current)
    }, intervalMs)
  }, [value, intervalMs])

  React.useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current)
    },
    []
  )

  return published
}
