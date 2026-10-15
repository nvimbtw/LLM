"use client"

import { useEffect, useRef, useState } from "react"

/** Owns a diagram's active elapsed time at at most 30 updates/second. Pausing, offscreen layout and hidden tabs retain the exact frame; all observers and frames are released on unmount. */
export function useArchitecturePlayback(running: boolean) {
  const ref = useRef<HTMLElement>(null)
  const elapsed = useRef(0)
  const [time, setTime] = useState(0)

  useEffect(() => {
    if (!running || !ref.current) return
    let visible = false
    let request = 0
    let previous = 0
    let published = 0

    /** Advances only active time; resumed tabs and offscreen diagrams start from their retained frame. */
    function tick(now: number) {
      if (previous) elapsed.current += now - previous
      previous = now
      if (now - published >= 1000 / 30) {
        setTime(elapsed.current)
        published = now
      }
      request = requestAnimationFrame(tick)
    }

    /** Reconciles the sole frame owner with viewport and document visibility. */
    function reconcile() {
      cancelAnimationFrame(request)
      previous = 0
      if (visible && !document.hidden) request = requestAnimationFrame(tick)
    }

    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting
      reconcile()
    })
    observer.observe(ref.current)
    document.addEventListener("visibilitychange", reconcile)
    return () => {
      cancelAnimationFrame(request)
      observer.disconnect()
      document.removeEventListener("visibilitychange", reconcile)
    }
  }, [running])

  return { ref, time }
}
