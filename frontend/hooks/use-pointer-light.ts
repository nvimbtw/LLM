import { useEffect, useRef, type PointerEvent } from "react"

import { usePrefersReducedMotion } from "./use-reduced-motion"

const TRAIL_LIFETIME_MS = 800
const TRAIL_POINTS = 8
const TRAIL_SPACING_PX = 24
const EMPTY_MASK = "linear-gradient(transparent, transparent)"
const IDLE_DELAY_MS = 240

interface TrailPoint {
  x: number
  y: number
  time: number
}

interface PointerLightOptions {
  trailLifetimeMs?: number
  trailRadiusRem?: number
}

export function usePointerLight({
  trailLifetimeMs = TRAIL_LIFETIME_MS,
  trailRadiusRem = 5,
}: PointerLightOptions = {}) {
  const scene = useRef<HTMLElement>(null)
  const trail = useRef<HTMLDivElement>(null)
  const points = useRef<TrailPoint[]>([])
  const frame = useRef<number | null>(null)
  const idle = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reducedMotion = usePrefersReducedMotion()

  useEffect(
    () => () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current)
      if (idle.current !== null) clearTimeout(idle.current)
    },
    []
  )

  function fadeTrail(now: number) {
    points.current = points.current.filter(
      ({ time }) => now - time < trailLifetimeMs
    )
    if (trail.current) {
      trail.current.style.maskImage =
        points.current
          .map(({ x, y, time }) => {
            const age = Math.max(0, now - time)
            const strength = (1 - age / trailLifetimeMs) ** 2
            return `radial-gradient(circle ${trailRadiusRem}rem at calc(${x}px - var(--pointer-trail-offset-x, 0px)) calc(${y}px - var(--pointer-trail-offset-y, 0px)), rgb(0 0 0 / ${strength}), transparent)`
          })
          .join(", ") || EMPTY_MASK
    }
    frame.current = points.current.length
      ? requestAnimationFrame(fadeTrail)
      : null
  }

  function illuminate(event: PointerEvent<HTMLElement>) {
    if (reducedMotion || event.pointerType !== "mouse" || !scene.current) return
    const bounds = event.currentTarget.getBoundingClientRect()
    const point = {
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
      time: performance.now(),
    }
    scene.current.style.setProperty("--pointer-light-x", `${point.x}px`)
    scene.current.style.setProperty("--pointer-light-y", `${point.y}px`)
    scene.current.dataset.illuminated = "true"
    if (idle.current !== null) clearTimeout(idle.current)
    idle.current = setTimeout(dim, IDLE_DELAY_MS)
    const previous = points.current.at(-1)
    if (
      !previous ||
      Math.hypot(point.x - previous.x, point.y - previous.y) >= TRAIL_SPACING_PX
    ) {
      points.current = [...points.current.slice(-(TRAIL_POINTS - 1)), point]
      if (frame.current === null)
        frame.current = requestAnimationFrame(fadeTrail)
    }
  }

  function dim() {
    if (idle.current !== null) clearTimeout(idle.current)
    idle.current = null
    if (scene.current) delete scene.current.dataset.illuminated
  }

  return { scene, trail, illuminate, dim }
}
