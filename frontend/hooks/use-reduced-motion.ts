import * as React from "react"

const QUERY = "(prefers-reduced-motion: reduce)"

// The same shape as `use-mobile.ts`, and for the same reason: a media query is
// an external store, subscribed to rather than mirrored into state on mount.
function subscribe(onChange: () => void) {
  const mql = window.matchMedia(QUERY)
  mql.addEventListener("change", onChange)
  return () => mql.removeEventListener("change", onChange)
}

const getSnapshot = () => window.matchMedia(QUERY).matches

// Full motion on the server. Anything that animates on mount is corrected on
// hydration, before a frame of it has been drawn.
const getServerSnapshot = () => false

export function usePrefersReducedMotion() {
  return React.useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
}
