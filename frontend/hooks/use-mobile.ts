import * as React from "react"

const MOBILE_BREAKPOINT = 768
const QUERY = `(max-width: ${MOBILE_BREAKPOINT - 1}px)`

// The viewport is an external store, so it is subscribed to rather than mirrored
// into state on mount — which also removes the `boolean | undefined` third state
// the mount-effect version needed to say "not measured yet".
function subscribe(onChange: () => void) {
  const mql = window.matchMedia(QUERY)
  mql.addEventListener("change", onChange)
  return () => mql.removeEventListener("change", onChange)
}

const getSnapshot = () => window.matchMedia(QUERY).matches

// Desktop on the server. The sidebar renders its desktop form first either way,
// and a mobile client corrects it on hydration.
const getServerSnapshot = () => false

export function useIsMobile() {
  return React.useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
}
