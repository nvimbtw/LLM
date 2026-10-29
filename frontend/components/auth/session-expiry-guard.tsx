"use client"

import * as React from "react"
import { usePathname, useRouter } from "next/navigation"

import { subscribeSessionExpiry } from "@/lib/api"

// Routes that stay readable after their stale token is discarded. Redirecting
// away from public documentation or an auth form would be a reload the reader
// did not ask for.
const PUBLIC_PATHS = ["/docs", "/login", "/signup"]

/** Reports whether a route remains usable without a session. */
function isPublicPath(pathname: string): boolean {
  return PUBLIC_PATHS.some(
    (path) => pathname === path || pathname.startsWith(`${path}/`)
  )
}

/** Answers a rejected session with a client navigation, so expiry never reloads the document. */
export function SessionExpiryGuard() {
  const router = useRouter()
  const pathname = usePathname()

  React.useEffect(
    () =>
      subscribeSessionExpiry(() => {
        if (!isPublicPath(pathname)) router.replace("/login")
      }),
    [pathname, router]
  )

  return null
}
