"use client"

import * as React from "react"

import {
  API_URL,
  AccessLevel,
  SESSION_CHANGED_EVENT,
  clearSession,
  getAccessLevel,
  getToken,
  setAccessLevel,
  setToken,
} from "@/lib/api"

export interface AuthUser {
  id: string
  name: string
  email: string
}

export interface LoginCredentials {
  username: string
  password: string
}

export interface AuthState {
  user: AuthUser | null
  isAuthenticated: boolean
  isAdmin: boolean
  accessLevel: AccessLevel | null
  isLoading: boolean
  login: (credentials: LoginCredentials) => Promise<void>
  logout: () => Promise<void>
}

// The session lives in localStorage, which is an external store — so it is read
// through `useSyncExternalStore` rather than copied into state on mount. That is
// not only what the lint rule wants: the old version snapshotted the token once
// per hook instance and never heard about a later change, so logging out left
// every *other* `useAuth()` consumer (the sidebar's admin nav, the jobs
// provider) believing the session was still live until something else
// re-rendered them. `SESSION_CHANGED_EVENT` is already broadcast by `setToken`,
// `setAccessLevel` and `clearSession`, so one subscription keeps them all honest.
function subscribe(onChange: () => void) {
  window.addEventListener(SESSION_CHANGED_EVENT, onChange)
  // Fired by the browser in *other* tabs. Logging out in one tab should not
  // leave a second tab holding a live-looking session.
  window.addEventListener("storage", onChange)
  return () => {
    window.removeEventListener(SESSION_CHANGED_EVENT, onChange)
    window.removeEventListener("storage", onChange)
  }
}

// Two subscriptions, not one returning `{ token, accessLevel }`: the snapshot
// is compared by identity, and a fresh object every call is an infinite loop.
const serverToken = () => null
const serverAccessLevel = () => null

// `isLoading` means "the server render and the hydration pass have not read
// localStorage yet", which is exactly the server/client snapshot split — so it
// needs no state of its own.
const clientHydrated = () => false
const serverHydrated = () => true

export function useAuth(): AuthState {
  const token = React.useSyncExternalStore(subscribe, getToken, serverToken)
  const accessLevel = React.useSyncExternalStore(
    subscribe,
    getAccessLevel,
    serverAccessLevel
  )
  const isLoading = React.useSyncExternalStore(
    subscribe,
    clientHydrated,
    serverHydrated
  )

  async function login(credentials: LoginCredentials) {
    const response = await fetch(`${API_URL}/auth`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(credentials),
    })

    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || "Unable to log in.")
    }

    const data = await response.json()
    const nextToken: string = data.token ?? data.access_token
    const nextAccessLevel = (
      data.access_level as string | undefined
    )?.toLowerCase() as AccessLevel | undefined

    // Access level first: both writes broadcast, and setting the token last
    // means the render that first sees `isAuthenticated` already sees the
    // level that goes with it.
    if (nextAccessLevel) setAccessLevel(nextAccessLevel)
    setToken(nextToken)
  }

  async function logout() {
    clearSession()
  }

  return {
    user: null,
    isAuthenticated: token !== null,
    isAdmin: accessLevel === "admin",
    accessLevel,
    isLoading,
    login,
    logout,
  }
}
