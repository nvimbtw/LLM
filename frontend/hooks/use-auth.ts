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

/** Subscribes to local and cross-tab session changes, returning cleanup for both listeners. */
function subscribe(onChange: () => void) {
  window.addEventListener(SESSION_CHANGED_EVENT, onChange)
  window.addEventListener("storage", onChange)
  return () => {
    window.removeEventListener(SESSION_CHANGED_EVENT, onChange)
    window.removeEventListener("storage", onChange)
  }
}

/** Returns the logged-out server snapshot before browser storage is available. */
const serverToken = () => null
/** Returns the unavailable server access-level snapshot. */
const serverAccessLevel = () => null

/** Reports that the client snapshot has read browser state. */
const clientHydrated = () => false
/** Reports that server rendering has not read browser state. */
const serverHydrated = () => true

/** Returns the actual bearer session identity, using null for server rendering and logged-out state. */
export function useSessionToken(): string | null {
  return React.useSyncExternalStore(subscribe, getToken, serverToken)
}

/** Returns reactive authentication/access state and login/logout actions for the current browser session. */
export function useAuth(): AuthState {
  const token = useSessionToken()
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

  /** Authenticates credentials and stores the returned access level before publishing the new token; refusals reject. */
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

    if (nextAccessLevel) setAccessLevel(nextAccessLevel)
    setToken(nextToken)
  }

  /** Clears the browser session and notifies all auth subscribers. */
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
