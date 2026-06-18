"use client"

import * as React from "react"

import {
  API_URL,
  AccessLevel,
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

export interface SignupDetails {
  name: string
  password: string
}

export interface AuthState {
  user: AuthUser | null
  isAuthenticated: boolean
  isAdmin: boolean
  accessLevel: AccessLevel | null
  isLoading: boolean
  login: (credentials: LoginCredentials) => Promise<void>
  signup: (details: SignupDetails) => Promise<void>
  logout: () => Promise<void>
}

export function useAuth(): AuthState {
  const [token, setTokenState] = React.useState<string | null>(null)
  const [accessLevel, setAccessLevelState] = React.useState<AccessLevel | null>(
    null
  )
  const [isLoading, setIsLoading] = React.useState(true)

  React.useEffect(() => {
    setTokenState(getToken())
    setAccessLevelState(getAccessLevel())
    setIsLoading(false)
  }, [])

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

    setToken(nextToken)
    setTokenState(nextToken)

    if (nextAccessLevel) {
      setAccessLevel(nextAccessLevel)
      setAccessLevelState(nextAccessLevel)
    }
  }

  async function signup(_details: SignupDetails) {
    // Handled by useSignup -> POST /user
  }

  async function logout() {
    clearSession()
    setTokenState(null)
    setAccessLevelState(null)
  }

  return {
    user: null,
    isAuthenticated: token !== null,
    isAdmin: accessLevel === "admin",
    accessLevel,
    isLoading,
    login,
    signup,
    logout,
  }
}
