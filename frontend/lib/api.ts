export const API_URL = "http://localhost:8080"

const TOKEN_KEY = "auth_token"
const ACCESS_LEVEL_KEY = "access_level"

export type AccessLevel = "user" | "admin"

export function getToken(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token)
}

export function getAccessLevel(): AccessLevel | null {
  if (typeof window === "undefined") return null
  const raw = localStorage.getItem(ACCESS_LEVEL_KEY)
  return raw ? (raw.toLowerCase() as AccessLevel) : null
}

export function setAccessLevel(level: AccessLevel) {
  localStorage.setItem(ACCESS_LEVEL_KEY, level)
}

/** Clears all stored session state (token + access level). */
export function clearSession() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(ACCESS_LEVEL_KEY)
}

export function authHeaders(): HeadersInit {
  const token = getToken()
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}
