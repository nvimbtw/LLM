export const API_URL = "http://localhost:8080"
export const WS_URL = API_URL.replace(/^http/, "ws")

const TOKEN_KEY = "auth_token"
const ACCESS_LEVEL_KEY = "access_level"

export type AccessLevel = "user" | "admin"

export function getToken(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem(TOKEN_KEY)
}

// Fired on the window whenever the session is set or cleared, so providers
// mounted above the auth flow (e.g. ChatsProvider in the root layout) can
// refetch — the "storage" event only fires in *other* tabs.
export const SESSION_CHANGED_EVENT = "session-changed"

function emitSessionChanged() {
  window.dispatchEvent(new Event(SESSION_CHANGED_EVENT))
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token)
  emitSessionChanged()
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
  emitSessionChanged()
}

export function authHeaders(): HeadersInit {
  const token = getToken()
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}
