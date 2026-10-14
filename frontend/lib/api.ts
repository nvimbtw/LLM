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
  // Broadcast like `setToken` does: `useAuth` reads the level as an external
  // store, so a write that stays silent leaves `isAdmin` stale until something
  // else happens to fire the event.
  emitSessionChanged()
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

/** Clears the session and lands the user on /login (token expired/invalid). */
export function handleSessionExpired() {
  clearSession()
  // Guard against a redirect loop: a 401 on the login page (e.g. a stray
  // background fetch) must not reload it, or we spin reloading /login forever.
  if (window.location.pathname !== "/login") {
    window.location.assign("/login")
  }
}

/**
 * Authenticated fetch for API routes: attaches auth headers and, on a 401,
 * logs the user out and redirects to /login instead of surfacing a generic
 * error on a broken page. Use this instead of raw fetch for every
 * authenticated call; the thrown error still rejects the caller's promise so
 * in-flight UI settles. A late 401 can clear only the session that authorized
 * its request. Multipart requests retain the browser-generated content boundary.
 */
export async function apiFetch(
  path: string,
  init: RequestInit = {}
): Promise<Response> {
  const headers = new Headers(authHeaders())
  new Headers(init.headers).forEach((value, key) => headers.set(key, value))
  if (init.body instanceof FormData) headers.delete("Content-Type")

  const response = await fetch(`${API_URL}${path}`, { ...init, headers })
  if (response.status === 401) {
    const currentToken = getToken()
    const currentAuthorization = currentToken ? `Bearer ${currentToken}` : null
    if (headers.get("Authorization") === currentAuthorization)
      handleSessionExpired()
    throw new Error("Your session has expired. Please log in again.")
  }
  return response
}
