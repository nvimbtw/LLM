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

// Fired when a request proves the stored session is no longer accepted. The app
// shell answers it with a client navigation, so expiry never reloads the
// document and discards in-page state.
export const SESSION_EXPIRED_EVENT = "session-expired"

let expiryReportedFor: string | null = null
let expirySubscribers = 0

function emitSessionChanged() {
  window.dispatchEvent(new Event(SESSION_CHANGED_EVENT))
}

export function setToken(token: string) {
  expiryReportedFor = null
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

/**
 * Registers the app shell's client-navigation handler for a rejected session
 * and returns its cleanup. While a handler is mounted, expiry never reloads the
 * document; without one, `handleSessionExpired` falls back to a document load.
 */
export function subscribeSessionExpiry(onExpired: () => void): () => void {
  expirySubscribers += 1
  window.addEventListener(SESSION_EXPIRED_EVENT, onExpired)
  return () => {
    expirySubscribers -= 1
    window.removeEventListener(SESSION_EXPIRED_EVENT, onExpired)
  }
}

/**
 * Clears the session that a 401 proved invalid and asks the mounted shell to
 * leave the page without a document reload. `expectedToken` is the token the
 * failed request was sent with, so a late rejection from a replaced session
 * cannot clear the newer one. Returns whether a stored session was cleared;
 * repeats on the same path are ignored so a burst redirects once.
 */
export function handleSessionExpired(
  expectedToken: string | null = getToken()
): boolean {
  const currentToken = getToken()
  if (expectedToken !== currentToken) return false
  const cleared = currentToken !== null
  if (cleared) clearSession()

  const path = window.location.pathname
  if (expiryReportedFor === path) return cleared
  expiryReportedFor = path

  if (expirySubscribers > 0) {
    window.dispatchEvent(new Event(SESSION_EXPIRED_EVENT))
  } else if (path !== "/login") {
    window.location.assign("/login")
  }
  return cleared
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
      handleSessionExpired(currentToken)
    throw new Error("Your session has expired. Please log in again.")
  }
  return response
}
