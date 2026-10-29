import { WS_URL, handleSessionExpired } from "./api"

export type ServiceState =
  | "online"
  | "reconnecting"
  | "offline"
  | "not_configured"
export interface ServiceStatus {
  id: string
  state: ServiceState
  detail: string | null
}

const IDS = [
  "llm-engine",
  "data-engine",
  "gyms-engine",
  "bench-engine",
  "just-bash",
  "searxng",
] as const
const STATES: readonly string[] = [
  "online",
  "reconnecting",
  "offline",
  "not_configured",
]

/** Decode a complete service snapshot, rejecting malformed or incomplete wire data. */
export function decodeServiceSnapshot(value: unknown): ServiceStatus[] | null {
  if (
    !value ||
    typeof value !== "object" ||
    !("type" in value) ||
    value.type !== "services" ||
    !("services" in value) ||
    !Array.isArray(value.services)
  )
    return null
  const items: ServiceStatus[] = []
  for (const item of value.services as unknown[]) {
    if (
      !item ||
      typeof item !== "object" ||
      !("id" in item) ||
      typeof item.id !== "string" ||
      !IDS.some((id) => id === item.id) ||
      !("state" in item) ||
      typeof item.state !== "string" ||
      !STATES.includes(item.state) ||
      !("detail" in item) ||
      (item.detail !== null && typeof item.detail !== "string")
    )
      return null
    items.push({
      id: item.id,
      state: item.state as ServiceState,
      detail: item.detail,
    })
  }
  return items.length === IDS.length &&
    new Set(items.map((item) => item.id)).size === IDS.length
    ? items
    : null
}

/** Own one authenticated dashboard socket, reconnecting until disposed and expiring stale snapshots. */
export function subscribeServiceHealth(
  token: string,
  onSnapshot: (items: ServiceStatus[] | null) => void
): () => void {
  let socket: WebSocket | null = null
  let retry: ReturnType<typeof setTimeout> | null = null
  let stale: ReturnType<typeof setTimeout> | null = null
  let disposed = false
  let denied = false

  /** Close a connection that has not delivered a valid snapshot within the freshness window. */
  function armFreshness(current: WebSocket) {
    if (stale) clearTimeout(stale)
    stale = setTimeout(() => {
      onSnapshot(null)
      current.close()
    }, 35_000)
  }

  /** Open the next connection for this captured session and ignore callbacks after disposal. */
  function connect() {
    if (disposed || denied) return
    const current = new WebSocket(`${WS_URL}/ws/services`)
    socket = current
    armFreshness(current)
    current.onopen = () => {
      if (!disposed && socket === current)
        current.send(JSON.stringify({ token }))
    }
    current.onmessage = (event) => {
      if (disposed || socket !== current) return
      let value: unknown
      try {
        value = JSON.parse(event.data)
      } catch {
        return
      }
      if (
        value &&
        typeof value === "object" &&
        "type" in value &&
        value.type === "error" &&
        "error" in value
      ) {
        if (value.error === "authentication failed") handleSessionExpired(token)
        if (
          value.error === "authentication failed" ||
          value.error === "admin only"
        ) {
          denied = true
          onSnapshot(null)
          current.close()
        }
        return
      }
      const snapshot = decodeServiceSnapshot(value)
      if (!snapshot) return
      onSnapshot(snapshot)
      armFreshness(current)
    }
    current.onclose = () => {
      if (disposed || socket !== current) return
      socket = null
      if (stale) clearTimeout(stale)
      onSnapshot(null)
      if (!denied) retry = setTimeout(connect, 2_000)
    }
    current.onerror = () => {
      if (!disposed && socket === current) current.close()
    }
  }
  connect()
  return () => {
    disposed = true
    if (retry) clearTimeout(retry)
    if (stale) clearTimeout(stale)
    socket?.close()
  }
}
