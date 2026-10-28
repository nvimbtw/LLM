import { apiFetch } from "./api"

export type NotificationSeverity = "default" | "success" | "warning" | "error"
export type NotificationLifetime = "sticky" | "timed" | "transient"
export interface NotificationRecord {
  id: string
  title: string
  description: string
  severity: NotificationSeverity
  lifetime: NotificationLifetime
  createdAt: number
}
export const MAX_NOTIFICATIONS = 100
const MAX_TEXT = 4000

/** Separates severity from delivery lifetime, applying explicit overrides before defaults. */
export function notificationPolicy(
  severity: NotificationSeverity,
  lifetime?: NotificationLifetime,
  timeout?: number
) {
  const resolved =
    lifetime ??
    (timeout === 0
      ? "sticky"
      : timeout !== undefined
        ? "timed"
        : severity === "success"
          ? "transient"
          : severity === "error"
            ? "sticky"
            : "timed")
  return {
    lifetime: resolved,
    timeout:
      resolved === "sticky"
        ? 0
        : Math.max(
            1,
            Math.min(
              60000,
              Number.isFinite(timeout) && timeout! > 0
                ? timeout!
                : severity === "success"
                  ? 4000
                  : 6000
            )
          ),
  }
}

/** Builds storage identity only from a verified durable account UUID. */
export function notificationKey(accountId: string): string {
  if (
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
      accountId
    )
  )
    throw new Error("Notification account identity is invalid.")
  return `slate-notifications:v1:${accountId.toLowerCase()}`
}

/** Resolves a captured bearer through the authenticated profile; failed or malformed reads never grant storage access. */
export async function loadNotificationAccount(
  token: string,
  signal?: AbortSignal
): Promise<string> {
  const response = await apiFetch("/user/me", {
    headers: { Authorization: `Bearer ${token}` },
    signal,
  })
  if (!response.ok)
    throw new Error(
      "Notification history could not verify your account. Try again."
    )
  const value: unknown = await response.json()
  if (!isRecord(value) || typeof value.id_user !== "string")
    throw new Error("Notification account identity is invalid.")
  notificationKey(value.id_user)
  return value.id_user
}

/** Narrows a JSON container without accepting null or arrays. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** Validates bounded versioned history; corruption is reported without overwriting its bytes. */
export function parseNotifications(raw: string): NotificationRecord[] {
  if (raw.length > 1000000)
    throw new Error("Notification history is too large.")
  const value: unknown = JSON.parse(raw)
  if (
    !isRecord(value) ||
    value.version !== 1 ||
    !Array.isArray(value.items) ||
    value.items.length > MAX_NOTIFICATIONS
  )
    throw new Error(
      "Notification history is invalid or uses an unsupported version."
    )
  const ids = new Set<string>()
  return value.items.map((item): NotificationRecord => {
    if (
      !isRecord(item) ||
      typeof item.id !== "string" ||
      !item.id ||
      item.id.length > 200 ||
      ids.has(item.id) ||
      typeof item.title !== "string" ||
      item.title.length > MAX_TEXT ||
      typeof item.description !== "string" ||
      item.description.length > MAX_TEXT ||
      !["default", "success", "warning", "error"].includes(
        String(item.severity)
      ) ||
      !["sticky", "timed"].includes(String(item.lifetime)) ||
      typeof item.createdAt !== "number" ||
      !Number.isFinite(item.createdAt) ||
      item.createdAt < 0 ||
      item.createdAt > 8640000000000000
    )
      throw new Error("Notification history contains an invalid entry.")
    ids.add(item.id)
    return {
      id: item.id,
      title: item.title,
      description: item.description,
      severity: item.severity as NotificationSeverity,
      lifetime: item.lifetime as NotificationLifetime,
      createdAt: item.createdAt,
    }
  })
}

/** Serializes only retained plain-text entries within the shared storage bounds. */
export function serializeNotifications(
  items: readonly NotificationRecord[]
): string {
  const raw = JSON.stringify({
    version: 1,
    items: items
      .filter((item) => item.lifetime !== "transient")
      .slice(-MAX_NOTIFICATIONS),
  })
  parseNotifications(raw)
  return raw
}

/** Applies expiration independently from dismissal: retained entries outlive their popup. */
export function expireNotification<
  T extends NotificationRecord & { popup: boolean },
>(items: readonly T[], id: string): T[] {
  return items.flatMap((item) =>
    item.id !== id
      ? [item]
      : item.lifetime === "transient"
        ? []
        : [{ ...item, popup: false }]
  )
}
