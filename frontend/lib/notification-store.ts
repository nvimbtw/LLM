import type { ReactNode } from "react"
import {
  expireNotification,
  MAX_NOTIFICATIONS,
  notificationKey,
  notificationPolicy,
  parseNotifications,
  serializeNotifications,
  type NotificationRecord,
  type NotificationLifetime,
  type NotificationSeverity,
} from "./notifications"

export interface NotificationInput {
  title?: ReactNode
  description?: ReactNode
  type?: NotificationSeverity
  lifetime?: NotificationLifetime
  timeout?: number
  dedupKey?: string
}
export interface LiveNotification extends NotificationRecord {
  popup: boolean
  titleNode?: ReactNode
  descriptionNode?: ReactNode
  expiresAt: number | null
  dedupKey?: string
}
export interface NotificationSnapshot {
  token: string | null
  items: readonly LiveNotification[]
  error: string | null
  ready: boolean
}
export interface NotificationEffects {
  token: () => string | null
  read: (key: string) => string | null
  write: (key: string, value: string) => void
  popup: (item: LiveNotification) => void
  close: (id: string) => void
  now: () => number
}

/** Owns one session's retained history and live popups, with one bounded expiry timer and guarded storage effects. */
export class NotificationStore {
  private state: NotificationSnapshot = {
    token: null,
    items: [],
    error: null,
    ready: true,
  }
  private accountId: string | null = null
  private storageReadable = false
  private listeners = new Set<() => void>()
  private timer: ReturnType<typeof setTimeout> | null = null
  private sequence = 0
  /** Captures the concrete session, storage, clock and popup adapters owned by this store. */
  constructor(private readonly effects: NotificationEffects) {}

  /** Returns the stable snapshot consumed by React subscriptions. */
  getSnapshot = () => this.state
  /** Subscribes to changes and returns this listener's cleanup. */
  subscribe = (listener: () => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }
  /** Publishes a complete state and schedules the next finite popup expiration. */
  private publish(next: NotificationSnapshot) {
    this.state = next
    if (this.timer) clearTimeout(this.timer)
    this.timer = null
    const deadlines = next.items.flatMap((item) =>
      item.popup && item.expiresAt !== null ? [item.expiresAt] : []
    )
    if (deadlines.length)
      this.timer = setTimeout(
        () => this.expireDue(),
        Math.max(1, Math.min(...deadlines) - this.effects.now())
      )
    this.listeners.forEach((listener) => listener())
  }
  /** Replaces session ownership, clearing private popups before accepting new events. */
  setSession(token: string | null) {
    if (this.state.token === token) return
    const old = this.state.items
    this.accountId = null
    this.storageReadable = false
    this.publish({ token, items: [], error: null, ready: !token })
    old.forEach((item) => this.effects.close(item.id))
  }
  /** Masks stale sessions before any imperative mutation or late callback can act. */
  private current(): boolean {
    const token = this.effects.token()
    if (token !== this.state.token) {
      this.setSession(token)
      return false
    }
    return true
  }
  /** Loads verified account history only for the captured current session, preserving events emitted during identity loading. */
  restore(token: string, accountId: string) {
    if (!this.current() || token !== this.state.token) return
    this.accountId = accountId
    try {
      const raw = this.effects.read(notificationKey(accountId))
      const saved = raw ? parseNotifications(raw) : []
      const liveIds = new Set(this.state.items.map((item) => item.id))
      const restored: LiveNotification[] = saved
        .filter((item) => !liveIds.has(item.id))
        .map((item) => ({
          ...item,
          popup: item.lifetime === "sticky",
          expiresAt: null,
        }))
      this.storageReadable = true
      this.publish({
        ...this.state,
        ready: true,
        error: null,
        items: [...restored, ...this.state.items].slice(-MAX_NOTIFICATIONS),
      })
      restored
        .filter((item) => item.popup)
        .forEach((item) => this.effects.popup(item))
      this.persist()
    } catch (cause) {
      this.fail(cause)
    }
  }
  /** Reports identity/storage failure visibly while retaining the current in-memory history for retry. */
  fail(cause: unknown) {
    const message =
      cause instanceof Error
        ? cause.message
        : "Notification history could not be saved. Try again."
    if (this.state.error !== message)
      this.publish({ ...this.state, error: message })
  }
  /** Persists only a still-current verified account whose existing history was successfully read. */
  private persist() {
    if (!this.current() || !this.accountId || !this.storageReadable) return
    try {
      this.effects.write(
        notificationKey(this.accountId),
        serializeNotifications(this.state.items)
      )
      if (this.state.error) this.publish({ ...this.state, error: null })
    } catch {
      this.fail(
        new Error(
          "Notification history could not be saved in this browser. Try again."
        )
      )
    }
  }
  /** Retries a failed write without discarding in-memory events. */
  retryWrite() {
    this.persist()
  }
  /** Adds one bounded notification or returns an existing matching event id. */
  add(input: NotificationInput, title: string, description: string): string {
    this.current()
    const existing = input.dedupKey
      ? this.state.items.find((item) => item.dedupKey === input.dedupKey)
      : undefined
    if (existing) return existing.id
    const severity = input.type ?? "default"
    const policy = notificationPolicy(severity, input.lifetime, input.timeout)
    const now = this.effects.now()
    const item: LiveNotification = {
      id: `notice-${now}-${++this.sequence}`,
      title: title.slice(0, 4000),
      description: description.slice(0, 4000),
      titleNode: input.title,
      descriptionNode: input.description,
      severity,
      lifetime: policy.lifetime,
      createdAt: now,
      popup: true,
      expiresAt: policy.timeout ? now + policy.timeout : null,
      dedupKey: input.dedupKey,
    }
    const next = [...this.state.items, item].slice(-MAX_NOTIFICATIONS)
    const retainedIds = new Set(next.map((entry) => entry.id))
    const evicted = this.state.items.filter(
      (entry) => !retainedIds.has(entry.id)
    )
    this.publish({ ...this.state, items: next })
    evicted.forEach((entry) => this.effects.close(entry.id))
    this.effects.popup(item)
    this.persist()
    return item.id
  }
  /** Removes one entry from both views, or dismisses every notification when no id is supplied. */
  dismiss(id?: string) {
    if (!this.current()) return
    const removed = this.state.items.filter((item) => !id || item.id === id)
    this.publish({
      ...this.state,
      items: this.state.items.filter((item) => id && item.id !== id),
    })
    removed.forEach((item) => this.effects.close(item.id))
    this.persist()
  }
  /** Treats Base UI close/swipe as dismissal only while the store still owns a visible popup. */
  popupClosed(id: string) {
    if (this.state.items.some((item) => item.id === id && item.popup))
      this.dismiss(id)
  }
  /** Expires due popups without deleting retained history and owns the next timer. */
  private expireDue() {
    if (!this.current()) return
    const due = this.state.items.filter(
      (item) =>
        item.popup &&
        item.expiresAt !== null &&
        item.expiresAt <= this.effects.now()
    )
    let items = [...this.state.items]
    due.forEach((item) => {
      items = expireNotification(items, item.id)
    })
    this.publish({ ...this.state, items })
    due.forEach((item) => this.effects.close(item.id))
  }
  /** Restarts deadlines after a provider remount or StrictMode lifecycle replay. */
  resume() {
    this.publish(this.state)
  }
  /** Releases popup deadlines when the provider unmounts. */
  dispose() {
    if (this.timer) clearTimeout(this.timer)
    this.timer = null
  }
}
