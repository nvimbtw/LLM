import { test } from "node:test"
import assert from "node:assert/strict"
import { NotificationStore, type LiveNotification } from "./notification-store"
import {
  loadNotificationAccount,
  notificationKey,
  notificationPolicy,
  parseNotifications,
  serializeNotifications,
  MAX_NOTIFICATIONS,
} from "./notifications"

const alice = "30000000-0000-4000-8000-000000000001"
const bob = "30000000-0000-4000-8000-000000000002"

/** Supplies isolated session/storage/clock adapters while recording popup effects for lifecycle assertions. */
function fixture() {
  let token: string | null = "opaque-a"
  let now = 1000
  let storageFailure = false
  const storage = new Map<string, string>()
  const popups: LiveNotification[] = []
  const closed: string[] = []
  const writes: string[] = []
  const store = new NotificationStore({
    token: () => token,
    now: () => now,
    read: (key) => {
      if (storageFailure) throw new Error("Storage blocked")
      return storage.get(key) ?? null
    },
    write: (key, value) => {
      if (storageFailure) throw new Error("Quota exceeded")
      storage.set(key, value)
      writes.push(key)
    },
    popup: (item) => {
      popups.push(item)
    },
    close: (id) => {
      closed.push(id)
    },
  })
  store.setSession(token)
  return {
    store,
    storage,
    popups,
    closed,
    writes,
    session: (value: string | null) => {
      token = value
    },
    advance: (milliseconds: number) => {
      now += milliseconds
      store.resume()
    },
    failStorage: (value: boolean) => {
      storageFailure = value
    },
  }
}

test("severity and lifetime have independent explicit overrides", () => {
  assert.deepEqual(notificationPolicy("success"), {
    lifetime: "transient",
    timeout: 4000,
  })
  assert.deepEqual(notificationPolicy("warning"), {
    lifetime: "timed",
    timeout: 6000,
  })
  assert.deepEqual(notificationPolicy("error"), {
    lifetime: "sticky",
    timeout: 0,
  })
  assert.deepEqual(notificationPolicy("error", "transient", 20), {
    lifetime: "transient",
    timeout: 20,
  })
  assert.deepEqual(notificationPolicy("success", "sticky"), {
    lifetime: "sticky",
    timeout: 0,
  })
})

test("expiry keeps timed history, removes transient events, and explicit dismissal removes both views", async () => {
  const f = fixture()
  try {
    f.store.restore("opaque-a", alice)
    const timed = f.store.add({ timeout: 10 }, "Timed", "Retained")
    const transient = f.store.add(
      { type: "success", timeout: 10, lifetime: "transient" },
      "Done",
      ""
    )
    const sticky = f.store.add({ type: "error" }, "Failed", "")
    f.advance(11)
    await new Promise((resolve) => setTimeout(resolve, 5))
    assert.equal(
      f.store.getSnapshot().items.find((item) => item.id === timed)?.popup,
      false
    )
    assert.equal(
      f.store.getSnapshot().items.some((item) => item.id === transient),
      false
    )
    assert.equal(
      f.store.getSnapshot().items.find((item) => item.id === sticky)?.popup,
      true
    )
    f.store.popupClosed(timed)
    assert.equal(
      f.store.getSnapshot().items.some((item) => item.id === timed),
      true
    )
    f.store.dismiss(timed)
    assert.equal(
      f.store.getSnapshot().items.some((item) => item.id === timed),
      false
    )
    f.store.popupClosed(sticky)
    assert.equal(f.store.getSnapshot().items.length, 0)
  } finally {
    f.store.dispose()
  }
})

test("verified durable identity survives new bearers, while stale profiles and logout cannot read or write another account", () => {
  const f = fixture()
  try {
    f.store.restore("opaque-a", alice)
    f.store.add({ type: "error" }, "Alice private", "")
    const saved = f.storage.get(notificationKey(alice))!
    f.session("opaque-b")
    f.store.restore("opaque-a", alice)
    assert.equal(f.store.getSnapshot().items.length, 0)
    assert.equal(f.storage.get(notificationKey(alice)), saved)
    f.store.restore("opaque-b", bob)
    assert.equal(f.store.getSnapshot().items.length, 0)
    f.store.add({}, "Bob private", "")
    f.session("new-opaque-a")
    f.store.setSession("new-opaque-a")
    f.store.restore("new-opaque-a", alice)
    assert.deepEqual(
      f.store.getSnapshot().items.map((item) => item.title),
      ["Alice private"]
    )
    assert.equal(f.store.getSnapshot().items[0].popup, true)
    f.session(null)
    f.store.dismiss()
    assert.equal(f.store.getSnapshot().items.length, 0)
    assert.equal(f.storage.get(notificationKey(alice)), saved)
  } finally {
    f.store.dispose()
  }
})

test("unknown identity and failed reads never authorize overwriting history; storage failures retain memory and retry", () => {
  const f = fixture()
  try {
    f.store.add({}, "Before identity", "")
    assert.equal(f.writes.length, 0)
    f.store.fail(new Error("Profile unavailable"))
    f.store.add({}, "Still unknown", "")
    assert.equal(f.writes.length, 0)
    f.failStorage(true)
    f.store.restore("opaque-a", alice)
    assert.match(f.store.getSnapshot().error!, /Storage blocked/)
    assert.equal(f.writes.length, 0)
    f.failStorage(false)
    f.store.restore("opaque-a", alice)
    assert.equal(f.store.getSnapshot().items.length, 2)
    f.failStorage(true)
    f.store.add({ type: "error" }, "Unsaved", "")
    assert.match(f.store.getSnapshot().error!, /could not be saved/)
    f.failStorage(false)
    f.store.retryWrite()
    assert.equal(f.store.getSnapshot().error, null)
    assert.equal(
      parseNotifications(f.storage.get(notificationKey(alice))!).length,
      3
    )
  } finally {
    f.store.dispose()
  }
})

test("deduplication, bounds and plain-text persistence survive repeated effects and reject malformed storage", () => {
  const f = fixture()
  try {
    f.store.restore("opaque-a", alice)
    const id = f.store.add({ dedupKey: "error-episode" }, "Duplicate", "")
    assert.equal(
      f.store.add({ dedupKey: "error-episode" }, "Duplicate", ""),
      id
    )
    for (let i = 0; i < 120; i++) f.store.add({}, `Notice ${i}`, "")
    assert.equal(f.store.getSnapshot().items.length, MAX_NOTIFICATIONS)
    assert.equal(
      parseNotifications(serializeNotifications(f.store.getSnapshot().items))
        .length,
      MAX_NOTIFICATIONS
    )
    assert.throws(() => parseNotifications('{"version":2,"items":[]}'))
    assert.throws(() => parseNotifications('{"version":1,"items":[{}]}'))
    const row = f.store.getSnapshot().items[0]
    assert.throws(() =>
      parseNotifications(JSON.stringify({ version: 1, items: [row, row] }))
    )
    assert.throws(() =>
      parseNotifications(
        JSON.stringify({ version: 1, items: [{ ...row, createdAt: 1e30 }] })
      )
    )
    assert.throws(() => notificationKey("opaque-session"))
  } finally {
    f.store.dispose()
  }
})

test("profile verification uses the captured bearer and rejects malformed or failed identity responses", async () => {
  const original = globalThis.fetch
  globalThis.fetch = (async (_url, init) => {
    assert.equal(
      new Headers(init?.headers).get("Authorization"),
      "Bearer captured"
    )
    return Response.json({ id_user: alice })
  }) as typeof fetch
  try {
    assert.equal(await loadNotificationAccount("captured"), alice)
    for (const response of [
      Response.json({ id: alice }),
      Response.json({ id_user: "opaque" }),
      new Response("Unavailable", { status: 503 }),
    ]) {
      globalThis.fetch = (async () => response) as typeof fetch
      await assert.rejects(loadNotificationAccount("captured"))
    }
  } finally {
    globalThis.fetch = original
  }
})
