import { test } from "node:test"
import assert from "node:assert/strict"
import { decodeServiceSnapshot, subscribeServiceHealth } from "./service-health"

const services = [
  "llm-engine",
  "data-engine",
  "gyms-engine",
  "bench-engine",
  "just-bash",
  "searxng",
].map((id) => ({ id, state: "online", detail: null }))

/** Capture browser socket events without creating a network connection. */
class FakeSocket {
  static latest: FakeSocket
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  /** Record the most recently created socket. */
  constructor() {
    FakeSocket.latest = this
  }
  /** Capture a sent frame. */
  send(value: string) {
    this.sent.push(value)
  }
  /** Simulate browser close. */
  close() {
    this.onclose?.()
  }
}

test("snapshots require all six unique services and recognized states", () => {
  assert.equal(decodeServiceSnapshot({ type: "services", services })?.length, 6)
  assert.equal(
    decodeServiceSnapshot({ type: "services", services: services.slice(1) }),
    null
  )
  assert.equal(
    decodeServiceSnapshot({
      type: "services",
      services: [...services.slice(0, 5), services[0]],
    }),
    null
  )
  assert.equal(
    decodeServiceSnapshot({
      type: "services",
      services: [{ ...services[0], state: "unknown" }, ...services.slice(1)],
    }),
    null
  )
})

test("disconnect clears current data and disposal ignores delayed events", () => {
  const original = globalThis.WebSocket
  globalThis.WebSocket = FakeSocket as unknown as typeof WebSocket
  try {
    const observations: unknown[] = []
    const dispose = subscribeServiceHealth("token-A", (value) =>
      observations.push(value)
    )
    const socket = FakeSocket.latest
    socket.onopen?.()
    assert.deepEqual(socket.sent, ['{"token":"token-A"}'])
    socket.onmessage?.({ data: JSON.stringify({ type: "services", services }) })
    assert.equal(observations.length, 1)
    socket.onclose?.()
    assert.equal(observations[1], null)
    dispose()
    socket.onopen?.()
    socket.onmessage?.({ data: JSON.stringify({ type: "services", services }) })
    assert.equal(socket.sent.length, 1)
    assert.equal(observations.length, 2)
  } finally {
    globalThis.WebSocket = original
  }
})
