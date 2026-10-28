import { test } from "node:test"
import assert from "node:assert/strict"
import { attachJob } from "./jobs-socket"

/** Provides a deterministic browser socket seam for auth, late-frame and cleanup behavior without opening a network listener. */
class FakeSocket {
  static latest: FakeSocket
  sent: string[] = []
  closes = 0
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  /** Records the created attachment without connecting a server. */
  constructor(readonly url: string) {
    FakeSocket.latest = this
  }
  /** Retains exact sent auth/control frames for assertions. */
  send(value: string): void {
    this.sent.push(value)
  }
  /** Records observation release, without generating a durable cancellation. */
  close(): void {
    this.closes++
  }
}

test("job sockets authenticate with captured identity and detach suppresses late callbacks without cancelling", () => {
  const original = globalThis.WebSocket
  globalThis.WebSocket = FakeSocket as unknown as typeof WebSocket
  try {
    let frames = 0
    let errors = 0
    const attachment = attachJob(
      "one",
      {
        onFrame: () => {
          frames++
        },
        onError: () => {
          errors++
        },
      },
      "captured-A"
    )
    const socket = FakeSocket.latest
    socket.onopen?.()
    assert.deepEqual(
      socket.sent.map((item) => JSON.parse(item)),
      [{ token: "captured-A" }]
    )
    socket.onmessage?.({ data: "null" })
    socket.onmessage?.({ data: "[]" })
    assert.equal(frames, 0)
    attachment.detach()
    socket.onopen?.()
    socket.onmessage?.({ data: '{"type":"done"}' })
    socket.onerror?.()
    attachment.cancel()
    assert.equal(frames, 0)
    assert.equal(errors, 0)
    assert.equal(socket.sent.length, 1)
    assert.equal(socket.closes, 1)
  } finally {
    globalThis.WebSocket = original
  }
})

test("terminal frames release a stream once and explicit cancellation is the sole control frame", () => {
  const original = globalThis.WebSocket
  globalThis.WebSocket = FakeSocket as unknown as typeof WebSocket
  try {
    let frames = 0
    const attachment = attachJob(
      "two",
      {
        onFrame: () => {
          frames++
        },
      },
      "captured-B"
    )
    const socket = FakeSocket.latest
    socket.onopen?.()
    attachment.cancel()
    socket.onmessage?.({ data: '{"type":"done"}' })
    socket.onmessage?.({ data: '{"type":"done"}' })
    assert.equal(frames, 1)
    assert.equal(socket.closes, 1)
    assert.deepEqual(
      socket.sent.map((item) => JSON.parse(item)),
      [{ token: "captured-B" }, { type: "cancel" }]
    )
  } finally {
    globalThis.WebSocket = original
  }
})
