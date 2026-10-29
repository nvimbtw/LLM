import assert from "node:assert/strict"
import { test } from "node:test"

import { FrameCoalescer } from "./frame-coalescer"

/** Replaces the frame APIs with a queue the test runs by hand, the way a browser would. */
function frames() {
  const queued = new Map<number, FrameRequestCallback>()
  let next = 1
  globalThis.requestAnimationFrame = ((callback: FrameRequestCallback) => {
    const id = next++
    queued.set(id, callback)
    return id
  }) as typeof requestAnimationFrame
  globalThis.cancelAnimationFrame = ((id: number) => {
    queued.delete(id)
  }) as typeof cancelAnimationFrame
  return {
    pending: () => queued.size,
    run() {
      const ready = [...queued.entries()]
      queued.clear()
      for (const [, callback] of ready) callback(performance.now())
    },
  }
}

test("a burst delivers only its last value", () => {
  const clock = frames()
  const seen: string[] = []
  const coalescer = new FrameCoalescer<string>((value) => seen.push(value))

  coalescer.push("a")
  coalescer.push("ab")
  coalescer.push("abc")
  assert.equal(clock.pending(), 1)
  assert.deepEqual(seen, [])

  clock.run()
  assert.deepEqual(seen, ["abc"])
})

test("a later value after a frame still arrives", () => {
  const clock = frames()
  const seen: string[] = []
  const coalescer = new FrameCoalescer<string>((value) => seen.push(value))

  coalescer.push("one")
  clock.run()
  coalescer.push("two")
  assert.equal(clock.pending(), 1)
  clock.run()
  assert.deepEqual(seen, ["one", "two"])
})

test("flush delivers a pending value without waiting for a frame", () => {
  const clock = frames()
  const seen: string[] = []
  const coalescer = new FrameCoalescer<string>((value) => seen.push(value))

  coalescer.push("final")
  coalescer.flush()
  assert.deepEqual(seen, ["final"])
  assert.equal(clock.pending(), 0)
  clock.run()
  assert.deepEqual(seen, ["final"])
})

test("dispose drops the pending value and the frame", () => {
  const clock = frames()
  const seen: string[] = []
  const coalescer = new FrameCoalescer<string>((value) => seen.push(value))

  coalescer.push("unwanted")
  coalescer.dispose()
  assert.equal(clock.pending(), 0)
  clock.run()
  assert.deepEqual(seen, [])
})
