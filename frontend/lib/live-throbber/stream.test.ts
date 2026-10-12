import { test } from "node:test"
import assert from "node:assert/strict"
import { ArtStream, validTokenId } from "./stream"
import {
  expression,
  HEADER_FLOATS,
  NODE_COUNT,
  PARAMETER_FLOATS,
} from "./program"

test("IDs cover the unsigned 32-bit range; invalid IDs use text", () => {
  for (const id of [0, 1, 255, 256, 0xffffffff]) assert.ok(validTokenId(id))
  for (const id of [-1, 0x100000000, 0.5, NaN, Infinity, "0", null, undefined])
    assert.equal(validTokenId(id), false)
  const invalid = new ArtStream("turn")
  const text = new ArtStream("turn")
  invalid.push("snow 雪 ❄", -1)
  text.push("snow 雪 ❄")
  assert.deepEqual(invalid.snapshot, text.snapshot)
})

test("ordered IDs determine art independently of decoded text", () => {
  const first = new ArtStream("turn")
  const replay = new ArtStream("turn")
  const reversed = new ArtStream("turn")
  for (const id of [0, 256, 0xffffffff]) {
    first.push("a", id)
    replay.push("different decoded text", id)
  }
  for (const id of [0xffffffff, 256, 0]) reversed.push("a", id)
  assert.deepEqual(first.snapshot, replay.snapshot)
  assert.notEqual(first.snapshot.seed, reversed.snapshot.seed)
  assert.equal(first.snapshot.source, "token ids")
})

test("zero, Unicode, domain and delta boundaries participate in the seed", () => {
  const seed = (tokens: [string, number?][]) => {
    const stream = new ArtStream("turn")
    for (const [text, id] of tokens) stream.push(text, id)
    return stream.snapshot.seed
  }
  assert.notEqual(seed([["", 0]]), seed([["", undefined]]))
  assert.notEqual(seed([["雪"]]), seed([["雪❄"]]))
  assert.notEqual(seed([["ab"]]), seed([["a"], ["b"]]))
  assert.notEqual(seed([["0"]]), seed([["0", 0]]))
})

test("reset replays a turn; unsubscribe releases notifications", () => {
  const stream = new ArtStream("turn")
  const initial = { ...stream.snapshot }
  let notifications = 0
  const unsubscribe = stream.subscribe(() => notifications++)
  stream.push("", 0)
  stream.push("fallback")
  assert.equal(stream.snapshot.source, "mixed")
  const completed = { ...stream.snapshot }
  stream.reset("turn")
  assert.deepEqual(stream.snapshot, initial)
  stream.push("", 0)
  stream.push("fallback")
  assert.deepEqual(stream.snapshot, completed)
  assert.equal(notifications, 5)
  unsubscribe()
  stream.wake()
  assert.equal(notifications, 5)
})

test("large bursts retain constant stream state and fixed grammar", () => {
  const stream = new ArtStream("turn")
  const structure = stream.snapshot.structure
  const before = expression(stream.snapshot.seed, structure)
  for (let id = 0; id < 100000; id++) stream.push("", id)
  assert.equal(stream.snapshot.count, 100000)
  assert.equal(stream.snapshot.structure, structure)
  assert.deepEqual(Object.keys(stream.snapshot).sort(), [
    "count",
    "seed",
    "source",
    "structure",
  ])
  const after = expression(stream.snapshot.seed, structure)
  assert.equal(after.length, NODE_COUNT * 4)
  assert.equal(PARAMETER_FLOATS, HEADER_FLOATS + after.length)
  assert.ok(after.every(Number.isFinite))
  assert.notDeepEqual(before, after)
  for (let node = 0; node < NODE_COUNT; node++) {
    assert.equal(before[node * 4], after[node * 4])
    assert.ok(after[node * 4] >= 0 && after[node * 4] <= 3)
  }
  assert.deepEqual(after, expression(stream.snapshot.seed, structure))
})
