import { test } from "node:test"
import assert from "node:assert/strict"
import { FrameClock, FRAME_MS, REDUCED_FRAME_MS } from "./frame-clock"

test("60 fps pacing survives timestamp rounding and different display refresh rates", () => {
  for (const refreshRate of [59.94, 60, 90, 120, 144, 165, 240]) {
    const clock = new FrameClock(FRAME_MS)
    let count = 0
    for (let frame = 0; frame < refreshRate * 10; frame++) {
      const now = Math.round((frame * 10000) / refreshRate) / 10
      if (clock.take(now)) count++
    }
    assert.ok(
      count >= 598 && count <= 602,
      `${refreshRate} Hz: ${count} frames`
    )
  }
})

test("missed frames use the latest time without a catch-up burst", () => {
  const clock = new FrameClock(FRAME_MS)
  assert.equal(clock.take(0), true)
  assert.equal(clock.take(8), false)
  assert.equal(clock.take(2000), true)
  assert.equal(clock.take(2000), false)
  assert.equal(clock.take(2008), false)
  assert.equal(clock.take(2016.7), true)
  clock.reset()
  assert.equal(clock.take(2017), true)
})

test("reduced motion stays at two frames per second", () => {
  const clock = new FrameClock(REDUCED_FRAME_MS)
  let count = 0
  for (let now = 0; now < 10000; now += 10) {
    if (clock.take(now)) count++
  }
  assert.equal(count, 20)
  assert.equal(clock.remaining(9990), 10)
})
