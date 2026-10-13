import { test } from "node:test"
import assert from "node:assert/strict"
import { ArtMorph } from "./morph"
import { TARGET_INTERVAL_MS } from "./sampler"

test("equal time steps change coefficients by equal amounts", () => {
  const morph = new ArtMorph(new Float32Array([0, 2]))
  morph.retarget(new Float32Array([1, 0]), 100)
  for (let step = 1; step <= 4; step++) {
    const active = morph.advance(100 + (step * TARGET_INTERVAL_MS) / 4)
    assert.deepEqual(Array.from(morph.values), [step / 4, 2 - step / 2])
    assert.equal(active, step < 4)
  }
  assert.equal(morph.advance(10000), false)
  assert.deepEqual(Array.from(morph.values), [1, 0])
})

test("retargeting between frames preserves position and begins a linear segment", () => {
  const morph = new ArtMorph(new Float32Array([0]))
  morph.retarget(new Float32Array([1]), 0)
  morph.advance(TARGET_INTERVAL_MS / 4)
  morph.retarget(new Float32Array([2]), TARGET_INTERVAL_MS / 2)
  assert.equal(morph.values[0], 0.5)
  morph.advance(TARGET_INTERVAL_MS)
  assert.equal(morph.values[0], 1.25)
  morph.advance(TARGET_INTERVAL_MS * 1.5)
  assert.equal(morph.values[0], 2)
})

test("a new turn or reduced motion can set the target immediately", () => {
  const morph = new ArtMorph(new Float32Array([0]))
  morph.retarget(new Float32Array([1]), 0)
  morph.reset(new Float32Array([-1]))
  assert.equal(morph.advance(100), false)
  assert.equal(morph.values[0], -1)
})
