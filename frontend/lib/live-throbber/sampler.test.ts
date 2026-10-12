import { test } from "node:test"
import assert from "node:assert/strict"
import { ArtStream } from "./stream"
import {
  ArtSampler,
  TARGET_INTERVAL_MS,
  VISUAL_TARGETS_PER_SECOND,
} from "./sampler"
import { motionPose, motionRate, POSE_FLOATS } from "./motion"

test("fast inference changes visual targets on a fixed clock without a backlog", () => {
  for (const tokensPerSecond of [24, 240, 1000]) {
    const stream = new ArtStream("turn")
    const sampler = new ArtSampler(stream.snapshot)
    const times: number[] = []
    for (let token = 1; token <= tokensPerSecond * 10; token++) {
      stream.push("", token)
      const now = (token * 1000) / tokensPerSecond
      if (sampler.sample(stream.snapshot, now)) times.push(now)
    }
    assert.ok(times.length <= Math.ceil(10 * VISUAL_TARGETS_PER_SECOND))
    for (let i = 1; i < times.length; i++)
      assert.ok(times[i] - times[i - 1] >= TARGET_INTERVAL_MS)
    sampler.sample(stream.snapshot, 11000)
    assert.equal(sampler.snapshot.seed, stream.snapshot.seed)
    assert.equal(sampler.pending(stream.snapshot), false)
    assert.equal(sampler.sample(stream.snapshot, 12000), undefined)
  }
})

test("slow streams use each arrival; a burst coalesces to its newest seed", () => {
  const stream = new ArtStream("turn")
  const sampler = new ArtSampler(stream.snapshot)
  for (let token = 1; token <= 10; token++) {
    stream.push("", token)
    assert.equal(sampler.sample(stream.snapshot, token * 1000), stream.snapshot)
  }
  for (let token = 11; token < 10011; token++) stream.push("", token)
  assert.equal(sampler.sample(stream.snapshot, 10001), undefined)
  assert.equal(sampler.pending(stream.snapshot), true)
  assert.equal(sampler.sample(stream.snapshot, 10800), stream.snapshot)
  assert.equal(sampler.sample(stream.snapshot, 10801), undefined)
})

test("replay and a new turn bypass an old turn's sampling deadline", () => {
  const stream = new ArtStream("turn")
  const sampler = new ArtSampler(stream.snapshot)
  stream.push("", 50)
  sampler.sample(stream.snapshot, 1000)
  stream.reset("turn")
  assert.equal(sampler.sample(stream.snapshot, 1001), stream.snapshot)
  stream.reset("other turn")
  assert.equal(sampler.sample(stream.snapshot, 1002), stream.snapshot)
})

test("every active state has a distinct bounded motion; idle is neutral", () => {
  const modes = [
    "idle",
    "loading",
    "thinking",
    "generating",
    "tool",
    "training",
    "benchmarking",
    "tokenizing",
  ] as const
  const poses = modes.map((mode) => {
    const pose = new Float32Array(POSE_FLOATS)
    motionPose(mode, 1.4, pose)
    return Array.from(pose)
  })
  assert.deepEqual(
    poses[0].filter((_, index) => index !== 6),
    [0, 0, 1, 0, 0, 0, 0]
  )
  assert.equal(
    new Set(poses.map((pose) => JSON.stringify(pose))).size,
    modes.length
  )
  for (const mode of modes) {
    for (const phase of [0, 0.5, 10, 1000000]) {
      const pose = new Float32Array(POSE_FLOATS)
      motionPose(mode, phase, pose)
      assert.ok(pose.every(Number.isFinite))
      assert.ok(pose[2] >= 0.9 && pose[2] <= 1.1)
      assert.ok(Math.abs(pose[0]) <= 0.13 && Math.abs(pose[1]) <= 0.13)
      assert.ok(Math.abs(pose[7]) <= 0.1)
      assert.ok(motionRate(mode, 1000000) >= 0)
    }
  }
  assert.ok(motionRate("loading", 600) < motionRate("loading", 0))
  assert.ok(motionRate("loading", 1000000) >= 0.25)
  assert.equal(motionRate("idle", 0), 0)
})
