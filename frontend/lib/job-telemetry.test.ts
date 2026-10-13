import { test } from "node:test"
import assert from "node:assert/strict"
import {
  createJobTelemetry,
  jobTelemetryEvent,
  reduceJobTelemetry,
  type JobTelemetry,
} from "./job-telemetry"

/** Reduces relayed fixture frames through the adapter/reducer boundary the job hook uses. */
function frame(
  previous: JobTelemetry,
  value: Record<string, unknown>
): JobTelemetry {
  const event = jobTelemetryEvent(value)
  return event ? reduceJobTelemetry(previous, event) : previous
}

test("build measurements merge across steps without inventing speed at an unmeasured boundary", () => {
  const initial = createJobTelemetry()
  const sampled = frame(initial, {
    type: "tokenize",
    step: "sampling",
    documents: 9,
    elapsed_secs: 1,
    tokens_per_sec: 8,
  })
  const packing = frame(sampled, {
    type: "tokenize",
    step: "packing",
    elapsed_secs: 2,
  })
  assert.equal(packing.tokenize.documents, 9)
  assert.equal(packing.tokenize.step, "packing")
  assert.equal(packing.buildSeries.length, 1)
  assert.equal(initial.buildSeries.length, 0)
  assert.deepEqual(initial.tokenize, {})
})

test("snapshot steps restore one-based epoch coordinates and bounded live charts retain the latest report", () => {
  let state = frame(createJobTelemetry(), {
    type: "snapshot",
    train: { total_chunks: 3 },
    loss_series: [
      [1, 5],
      [3, 3],
      [4, 2],
    ],
    batch_series: [],
    diag_series: [],
    log: [],
  })
  assert.deepEqual(state.loss, [
    { epoch: 1, chunk: 1, loss: 5 },
    { epoch: 1, chunk: 3, loss: 3 },
    { epoch: 2, chunk: 1, loss: 2 },
  ])
  for (let step = 5; step <= 1200; step++)
    state = frame(state, {
      type: "train",
      step,
      epoch: 2,
      chunk: step,
      loss: 1 / step,
      batch_min: 1,
      batch_avg: 2,
      batch_max: 3,
      epoch_done: false,
    })
  assert.ok(state.loss.length <= 500)
  assert.equal(state.loss.at(-1)?.chunk, 1200)
  assert.equal(state.batchSeries.length, 60)
  assert.equal(state.batchSeries[0].step, 1141)
  assert.equal(state.batchSeries.at(-1)?.step, 1200)
})

test("epoch averages omit batch points; sparse RL diagnostics and checkpoint identities retain their meaning", () => {
  const boundary = frame(createJobTelemetry(), {
    type: "train",
    epoch_done: true,
    loss: 0.5,
    epoch: 1,
    chunk: 3,
    batch_avg: 0,
    diag: null,
  })
  assert.equal(boundary.batchSeries.length, 0)
  const practice = frame(boundary, {
    type: "train",
    diag: {
      step: 3,
      pass_rate: 0.7,
      task_sequence: ["math"],
      source_scores: {
        math: { graded: 10, correct: 7, episode_count: 10, score: 0.7 },
      },
    },
  })
  assert.equal(practice.diag[0].pass_rate, 0.7)
  assert.equal(practice.diag[0].grad_layers, undefined)
  assert.equal(practice.diag[0].loss_masked, undefined)
  const saved = frame(practice, {
    type: "checkpoint",
    model_id: "intermediate",
    final: false,
  })
  assert.equal(
    frame(saved, {
      type: "checkpoint",
      model_id: "intermediate",
      final: false,
    }),
    saved
  )
  assert.equal(
    frame(saved, { type: "checkpoint", model_id: "finished", final: true }),
    saved
  )
})

test("simple progress preserves absent values, concatenates deltas and leaves terminal status REST-owned", () => {
  const initial = frame(createJobTelemetry(), {
    type: "snapshot",
    stage: "Scoring",
    fraction: 0.5,
    text: "Start",
  })
  const token = frame(frame(initial, { type: "progress", fraction: 0 }), {
    type: "token",
    value: " now",
  })
  assert.equal(token.stage, "Scoring")
  assert.equal(token.fraction, 0)
  assert.equal(token.text, "Start now")
  assert.equal(frame(token, { type: "done" }), token)
})
