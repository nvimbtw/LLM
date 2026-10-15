import { test } from "node:test"
import assert from "node:assert/strict"
import { createElement } from "react"
import { renderToStaticMarkup } from "react-dom/server"
import { ArchitectureDiagram } from "../components/training/architecture-diagram"
import {
  ARCHITECTURE_STEP_MS,
  ARCHITECTURE_RETURN_MS,
  smooth,
  stageEmphasis,
  type Operation,
  architectureFrame,
  operationProgress,
  pointOnRoute,
  promptTokens,
} from "./architecture-flow"

const cfg = {
  dimensions: 512,
  ctx_window: 1024,
  heads: 8,
  layers: 16,
  ff_multiplier: 4,
  mode: "sft" as const,
}

test("generation samples and appends before reading the extended prompt", () => {
  const step = ARCHITECTURE_STEP_MS
  assert.equal(architectureFrame("generation", 6 * step).step, "sample")
  assert.equal(architectureFrame("generation", 7 * step).step, "append")
  const next = architectureFrame(
    "generation",
    7 * step + ARCHITECTURE_RETURN_MS
  )
  assert.deepEqual(next, { step: "input", progress: 0, cycle: 1 })
  assert.equal(promptTokens(next, "generation").tokens.length, 5)
})

test("appended tokens retain their exact position and identity across the loop seam, including bounded overflow", () => {
  for (const cycle of [0, 1, 2, 3, 30, 10000]) {
    const before = promptTokens(
      { cycle, step: "append", progress: 0.999 },
      "generation"
    )
    const after = promptTokens(
      { cycle: cycle + 1, step: "input", progress: 0 },
      "generation"
    )
    const visibleBefore = [...before.tokens, before.next].filter(
      (token) => token.x >= 0
    )
    assert.deepEqual(after.tokens, visibleBefore)
    assert.ok(after.tokens.length <= 6)
    assert.ok(after.tokens.every((token) => token.x + token.width <= 278))
  }
})

test("training updates weights before replacing its batch, and never samples or appends", () => {
  const stages = Array.from(
    { length: 10 },
    (_, index) =>
      architectureFrame("training", index * ARCHITECTURE_STEP_MS).step
  )
  assert.deepEqual(stages.slice(6), ["loss", "backward", "optimizer", "update"])
  assert.equal(
    architectureFrame(
      "training",
      9 * ARCHITECTURE_STEP_MS + ARCHITECTURE_RETURN_MS
    ).step,
    "batch"
  )
  const oldBatch = promptTokens(
    { cycle: 0, step: "batch", progress: 0.49 },
    "training"
  )
  const replaced = promptTokens(
    { cycle: 0, step: "batch", progress: 0.51 },
    "training"
  )
  const next = promptTokens(
    architectureFrame(
      "training",
      10 * ARCHITECTURE_STEP_MS + ARCHITECTURE_RETURN_MS
    ),
    "training"
  )
  assert.notDeepEqual(oldBatch.tokens, replaced.tokens)
  assert.deepEqual(replaced.tokens, next.tokens)
  assert.equal(next.tokens.length, 4)
  assert.equal(next.clipped, false)
})

test("operations finish, settle and retain their completed result until the next cycle", () => {
  const operations: Operation[] = [
    "embedding",
    "attention",
    "feedforward",
    "norm",
    "output",
    "sample",
    "loss",
    "backward",
    "optimizer",
  ]
  for (const operation of operations) {
    const view = operation === "sample" ? "generation" : "training"
    assert.equal(
      operationProgress(
        { step: operation, progress: 0.2, cycle: 0 },
        operation,
        view
      ),
      0
    )
    assert.equal(
      operationProgress(
        { step: operation, progress: 0.3, cycle: 0 },
        operation,
        view
      ),
      0
    )
    assert.equal(
      operationProgress(
        { step: operation, progress: 0.82, cycle: 0 },
        operation,
        view
      ),
      1
    )
    assert.equal(
      operationProgress(
        { step: operation, progress: 0.999, cycle: 0 },
        operation,
        view
      ),
      1
    )
    assert.equal(
      operationProgress(
        {
          step: view === "generation" ? "append" : "batch",
          progress: 0.5,
          cycle: 0,
        },
        operation,
        view
      ),
      1
    )
    assert.equal(
      operationProgress(
        { step: "input", progress: 0, cycle: 1 },
        operation,
        view
      ),
      1
    )
    assert.equal(
      operationProgress(
        { step: "input", progress: 0.7, cycle: 1 },
        operation,
        view
      ),
      0
    )
  }
})

test("emphasis and easing have continuous, settled boundaries without independent CSS timers", () => {
  assert.equal(smooth(-1), 0)
  assert.equal(smooth(2), 1)
  assert.ok(smooth(0.001) < 0.00001)
  assert.ok(1 - smooth(0.999) < 0.00001)
  assert.equal(
    stageEmphasis({ step: "embedding", progress: 0, cycle: 0 }, "embedding"),
    0
  )
  assert.equal(
    stageEmphasis({ step: "embedding", progress: 1, cycle: 0 }, "embedding"),
    0
  )
  assert.equal(stageEmphasis(null, "embedding"), 0)
})

test("travel rounds bends while preserving exact connection endpoints", () => {
  const points = [
    { x: 20, y: 10 },
    { x: 20, y: 50 },
    { x: 80, y: 50 },
  ]
  assert.deepEqual(pointOnRoute(points, 0), points[0])
  const bend = pointOnRoute(points, 0.4)
  assert.ok(bend.x > 20 && bend.x < 28)
  assert.ok(bend.y > 42 && bend.y < 50)
  const after = pointOnRoute(points, 0.7)
  assert.ok(after.x > 28 && after.x < 80)
  assert.equal(after.y, 50)
  assert.deepEqual(pointOnRoute(points, 1), points[2])
  assert.deepEqual(pointOnRoute([points[0], points[0]], 0.5), points[0])
})

test("both responsive diagrams show the page's flow and configured architecture without fabricating invalid counts", () => {
  for (const vertical of [false, true]) {
    const generation = renderToStaticMarkup(
      createElement(ArchitectureDiagram, {
        cfg,
        vertical,
        view: "generation",
        frame: null,
      })
    )
    assert.match(generation, /Sampling/)
    assert.match(generation, /× 16 layers/)
    assert.match(generation, /2,048 hidden/)
    assert.doesNotMatch(
      generation,
      /data-flow-node="(?:loss|backward|optimizer)"/
    )
    const training = renderToStaticMarkup(
      createElement(ArchitectureDiagram, {
        cfg,
        vertical,
        view: "training",
        frame: null,
      })
    )
    assert.match(training, /Masked CE/)
    assert.match(training, /Assistant targets/)
    assert.doesNotMatch(training, /Sampling|Append selected token/)
    const invalid = renderToStaticMarkup(
      createElement(ArchitectureDiagram, {
        cfg: { ...cfg, layers: NaN, dimensions: Infinity },
        vertical,
        view: "training",
        frame: null,
      })
    )
    assert.doesNotMatch(invalid, /NaN|Infinity/)
    assert.match(invalid, /× — layers/)
  }
})
