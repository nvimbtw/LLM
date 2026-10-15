import { test } from "node:test"
import assert from "node:assert/strict"
import { estimateTrainingParameters } from "./training-architecture"

test("parameter estimates count tied embeddings once and include the configured blocks and layer norms", () => {
  const shape = { dimensions: 64, layers: 2, ff_multiplier: 4 }
  assert.equal(estimateTrainingParameters(shape, 1024), 165120)
  assert.equal(
    estimateTrainingParameters(shape, 2048)! -
      estimateTrainingParameters(shape, 1024)!,
    64 * 1024
  )
  assert.equal(
    estimateTrainingParameters({ ...shape, layers: 3 }, 1024)! -
      estimateTrainingParameters(shape, 1024)!,
    49728
  )
})

test("unknown vocabularies and invalid or unrepresentable draft shapes have no parameter estimate", () => {
  const shape = { dimensions: 64, layers: 2, ff_multiplier: 4 }
  assert.equal(estimateTrainingParameters(shape, null), null)
  for (const dimensions of [
    0,
    -1,
    Infinity,
    NaN,
    0.5,
    Number.MAX_SAFE_INTEGER,
  ]) {
    assert.equal(
      estimateTrainingParameters({ ...shape, dimensions }, 1024),
      null
    )
  }
})
