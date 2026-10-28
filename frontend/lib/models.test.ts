import { describe, test } from "node:test"
import assert from "node:assert/strict"
import {
  architectureOf,
  estimateParams,
  mapModel,
  modelsFromRegistry,
  tierFor,
  vocabularySizeOf,
  type RawModel,
} from "@/hooks/use-models"
import { estimateTrainingParameters } from "./training-architecture"

const architecture = {
  dimensions: 512,
  heads: 8,
  layers: 16,
  ff_multiplier: 4,
  ctx_window: 1024,
}

/** Builds a model response fixture, allowing each test to vary the wire contract it exercises. */
function model(overrides: Partial<RawModel> = {}): RawModel {
  return {
    model_id: "model-id",
    name: "Mercurian",
    family: "sft",
    description: null,
    parameters: null,
    context: null,
    published: true,
    has_weights: true,
    created_at: null,
    run_id: null,
    parent_id: null,
    checkpoint_step: null,
    checkpoint_index: null,
    ...overrides,
  }
}

describe("resolved model facts", () => {
  test("published models remain selectable when ordinary-user projection removes private parameters", () => {
    const publicRow = model({
      parameters: null,
      parameter_count: 712,
      context: 8,
      definition_id: "a".repeat(64),
    })
    const mapped = modelsFromRegistry([publicRow])
    assert.equal(mapped.length, 1)
    assert.equal(mapped[0].id, publicRow.model_id)
    assert.equal(mapped[0].params, 712)
    assert.equal(mapped[0].definitionId, "a".repeat(64))
  })
  test("an older SFT uses inherited shape and recorded count without needing architecture in its request", () => {
    const raw = model({
      architecture,
      vocab_size: 8192,
      parameter_count: 54600704,
      parameters: { config: { base_model: "base-id" } },
    })
    assert.deepEqual(architectureOf(raw), architecture)
    const mapped = mapModel(raw)
    assert.equal(mapped.params, 54600704)
    assert.equal(mapped.contextWindow, 1024)
    assert.equal(mapped.description, "16-layer transformer, 512d")
    assert.equal(mapped.tier, null)
  })

  test("actual results override stale requested dimensions", () => {
    assert.deepEqual(
      architectureOf(
        model({
          parameters: {
            config: { ...architecture, dimensions: 128 },
            result: { architecture },
          },
        })
      ),
      architecture
    )
  })

  test("explicit unavailable shape cannot become a legacy Transformer from request defaults", () => {
    const raw = model({
      architecture: null,
      parameters: { config: architecture, result: { architecture } },
      parameter_count: 123,
      vocab_size: 320,
    })
    assert.equal(architectureOf(raw), null)
    assert.equal(estimateParams(raw), 123)
    assert.equal(
      architectureOf({ ...raw, architecture: { ...architecture, heads: -1 } }),
      null
    )
  })

  test("legacy nested vocabulary is measured rather than copied from max_vocab", () => {
    const raw = model({
      parameters: {
        config: { ...architecture, max_vocab: 1024 },
        result: { dataset: { vocab_size: 8192 } },
      },
    })
    assert.equal(vocabularySizeOf(raw), 8192)
    assert.equal(
      estimateParams(raw),
      estimateTrainingParameters(architecture, 8192)
    )
    assert.equal(
      vocabularySizeOf(model({ parameters: { config: { max_vocab: 1024 } } })),
      null
    )
  })

  test("a measured parameter count remains available without architecture", () => {
    assert.equal(
      estimateParams(
        model({ parameters: { result: { parameters: 54600704 } } })
      ),
      54600704
    )
  })

  test("invalid dimensions and counts remain unavailable", () => {
    const raw = model({
      architecture: { ...architecture, layers: -1 },
      parameter_count: NaN,
      vocab_size: Infinity,
    })
    assert.equal(architectureOf(raw), null)
    assert.equal(vocabularySizeOf(raw), null)
    assert.equal(estimateParams(raw), null)
  })

  test("a measured zero remains distinct from unavailable and legacy estimated counts", () => {
    const zero = mapModel(model({ architecture: null, parameter_count: 0 }))
    assert.equal(zero.params, 0)
    assert.equal(zero.parameterCountEstimated, false)
    const unknown = mapModel(
      model({ architecture: null, parameter_count: null })
    )
    assert.equal(unknown.params, null)
    assert.equal(unknown.parameterCountEstimated, false)
    const estimated = mapModel(
      model({
        parameters: {
          config: architecture,
          result: { dataset: { vocab_size: 320 } },
        },
      })
    )
    assert.equal(estimated.parameterCountEstimated, true)
    assert.ok(estimated.params! > 0)
  })

  test("size is manually assigned and never inferred from parameter count", () => {
    assert.equal(tierFor(model({ parameter_count: 200_000_000 })), null)
    assert.equal(
      tierFor(model({ size: "fast", parameter_count: 200_000_000 })),
      "fast"
    )
    assert.equal(
      tierFor(model({ size: "flagship", parameter_count: 500 })),
      "flagship"
    )
  })
})
