import { test } from "node:test"
import assert from "node:assert/strict"
import { DEFAULT_CONFIG } from "@/hooks/use-training"
import { validate, fieldsFor, derived } from "@/components/training/run-config"
import {
  trainingSelectionKey,
  trainingParameterErrors,
} from "./training-selection"
import type { TrainConfig } from "./training"
import type { Recipe } from "./architecture/contracts.generated"

const revision = "30000000-0000-4000-8000-000000000001"
const tokenizer = "30000000-0000-4000-8000-000000000002"

/** Supplies selected metadata inputs with intentionally incompatible legacy shape defaults, without claiming numerical validation. */
function selected(): TrainConfig {
  return {
    ...DEFAULT_CONFIG,
    architecture: { revision_id: revision, parameters: { width: 3 } },
    dimensions: 0,
    ctx_window: 3,
    heads: 13,
    layers: 0,
    max_vocab: 0,
    tokenizer_id: tokenizer,
    datasets: [{ id: "data", count: 0 }],
    batch_size: 3,
  }
}

test("selected recipes use declared structure and current profile checks instead of hidden legacy estimates or head rules", () => {
  const config = selected()
  assert.deepEqual(
    validate(config, false, { key: trainingSelectionKey(config), errors: [] }),
    []
  )
  assert.deepEqual(fieldsFor(config).arch, [])
  assert.deepEqual(derived(config), { headDim: null, dFf: null })
  assert.ok(
    validate(config).some((error) => error.includes("current selected recipe"))
  )
  assert.deepEqual(
    validate(config, false, {
      key: trainingSelectionKey(config),
      errors: ["width: Required parameter."],
    }),
    ["width: Required parameter."]
  )
})

test("changed parameters, tokenizer or numerical settings cannot reuse a previously loaded training preview", () => {
  const config = selected()
  const checked = { key: trainingSelectionKey(config), errors: [] }
  for (const changed of [
    {
      ...config,
      architecture: { ...config.architecture!, parameters: { width: 5 } },
    },
    { ...config, tokenizer_id: revision },
    { ...config, batch_size: 1 },
    { ...config, seed: 42 },
    { ...config, precision: "fp32" as const },
    { ...config, allow_tf32: false },
  ])
    assert.ok(
      validate(changed, false, checked).some((error) =>
        error.includes("current selected recipe")
      )
    )
})

test("required, unknown, zero and false recipe parameters follow declarations without imposing Transformer constraints", () => {
  const recipe: Recipe = {
    schema_version: 1,
    interface: "tensor",
    context_window: null,
    parameters: {
      width: { type: "integer", required: true, minimum: 3 },
      gain: { type: "number", required: false, minimum: 0 },
      enabled: { type: "boolean", required: false },
    },
    inputs: {},
    outputs: {},
    result: {},
    modules: {},
    flow: [],
    aliases: [],
  }
  assert.deepEqual(
    trainingParameterErrors(
      {
        revision_id: revision,
        parameters: { width: 3, gain: 0, enabled: false },
      },
      recipe
    ),
    {}
  )
  assert.equal(
    trainingParameterErrors({ revision_id: revision, parameters: {} }, recipe)
      .width,
    "Required parameter."
  )
  assert.equal(
    trainingParameterErrors(
      { revision_id: revision, parameters: { width: 3, unknown: 1 } },
      recipe
    ).unknown,
    "Unknown recipe parameter."
  )
  assert.ok(
    trainingParameterErrors(
      { revision_id: revision, parameters: { width: 2 } },
      recipe
    ).width
  )
})
