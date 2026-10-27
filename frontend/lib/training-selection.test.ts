import { test } from "node:test"
import assert from "node:assert/strict"
import { DEFAULT_CONFIG } from "@/hooks/use-training"
import { trainingForEngine, queueTraining } from "./jobs"
import { LEGACY_STRUCTURE_FIELDS, type TrainConfig } from "./training"
import {
  trainingPreviewRequest,
  trainingSelectionKey,
} from "./training-selection"
import {
  createPreset,
  loadModelConfig,
  toTrainConfig,
} from "./training-presets"

const revision = "30000000-0000-4000-8000-000000000002"
const validation = "30000000-0000-4000-8000-000000000003"
const tokenizer = "30000000-0000-4000-8000-000000000001"

/** Supplies an explicitly synthetic selected configuration for transport/preset behavior, without claiming numerical validity. */
function selected(): TrainConfig {
  return {
    ...DEFAULT_CONFIG,
    architecture: {
      revision_id: revision,
      parameters: { gain: 0, enabled: false },
      validation_run_id: validation,
    },
    tokenizer_id: tokenizer,
    precision: "fp32",
    seed: 0,
    allow_tf32: false,
  }
}

test("a selected recipe is the sole structural authority on the wire; optimization/data settings remain separate", () => {
  const config = selected()
  const actual = trainingForEngine({
    ...config,
    rope_base: 10000,
    learned_temperature: false,
  } as TrainConfig)
  for (const field of LEGACY_STRUCTURE_FIELDS)
    assert.equal(field in actual, false)
  assert.equal("architecture_validation_key" in actual, false)
  assert.equal(actual.batch_size, config.batch_size)
  assert.equal(actual.lr, config.lr)
  assert.deepEqual(actual.architecture, {
    ...config.architecture,
    validation_run_id: null,
  })
})

test("inherited SFT and resume serialization omit all structural and tokenizer overrides", () => {
  const config = selected()
  for (const next of [
    { ...config, mode: "sft" as const, base_model_id: "saved-base" },
    { ...config, resume_from: "saved-checkpoint" },
  ]) {
    const actual = trainingForEngine(next)
    for (const field of [
      ...LEGACY_STRUCTURE_FIELDS,
      "architecture",
      "tokenizer_id",
      "max_vocab",
      "architecture_validation_key",
    ])
      assert.equal(field in actual, false)
    assert.equal(actual.batch_size, config.batch_size)
  }
  assert.equal(
    trainingForEngine({ ...config, mode: "sft", base_model_id: "saved-base" })
      .base_model,
    "saved-base"
  )
})

test("only unchanged selection and numerical inputs retain explicitly attached receipt identity", () => {
  const config = selected()
  config.architecture_validation_key = trainingSelectionKey(config)
  assert.equal(
    (trainingForEngine(config).architecture as { validation_run_id: string })
      .validation_run_id,
    validation
  )
  for (const next of [
    { ...config, batch_size: config.batch_size + 1 },
    { ...config, tokenizer_id: revision },
    { ...config, precision: "bf16" as const },
    { ...config, seed: 1 },
    { ...config, allow_tf32: true },
    {
      ...config,
      architecture: { ...config.architecture!, parameters: { gain: 1 } },
    },
  ]) {
    const actual = trainingForEngine(next).architecture as {
      validation_run_id: null
      revision_id: string
    }
    assert.equal(actual.validation_run_id, null)
    assert.equal(actual.revision_id, revision)
  }
  assert.equal(
    (
      trainingForEngine({ ...config, lr: 0.02, epochs: 7 }).architecture as {
        validation_run_id: string
      }
    ).validation_run_id,
    validation
  )
})

test("prospective requests carry explicit numerical controls without shape defaults or authorization claims", () => {
  const config = selected()
  const value = trainingPreviewRequest(config)
  assert.equal(value.selection.revision_id, revision)
  assert.equal(value.selection.validation_run_id, null)
  assert.equal(value.batch_size, config.batch_size)
  assert.equal(value.tokenizer_id, tokenizer)
  assert.equal("context_window" in value, false)
  assert.equal("dimensions" in value, false)
})

test("legacy and version-two presets retain pins/settings while requiring fresh validation", () => {
  const legacy = toTrainConfig(
    {
      dimensions: 128,
      tokenizer_id: tokenizer,
      datasets: [{ id: "dataset", count: 5 }],
    },
    "Legacy"
  )
  assert.equal(legacy.dimensions, 128)
  assert.equal(legacy.tokenizer_id, tokenizer)
  const config = selected()
  config.architecture_validation_key = trainingSelectionKey(config)
  const loaded = toTrainConfig({ schema_version: 2, config }, "Fallback")
  assert.equal(loaded.architecture?.revision_id, revision)
  assert.deepEqual(
    loaded.architecture?.parameters,
    config.architecture?.parameters
  )
  assert.equal(loaded.architecture?.validation_run_id, null)
  assert.equal(loaded.architecture_validation_key, undefined)
  assert.equal(loaded.tokenizer_id, tokenizer)
  assert.throws(
    () => toTrainConfig({ schema_version: 99, config }, "Fallback"),
    /unsupported/
  )
  assert.throws(
    () =>
      toTrainConfig(
        { architecture: { revision_id: "not-a-revision", parameters: {} } },
        "Fallback"
      ),
    /Invalid ArchitectureSelection/
  )
})

test("preset creation stores the versioned pin envelope using captured identity and no carried receipt", async () => {
  const original = globalThis.fetch
  const controller = new AbortController()
  let saved: Record<string, unknown> | undefined
  globalThis.fetch = (async (_url, init) => {
    assert.equal(
      new Headers(init?.headers).get("Authorization"),
      "Bearer captured-session"
    )
    assert.equal(init?.signal, controller.signal)
    const body = JSON.parse(String(init?.body))
    saved = body.config
    return Response.json({
      id_preset: "preset",
      name: body.name,
      config: body.config,
      updated_at: "2026-10-24T20:00:00Z",
    })
  }) as typeof fetch
  try {
    const preset = await createPreset("Pinned", selected(), {
      token: "captured-session",
      signal: controller.signal,
    })
    assert.equal(saved?.schema_version, 2)
    assert.equal(
      toTrainConfig(preset.config, preset.name).architecture?.revision_id,
      revision
    )
    assert.equal(
      toTrainConfig(preset.config, preset.name).architecture?.validation_run_id,
      null
    )
  } finally {
    globalThis.fetch = original
  }
})

test("model settings without reusable modular selection cannot silently become the default standard architecture", async () => {
  const original = globalThis.fetch
  globalThis.fetch = (async () =>
    Response.json({
      definition_id: "a".repeat(64),
      parameters: { config: { model_name: "Imported" } },
    })) as typeof fetch
  try {
    await assert.rejects(
      loadModelConfig("saved", "Imported"),
      /no reusable training selection/
    )
  } finally {
    globalThis.fetch = original
  }
})

test("root and chained training requests share pin/inheritance serialization", async () => {
  const original = globalThis.fetch
  let body: Array<{ kind: string; config: Record<string, unknown> }> = []
  globalThis.fetch = (async (url, init) => {
    const requested = new URL(String(url))
    assert.equal(requested.pathname, "/runs/chain")
    assert.match(
      requested.searchParams.get("request_id") ?? "",
      /^[0-9a-f-]{36}$/
    )
    body = JSON.parse(String(init?.body))
    return Response.json(
      body.map((step, index) => ({
        run_id: validation,
        kind: "train",
        status: "queued",
        label: "Queued selection",
        config: step.config,
        progress: 0,
        stage: null,
        ref_id: null,
        output: { type: "model", id: null },
        result: null,
        depends_on: index === 0 ? null : validation,
        error: null,
        created_at: null,
        started_at: null,
        finished_at: null,
      }))
    )
  }) as typeof fetch
  try {
    await queueTraining(selected(), [
      {
        kind: "train",
        config: { ...selected(), mode: "sft", base_model_id: undefined },
      },
    ])
    assert.equal(
      (body[0].config.architecture as { revision_id: string }).revision_id,
      revision
    )
    for (const field of LEGACY_STRUCTURE_FIELDS)
      assert.equal(field in body[0].config, false)
    for (const field of [
      ...LEGACY_STRUCTURE_FIELDS,
      "architecture",
      "tokenizer_id",
      "max_vocab",
    ])
      assert.equal(field in body[1].config, false)
  } finally {
    globalThis.fetch = original
  }
})
