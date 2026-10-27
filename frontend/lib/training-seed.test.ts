import { test } from "node:test"
import assert from "node:assert/strict"
import { loadTrainingSeed } from "./training-seed"
import { DEFAULT_CONFIG } from "@/hooks/use-training"

const id = "30000000-0000-4000-8000-000000000001"
const revision = "30000000-0000-4000-8000-000000000002"

/** Supplies only a valid queue envelope around explicit reusable settings; no model execution is implied. */
function row(config: Record<string, unknown>, kind = "train") {
  return {
    run_id: id,
    kind,
    config,
    status: "done",
    label: "Saved settings",
    progress: 1,
    stage: null,
    output: { type: "model", id: null },
    ref_id: null,
    result: null,
    depends_on: null,
    error: null,
    created_at: null,
    started_at: null,
    finished_at: null,
  }
}

test("rerun settings preserve exact recipe pins and captured reads, remove old receipts and apply the requested checkpoint", async () => {
  const original = globalThis.fetch
  const signal = new AbortController().signal
  globalThis.fetch = (async (url, init) => {
    assert.equal(String(url), `http://localhost:8080/runs/${id}`)
    assert.equal(
      new Headers(init?.headers).get("Authorization"),
      "Bearer captured"
    )
    assert.equal(init?.signal, signal)
    return Response.json(
      row({
        ...DEFAULT_CONFIG,
        seed: 0,
        resume_from: "old-checkpoint",
        architecture_validation_key: "old-key",
        architecture: {
          revision_id: revision,
          parameters: { iterations: 7 },
          validation_run_id: revision,
        },
      })
    )
  }) as typeof fetch
  try {
    const config = await loadTrainingSeed(id, "selected-checkpoint", {
      token: "captured",
      signal,
    })
    assert.equal(config.resume_from, "selected-checkpoint")
    assert.equal(config.seed, 0)
    assert.deepEqual(config.architecture, {
      revision_id: revision,
      parameters: { iterations: 7 },
      validation_run_id: null,
    })
    assert.equal(config.architecture_validation_key, undefined)
  } finally {
    globalThis.fetch = original
  }
})

test("rerun settings retain preference modes and infer legacy fine-tuning without converting it to pretraining", async () => {
  const original = globalThis.fetch
  try {
    for (const mode of ["sft", "kto", "raft"] as const) {
      globalThis.fetch = (async () =>
        Response.json(
          row({ ...DEFAULT_CONFIG, mode, base_model: "saved-base" })
        )) as typeof fetch
      const config = await loadTrainingSeed(id, null, { token: "captured" })
      assert.equal(config.mode, mode)
      assert.equal(config.base_model_id, "saved-base")
    }
    globalThis.fetch = (async () =>
      Response.json(
        row({ model_name: "legacy", base_model: "saved-base" })
      )) as typeof fetch
    assert.equal(
      (await loadTrainingSeed(id, null, { token: "captured" })).mode,
      "sft"
    )
  } finally {
    globalThis.fetch = original
  }
})

test("missing, foreign, non-training and failed seed reads reject instead of returning default settings", async () => {
  const original = globalThis.fetch
  try {
    for (const response of [
      new Response("missing", { status: 404 }),
      new Response("failed", { status: 503 }),
      Response.json({ ...row({ ...DEFAULT_CONFIG }), run_id: revision }),
      Response.json(row({}, "benchmark")),
    ]) {
      globalThis.fetch = (async () => response) as typeof fetch
      await assert.rejects(loadTrainingSeed(id, null, { token: "captured" }))
    }
  } finally {
    globalThis.fetch = original
  }
})
