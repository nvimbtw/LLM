import { test } from "node:test"
import assert from "node:assert/strict"
import { DEFAULT_CONFIG } from "@/hooks/use-training"
import { trainingEntryConfig, trainingEntryMode } from "./training-entry"

test("Workbench hints initialize fresh training modes and reject unsupported query values", () => {
  assert.equal(trainingEntryMode("sft"), "sft")
  assert.equal(trainingEntryMode("pretrain"), "pretrain")
  assert.equal(trainingEntryMode("raft"), null)
  assert.equal(trainingEntryMode(null), null)
  const config = trainingEntryConfig({
    defaults: DEFAULT_CONFIG,
    draft: null,
    resume: null,
    mode: "sft",
  })
  assert.equal(config.mode, "sft")
  assert.notEqual(config.lr, DEFAULT_CONFIG.lr)
  assert.equal(DEFAULT_CONFIG.mode, "pretrain")
})

test("mode entry preserves the exact retained draft, including cycle variants and data", () => {
  for (const mode of ["pretrain", "sft", "raft", "kto"] as const) {
    const draft = {
      ...DEFAULT_CONFIG,
      mode,
      model_name: "retained",
      datasets: [{ id: "source", count: 12 }],
      lr: 0.003,
    }
    const config = trainingEntryConfig({
      defaults: DEFAULT_CONFIG,
      draft,
      resume: null,
      mode: "sft",
    })
    assert.equal(config, draft)
    assert.equal(config.mode, mode)
    assert.equal(config.datasets[0]?.id, "source")
    assert.equal(config.lr, 0.003)
  }
})

test("explicit resume wins over hints and saved configuration without mutating either", () => {
  const draft = {
    ...DEFAULT_CONFIG,
    mode: "sft" as const,
    model_name: "retained",
  }
  const config = trainingEntryConfig({
    defaults: DEFAULT_CONFIG,
    draft,
    resume: "checkpoint",
    mode: "sft",
  })
  assert.equal(config.resume_from, "checkpoint")
  assert.equal(config.mode, "pretrain")
  assert.equal(draft.model_name, "retained")
  assert.equal(draft.mode, "sft")
})
