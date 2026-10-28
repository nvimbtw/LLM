import { test } from "node:test"
import assert from "node:assert/strict"
import { DEFAULT_CONFIG } from "@/hooks/use-training"
import type { TrainingStep } from "./training"
import {
  loadTrainingDraftAccount,
  parseTrainingDraft,
  readTrainingDraft,
  removeTrainingDraft,
  serializeTrainingDraft,
  TRAINING_DRAFT_CHANGED,
  trainingDraftKey,
  writeTrainingDraft,
} from "./training-draft"
import type { TrainingDraftIdentity } from "./training-draft"

const alice = "30000000-0000-4000-8000-000000000001"
const bob = "30000000-0000-4000-8000-000000000002"
const revision = "30000000-0000-4000-8000-000000000003"

/** Pairs a profile identity with an opaque bearer fixture, matching the browser's encrypted-token contract. */
function session(accountId: string, revision = 1): TrainingDraftIdentity {
  return {
    accountId,
    token: `opaque-encrypted-session-${accountId}-${revision}`,
  }
}

/** Installs isolated browser storage/events for one synchronous persistence test and restores the original globals afterward. */
function browser(run: (storage: Storage, events: EventTarget) => void): void {
  const oldStorage = Object.getOwnPropertyDescriptor(globalThis, "localStorage")
  const oldWindow = Object.getOwnPropertyDescriptor(globalThis, "window")
  const entries = new Map<string, string>()
  const storage: Storage = {
    get length() {
      return entries.size
    },
    clear: () => entries.clear(),
    getItem: (key) => entries.get(key) ?? null,
    key: (index) => [...entries.keys()][index] ?? null,
    removeItem: (key) => {
      entries.delete(key)
    },
    setItem: (key, value) => {
      entries.set(key, value)
    },
  }
  const events = new EventTarget()
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: storage,
  })
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: events,
  })
  try {
    run(storage, events)
  } finally {
    if (oldStorage)
      Object.defineProperty(globalThis, "localStorage", oldStorage)
    else Reflect.deleteProperty(globalThis, "localStorage")
    if (oldWindow) Object.defineProperty(globalThis, "window", oldWindow)
    else Reflect.deleteProperty(globalThis, "window")
  }
}

test("draft restoration preserves incomplete root fields, curriculum and ordered follow-ups without carrying validation receipts", () => {
  const config = {
    ...structuredClone(DEFAULT_CONFIG),
    model_name: "",
    seed: Number.NaN,
    lr: -1,
    curriculum: [{ name: "", epochs: 0, datasets: [] }],
    datasets: [{ id: "dataset", count: Number.NaN }],
    architecture_validation_key: "old-validation",
    architecture: {
      revision_id: revision,
      parameters: { width: 48 },
      validation_run_id: revision,
    },
  }
  const steps: TrainingStep[] = [
    {
      id: "fine-tune",
      kind: "train",
      train: {
        ...config,
        mode: "sft",
        base_model_id: "base",
        resume_from: "checkpoint",
      },
      suite: "language",
      errors: ["Old error"],
    },
    {
      id: "score",
      kind: "benchmark",
      train: DEFAULT_CONFIG,
      suite: "assistant",
      errors: [],
    },
  ]
  const restored = parseTrainingDraft(serializeTrainingDraft(config, steps))
  assert.equal(restored.config.model_name, "")
  assert.ok(Number.isNaN(restored.config.seed))
  assert.ok(Number.isNaN(restored.config.datasets[0].count))
  assert.equal(restored.config.lr, -1)
  assert.deepEqual(restored.config.curriculum, config.curriculum)
  assert.deepEqual(restored.config.architecture, {
    revision_id: revision,
    parameters: { width: 48 },
    validation_run_id: null,
  })
  assert.equal(restored.config.architecture_validation_key, undefined)
  assert.deepEqual(
    restored.steps.map((step) => step.id),
    ["fine-tune", "score"]
  )
  assert.equal(restored.steps[0].train.base_model_id, "base")
  assert.equal(restored.steps[0].train.resume_from, "checkpoint")
  assert.equal(restored.steps[0].train.architecture?.validation_run_id, null)
  assert.deepEqual(restored.steps[0].errors, [])
  assert.equal(config.architecture.validation_run_id, revision)
})

test("drafts survive token replacement for the same account and remain isolated across accounts", () =>
  browser((storage, events) => {
    const first = session(alice)
    storage.setItem("auth_token", first.token)
    let changed = 0
    events.addEventListener(TRAINING_DRAFT_CHANGED, () => {
      changed += 1
    })
    writeTrainingDraft(
      first,
      { ...DEFAULT_CONFIG, model_name: "Alice draft" },
      []
    )
    const refreshed = session(alice, 2)
    assert.equal(
      trainingDraftKey(first.accountId),
      trainingDraftKey(refreshed.accountId)
    )
    assert.equal(readTrainingDraft(refreshed), readTrainingDraft(first))
    assert.equal(readTrainingDraft(session(bob)), null)
    storage.setItem("auth_token", session(bob).token)
    assert.throws(
      () => writeTrainingDraft(first, DEFAULT_CONFIG, []),
      /session changed/
    )
    writeTrainingDraft(
      session(bob),
      { ...DEFAULT_CONFIG, model_name: "Bob draft" },
      []
    )
    assert.equal(
      parseTrainingDraft(readTrainingDraft(first) as string).config.model_name,
      "Alice draft"
    )
    assert.equal(changed, 2)
    assert.equal(readTrainingDraft(null), null)
    assert.throws(() => trainingDraftKey("invalid"))
  }))

test("acknowledging a submission clears only its exact draft while preserving newer cross-tab edits and replacement sessions", () =>
  browser((storage) => {
    const identity = session(alice)
    storage.setItem("auth_token", identity.token)
    writeTrainingDraft(identity, DEFAULT_CONFIG, [])
    const submitted = readTrainingDraft(identity)
    writeTrainingDraft(
      identity,
      { ...DEFAULT_CONFIG, model_name: "Newer draft" },
      []
    )
    assert.equal(removeTrainingDraft(identity, submitted), false)
    const newer = readTrainingDraft(identity)
    storage.setItem("auth_token", session(bob).token)
    assert.equal(removeTrainingDraft(identity, newer), false)
    storage.setItem("auth_token", identity.token)
    assert.equal(removeTrainingDraft(identity, newer), true)
    assert.equal(readTrainingDraft(identity), null)
    writeTrainingDraft(identity, DEFAULT_CONFIG, [])
    const restored = parseTrainingDraft(readTrainingDraft(identity) as string)
    assert.equal(
      removeTrainingDraft(
        identity,
        serializeTrainingDraft(restored.config, restored.steps)
      ),
      true
    )
  }))

test("invalid or oversized browser data rejects without being silently normalized", () => {
  const valid = JSON.parse(serializeTrainingDraft(DEFAULT_CONFIG, []))
  for (const value of [
    null,
    [],
    { ...valid, version: 2 },
    { ...valid, config: { ...valid.config, architecture: false } },
    { ...valid, config: { ...valid.config, epochs: "5" } },
    { ...valid, config: { ...valid.config, shuffle: "false" } },
    {
      ...valid,
      steps: [
        { id: "bad", kind: "benchmark", suite: "unknown", train: valid.config },
      ],
    },
  ])
    assert.throws(() => parseTrainingDraft(JSON.stringify(value)))
  assert.throws(() => parseTrainingDraft("{"))
  assert.throws(() => parseTrainingDraft(" ".repeat(262145)), /too large/)
  assert.throws(
    () =>
      serializeTrainingDraft(
        { ...DEFAULT_CONFIG, model_name: "x".repeat(262145) },
        []
      ),
    /too large/
  )
})

test("unavailable storage and quota errors preserve the last persisted draft and never report success", () =>
  browser((storage) => {
    const identity = session(alice)
    storage.setItem("auth_token", identity.token)
    writeTrainingDraft(identity, DEFAULT_CONFIG, [])
    const saved = readTrainingDraft(identity)
    storage.setItem = () => {
      throw new Error("Quota exceeded")
    }
    assert.throws(
      () =>
        writeTrainingDraft(
          identity,
          { ...DEFAULT_CONFIG, model_name: "Unsaved" },
          []
        ),
      /Quota/
    )
    assert.equal(readTrainingDraft(identity), saved)
    storage.getItem = () => {
      throw new Error("Storage blocked")
    }
    const error = readTrainingDraft(identity)
    assert.ok(error instanceof Error)
    assert.equal(readTrainingDraft(identity), error)
  }))

test("opaque sessions resolve draft ownership through the authenticated profile and reject unavailable or malformed identities", async () => {
  const original = globalThis.fetch
  const signal = new AbortController().signal
  const identity = session(alice)
  globalThis.fetch = (async (url, init) => {
    assert.equal(String(url), "http://localhost:8080/user/me")
    assert.equal(
      new Headers(init?.headers).get("Authorization"),
      `Bearer ${identity.token}`
    )
    assert.equal(init?.signal, signal)
    return Response.json({
      id_user: alice,
      username: "agents",
      access_level: "admin",
    })
  }) as typeof fetch
  try {
    assert.equal(await loadTrainingDraftAccount(identity.token, signal), alice)
    for (const response of [
      new Response("Unavailable", { status: 503 }),
      Response.json({ sub: alice }),
      Response.json({ id_user: "not-an-account" }),
    ]) {
      globalThis.fetch = (async () => response) as typeof fetch
      await assert.rejects(loadTrainingDraftAccount(identity.token))
    }
  } finally {
    globalThis.fetch = original
  }
})
