import { test } from "node:test"
import assert from "node:assert/strict"
import { readDraftStrings, readPageDraft, writePageDraft } from "./page-draft"

/** Supplies isolated tab storage and an opaque current bearer, then restores all browser globals after the test. */
function browser(run: (entries: Map<string, string>) => void) {
  const original = ["window", "localStorage", "sessionStorage"].map(
    (key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)] as const
  )
  const entries = new Map<string, string>()
  const sessionStorage = {
    getItem: (key: string) => entries.get(key) ?? null,
    setItem: (key: string, value: string) => entries.set(key, value),
    removeItem: (key: string) => entries.delete(key),
  }
  Object.defineProperty(globalThis, "window", { configurable: true, value: {} })
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: { getItem: () => "opaque-current-session" },
  })
  Object.defineProperty(globalThis, "sessionStorage", {
    configurable: true,
    value: sessionStorage,
  })
  try {
    run(entries)
  } finally {
    for (const [key, descriptor] of original) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor)
      else Reflect.deleteProperty(globalThis, key)
    }
  }
}

/** Restores an incomplete single text field for these persistence contracts. */
function parse(value: unknown) {
  return readDraftStrings(["text"], value)
}

test("resource drafts survive navigation/reload within their opaque session and never share fields with another session or resource", () =>
  browser((entries) => {
    const identity = {
      token: "opaque-current-session",
      resource: "project:one",
    }
    const saved = { text: "Server notes" }
    const fields = { text: "\nUnfinished notes\n" }
    writePageDraft(identity, saved, fields)
    assert.deepEqual(readPageDraft(identity, saved, parse), {
      fields,
      error: null,
      retained: true,
    })
    assert.equal(
      [...entries.keys()].some((key) => key.includes(identity.token)),
      false
    )
    assert.deepEqual(
      readPageDraft({ ...identity, resource: "project:two" }, saved, parse)
        .fields,
      saved
    )
    assert.deepEqual(
      readPageDraft({ ...identity, token: "other-session" }, saved, parse)
        .fields,
      saved
    )
    assert.throws(
      () =>
        writePageDraft({ ...identity, token: "other-session" }, saved, fields),
      /session changed/
    )
    writePageDraft(identity, saved, saved)
    assert.equal(entries.size, 0)
  }))

test("changed server baselines preserve the attempted draft with a review warning instead of silently discarding it", () =>
  browser(() => {
    const identity = {
      token: "opaque-current-session",
      resource: "source:architecture",
    }
    writePageDraft(identity, { text: "v1" }, { text: "Unfinished YAML: [" })
    const restored = readPageDraft(identity, { text: "v2" }, parse)
    assert.equal(restored.fields.text, "Unfinished YAML: [")
    assert.match(restored.error ?? "", /Saved values changed/)
    assert.equal(restored.retained, true)
  }))

test("malformed, oversized and unavailable drafts preserve stored bytes and return a visible recovery failure", () =>
  browser((entries) => {
    const identity = {
      token: "opaque-current-session",
      resource: "project:one",
    }
    const saved = { text: "Saved" }
    writePageDraft(identity, saved, { text: "Draft" })
    const key = [...entries.keys()][0]
    for (const raw of [
      "{",
      JSON.stringify({ version: 2, baseline: "{}", fields: {} }),
      JSON.stringify({ version: 1, baseline: "{}", fields: { text: 42 } }),
      " ".repeat(600001),
    ]) {
      entries.set(key, raw)
      const restored = readPageDraft(identity, saved, parse)
      assert.equal(restored.retained, false)
      assert.ok(restored.error)
      assert.deepEqual(restored.fields, saved)
      assert.equal(entries.get(key), raw)
    }
    assert.throws(
      () => writePageDraft(identity, saved, { text: "x".repeat(600001) }),
      /too large/
    )
  }))
