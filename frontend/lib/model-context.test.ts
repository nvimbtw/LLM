import { test } from "node:test"
import assert from "node:assert/strict"
import { effectiveContextMultiplier } from "./model-context"
import type { Model } from "@/hooks/use-models"
import { formatParams } from "@/components/models/format"
import { SavedModelDefinition } from "@/components/models/saved-definition"
import { createElement } from "react"
import { renderToStaticMarkup } from "react-dom/server"

const model: Model = {
  id: "selected",
  name: "Selected",
  family: "Custom",
  tier: null,
  params: 712,
  contextWindow: 8,
  description: "",
  published: true,
}

test("only explicit implemented context support affects requests/meters; declarations, missing metadata and custom models keep the trained window", () => {
  assert.equal(
    effectiveContextMultiplier({ ...model, supportsContextExtension: true }, 4),
    4
  )
  assert.equal(
    effectiveContextMultiplier(
      {
        ...model,
        definitionId: "a".repeat(64),
        supportsContextExtension: false,
      },
      4
    ),
    1
  )
  assert.equal(effectiveContextMultiplier(model, 4), 1)
  assert.equal(effectiveContextMultiplier(null, 4), 1)
  assert.equal(
    effectiveContextMultiplier(
      { ...model, supportsContextExtension: true },
      Number.NaN
    ),
    1
  )
})

test("measured zero, unavailable counts and legacy estimates have distinct model labels", () => {
  assert.equal(formatParams(0), "0")
  assert.equal(formatParams(null), "Unavailable")
  assert.equal(formatParams(712, true), "≈ 712")
})

test("server rendering never mounts private saved-definition/graph reads before administrator hydration", () => {
  const original = globalThis.fetch
  let calls = 0
  globalThis.fetch = (async () => {
    calls++
    throw new Error("Unexpected private read")
  }) as typeof fetch
  try {
    assert.equal(
      renderToStaticMarkup(
        createElement(SavedModelDefinition, {
          modelId: "selected",
          definitionId: "a".repeat(64),
        })
      ),
      ""
    )
    assert.equal(calls, 0)
  } finally {
    globalThis.fetch = original
  }
})
