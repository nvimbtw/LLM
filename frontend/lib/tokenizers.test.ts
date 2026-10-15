import assert from "node:assert/strict"
import { test } from "node:test"
import {
  parseTokenizerDetail,
  validateTokenizerFields,
  editTokenizer,
} from "./tokenizers"

const id = "11111111-1111-4111-8111-111111111111"
const detail = {
  tokenizer_id: id,
  name: "Prose",
  description: null,
  implicit: false,
  vocab_size: null,
  merges: 0,
  files: null,
  bytes: 0,
  datasets: [],
  has_vocab: false,
  models: [
    { model_id: "22222222-2222-4222-8222-222222222222", name: "Reader" },
  ],
}

test("tokenizer detail preserves unavailable measurements and validates model links", () => {
  const parsed = parseTokenizerDetail(detail)
  assert.equal(parsed.vocabSize, null)
  assert.equal(parsed.merges, 0)
  assert.equal(parsed.description, "")
  assert.equal(parsed.models[0].name, "Reader")
  assert.throws(() =>
    parseTokenizerDetail({
      ...detail,
      models: [{ model_id: "bad", name: "Reader" }],
    })
  )
  assert.throws(() => parseTokenizerDetail({ ...detail, bytes: -1 }))
})

test("tokenizer edits trim names, count Unicode characters and retain literal descriptions", () => {
  assert.deepEqual(
    validateTokenizerFields({ name: "  Prose  ", description: " line\n" }),
    { name: "Prose", description: " line\n" }
  )
  assert.equal(
    Array.from(
      validateTokenizerFields({ name: "🌙".repeat(120), description: "" }).name
    ).length,
    120
  )
  assert.throws(() => validateTokenizerFields({ name: "  ", description: "" }))
  assert.throws(() =>
    validateTokenizerFields({ name: "x".repeat(121), description: "" })
  )
  assert.throws(() =>
    validateTokenizerFields({ name: "Prose", description: "x".repeat(2001) })
  )
})

test("a stale session cannot send a tokenizer metadata write", async () => {
  const previous = globalThis.window
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: undefined,
  })
  try {
    await assert.rejects(
      editTokenizer("retired-session", id, { name: "Prose", description: "" }),
      /account changed/
    )
  } finally {
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: previous,
    })
  }
})
