import { test } from "node:test"
import assert from "node:assert/strict"
import { parseAccountSettings } from "./account-settings"
import { DEFAULT_SETTINGS } from "./settings"

test("older account blobs inherit current defaults while valid false/zero values stay explicit", () => {
  assert.deepEqual(parseAccountSettings({}), DEFAULT_SETTINGS)
  assert.deepEqual(
    parseAccountSettings({
      showReasoning: false,
      temperature: 0,
      systemPrompt: "Account prompt",
    }),
    {
      ...DEFAULT_SETTINGS,
      showReasoning: false,
      temperature: 0,
      systemPrompt: "Account prompt",
    }
  )
})

test("malformed preference values use defaults and malformed containers reject rather than authorizing replacement", () => {
  assert.deepEqual(
    parseAccountSettings({
      temperature: Infinity,
      topK: "50",
      showReasoning: "true",
      unknown: "ignored",
    }),
    DEFAULT_SETTINGS
  )
  for (const invalid of [undefined, null, [], "settings", 4])
    assert.throws(
      () => parseAccountSettings(invalid),
      /Unable to load settings/
    )
})
