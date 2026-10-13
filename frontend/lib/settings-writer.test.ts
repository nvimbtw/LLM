import { test } from "node:test"
import assert from "node:assert/strict"
import { createSettingsWriter } from "./settings-writer"
import { DEFAULT_SETTINGS, type Settings } from "./settings"

/** Waits beyond the test writer's one-millisecond debounce so pending callbacks can execute. */
async function debounceSettled(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 10))
}

test("a burst writes only the final complete settings value", async () => {
  const writes: Settings[] = []
  const writer = createSettingsWriter((value) => {
    writes.push(value)
  }, 1)
  writer.schedule({ ...DEFAULT_SETTINGS, temperature: 0.2 })
  writer.schedule({ ...DEFAULT_SETTINGS, temperature: 0.8 })
  await debounceSettled()
  assert.deepEqual(
    writes.map((value) => value.temperature),
    [0.8]
  )
  writer.flush()
  assert.equal(writes.length, 1)
})

test("flush writes once immediately and discard cancels a logged-out session's delayed write", async () => {
  const writes: Settings[] = []
  const writer = createSettingsWriter((value) => {
    writes.push(value)
  }, 1)
  writer.schedule(DEFAULT_SETTINGS)
  writer.flush()
  writer.flush()
  await debounceSettled()
  assert.equal(writes.length, 1)
  writer.schedule({ ...DEFAULT_SETTINGS, systemPrompt: "New prompt" })
  writer.discard()
  await debounceSettled()
  assert.equal(writes.length, 1)
})
