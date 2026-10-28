import { test } from "node:test"
import assert from "node:assert/strict"
import { ArchitectureReadOwner } from "./read-owner"
import { parameterError } from "./parameters"
/** Provides a controllable response for read replacement races. */
function pending<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((complete) => {
    resolve = complete
  })
  return { promise, resolve }
}
test("replacement aborts transport and suppresses a late noncooperative response", async () => {
  const owner = new ArchitectureReadOwner({
    token: "session-a",
    resource: "block-a",
  })
  const first = pending<string>()
  let signal: AbortSignal | undefined
  const old = owner.read((requestSignal) => {
    signal = requestSignal
    return first.promise
  })
  const current = owner.read(async () => "new")
  assert.equal(signal?.aborted, true)
  first.resolve("old")
  assert.equal(await old, undefined)
  assert.equal(await current, "new")
})
test("session replacement disposes the old owner without affecting the replacement", async () => {
  const oldOwner = new ArchitectureReadOwner({
    token: "session-a",
    resource: "block-a",
  })
  const old = pending<string>()
  const response = oldOwner.read(() => old.promise)
  oldOwner.dispose()
  const replacement = new ArchitectureReadOwner({
    token: "session-b",
    resource: "block-a",
  })
  old.resolve("private session-a value")
  assert.equal(await response, undefined)
  assert.equal(
    await replacement.read(async () => "session-b value"),
    "session-b value"
  )
})
test("active errors remain actionable while disposed failures are ignored", async () => {
  const owner = new ArchitectureReadOwner({ token: "a", resource: "a" })
  await assert.rejects(
    owner.read(async () => {
      throw new Error("failed")
    }),
    /failed/
  )
  owner.dispose()
  assert.equal(await owner.read(async () => "unexpected"), undefined)
})
test("parameter controls preserve scalar types, finite bounds and exact integers", () => {
  assert.equal(
    parameterError({ type: "integer", minimum: 1, maximum: 8 }, 4),
    null
  )
  for (const value of [
    true,
    "4",
    1.5,
    0,
    9,
    Infinity,
    Number.MAX_SAFE_INTEGER + 1,
  ])
    assert.ok(
      parameterError({ type: "integer", minimum: 1, maximum: 8 }, value)
    )
  assert.equal(
    parameterError({ type: "enum", choices: [true, "true", 1] }, "true"),
    null
  )
  assert.ok(parameterError({ type: "enum", choices: [true, 1] }, "true"))
  assert.ok(parameterError({ type: "number" }, NaN))
})
