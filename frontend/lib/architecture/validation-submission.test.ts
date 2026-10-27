import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { ValidationSubmission } from "./validation-submission"
import { decodeArchitecture } from "./decoder"
import { ArchitectureApiError } from "./http"
import type { ValidationRequest, ValidationRun } from "./contracts.generated"

const cases = JSON.parse(
  readFileSync(
    new URL(
      "../../../contracts/modular-architecture/v1/api/cases.json",
      import.meta.url
    ),
    "utf8"
  )
).cases

/** Supplies a synthetic saved request for transport lifecycle checks, without claiming runtime/numerical validation. */
function request(): Omit<ValidationRequest, "request_id"> {
  const { request_id: omitted, ...value } = decodeArchitecture(
    "ValidationRequest",
    cases.find(
      (item: { name: string }) => item.name === "ValidationRequest-positive"
    ).value
  )
  assert.ok(omitted)
  return value
}

/** Creates an interpreted synthetic queue acknowledgement retaining the exact submitted request. */
function acknowledged(config: ValidationRequest): ValidationRun {
  const run = decodeArchitecture(
    "ValidationRun",
    structuredClone(
      cases.find(
        (item: { name: string }) => item.name === "ValidationRun-positive"
      ).value
    )
  )
  return { ...run, config, status: "queued", result: null }
}

test("unknown submission retries retain deduplication identity only for unchanged bytes and configuration", async () => {
  const ids: string[] = []
  let succeed = false
  const owner = new ValidationSubmission(async (value) => {
    ids.push(value.request_id)
    if (!succeed) throw new Error("Response lost")
    return acknowledged(value)
  })
  try {
    const value = request()
    await owner.submit("generation-1", value)
    assert.equal(owner.getSnapshot().uncertain, true)
    await owner.submit("generation-1", value)
    assert.equal(ids[0], ids[1])
    await owner.submit("generation-1", { ...value, parameters: { changed: 3 } })
    assert.notEqual(ids[2], ids[1])
    succeed = true
    await owner.submit("generation-1", { ...value, parameters: { changed: 3 } })
    assert.equal(ids[3], ids[2])
    assert.equal(owner.getSnapshot().run?.status, "queued")
    assert.equal(owner.getSnapshot().uncertain, false)
  } finally {
    owner.dispose()
  }
})

test("one pending submit owns transport; replacement aborts reads and cannot publish a late queued identity", async () => {
  let complete!: (value: ValidationRun) => void
  let config!: ValidationRequest
  let signal: AbortSignal | undefined
  let calls = 0
  const owner = new ValidationSubmission((value, currentSignal) => {
    calls++
    config = value
    signal = currentSignal
    return new Promise((resolve) => {
      complete = resolve
    })
  })
  const pending = owner.submit("first", request())
  await owner.submit("second", request())
  const before = owner.getSnapshot()
  owner.dispose()
  complete(acknowledged(config))
  await pending
  assert.equal(calls, 1)
  assert.equal(signal?.aborted, true)
  assert.equal(owner.getSnapshot(), before)
})

test("known refusal permits a fresh request while malformed outgoing input never reaches the queue", async () => {
  let calls = 0
  const ids: string[] = []
  const owner = new ValidationSubmission(async (value) => {
    calls++
    ids.push(value.request_id)
    throw new ArchitectureApiError(422, null)
  })
  try {
    await owner.submit("first", request())
    assert.equal(owner.getSnapshot().uncertain, false)
    await owner.submit("first", request())
    assert.notEqual(ids[0], ids[1])
    await assert.rejects(
      owner.submit("invalid", { ...request(), profile: null })
    )
    assert.equal(calls, 2)
  } finally {
    owner.dispose()
  }
})

test("foreign queue configuration cannot become this editor's validation result", async () => {
  const owner = new ValidationSubmission(async (value) =>
    acknowledged({ ...value, parameters: { foreign: true } })
  )
  try {
    await owner.submit("first", request())
    assert.equal(owner.getSnapshot().run, null)
    assert.equal(owner.getSnapshot().uncertain, true)
  } finally {
    owner.dispose()
  }
})
