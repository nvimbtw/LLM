import { afterEach, test } from "node:test"
import assert from "node:assert/strict"
import { readModel } from "./model-reads"

const originalFetch = globalThis.fetch

/** Restores the process fetch boundary after each isolated request test. */
afterEach(() => {
  globalThis.fetch = originalFetch
})

test("anonymous model reads do not borrow browser credentials and carry cancellation", async () => {
  const controller = new AbortController()
  let request: RequestInit | undefined
  globalThis.fetch = (async (_url, init) => {
    request = init
    return new Response('{"models":[]}', { status: 200 })
  }) as typeof fetch
  await readModel("/models", null, controller.signal)
  assert.equal(new Headers(request?.headers).has("Authorization"), false)
  assert.equal(request?.signal, controller.signal)
})

test("account model reads carry the captured identity through later session replacement", async () => {
  let authorization: string | null = null
  globalThis.fetch = (async (_url, init) => {
    authorization = new Headers(init?.headers).get("Authorization")
    return new Response("{}", { status: 200 })
  }) as typeof fetch
  await readModel("/models", "captured-session")
  assert.equal(authorization, "Bearer captured-session")
})
