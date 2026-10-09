import { after, beforeEach, test } from "node:test"
import assert from "node:assert/strict"
import {
  cachedChatHistory,
  invalidateChatHistory,
  loadChatHistory,
  prefetchChatHistory,
  type MessageWire,
} from "./chat-history"
import { SESSION_CHANGED_EVENT } from "./api"

const originalFetch = globalThis.fetch
const originalNow = Date.now
const browser = new EventTarget()

function stubFetch(respond: () => Promise<Response>) {
  const stub = {
    calls: [] as { init?: RequestInit }[],
    respond,
  }
  globalThis.fetch = async (_input, init) => {
    stub.calls.push({ init })
    return stub.respond()
  }
  return stub
}
const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window")
const originalStorage = Object.getOwnPropertyDescriptor(
  globalThis,
  "localStorage"
)
let token: string | null = "session-a"
Object.defineProperty(globalThis, "window", {
  configurable: true,
  value: browser,
})
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: { getItem: () => token },
})

const messages: MessageWire[] = [
  {
    message_id: "message-a",
    role: "user",
    content: "Prefetched question",
    model_name: null,
    feedback: null,
    attachments: [],
  },
]

beforeEach(() => {
  globalThis.fetch = originalFetch
  Date.now = originalNow
  token = "session-a"
  browser.dispatchEvent(new Event(SESSION_CHANGED_EVENT))
})

after(() => {
  globalThis.fetch = originalFetch
  Date.now = originalNow
  if (originalWindow)
    Object.defineProperty(globalThis, "window", originalWindow)
  else Reflect.deleteProperty(globalThis, "window")
  if (originalStorage)
    Object.defineProperty(globalThis, "localStorage", originalStorage)
  else Reflect.deleteProperty(globalThis, "localStorage")
})

test("hover, focus and navigation share an in-flight request and completed history", async () => {
  let finish!: (response: Response) => void
  const fetch = stubFetch(
    () =>
      new Promise<Response>((resolve) => {
        finish = resolve
      })
  )
  prefetchChatHistory("chat-a")
  prefetchChatHistory("chat-a")
  const navigation = loadChatHistory("chat-a")
  assert.equal(fetch.calls.length, 1)
  assert.equal(cachedChatHistory("chat-a"), undefined)
  finish(Response.json({ messages }))
  assert.deepEqual(await navigation, messages)
  assert.deepEqual(cachedChatHistory("chat-a"), messages)
  assert.deepEqual(await loadChatHistory("chat-a"), messages)
  assert.equal(fetch.calls.length, 1)
  const headers = new Headers(fetch.calls[0].init?.headers)
  assert.equal(headers.get("Authorization"), "Bearer session-a")
})

test("a failed speculative load is retried on navigation", async () => {
  const fetch = stubFetch(async () => new Response("", { status: 503 }))
  prefetchChatHistory("chat-a")
  await assert.rejects(loadChatHistory("chat-a"), /Unable to load/)
  assert.equal(cachedChatHistory("chat-a"), undefined)
  fetch.respond = async () => Response.json({ messages })
  assert.deepEqual(await loadChatHistory("chat-a"), messages)
  assert.equal(fetch.calls.length, 2)
})

test("expired history is fetched again", async () => {
  Date.now = () => 1_000
  const fetch = stubFetch(async () => Response.json({ messages }))
  await loadChatHistory("chat-a")
  Date.now = () => 31_001
  assert.equal(cachedChatHistory("chat-a"), undefined)
  await loadChatHistory("chat-a")
  assert.equal(fetch.calls.length, 2)
})

test("the cache evicts old conversations instead of growing with the archive", async () => {
  stubFetch(async () => Response.json({ messages }))
  for (let index = 0; index < 11; index++)
    await loadChatHistory(`chat-${index}`)
  assert.equal(cachedChatHistory("chat-0"), undefined)
  assert.deepEqual(cachedChatHistory("chat-10"), messages)
})

test("session changes and cross-tab logout discard private history", async () => {
  const fetch = stubFetch(async () => Response.json({ messages }))
  await loadChatHistory("chat-a")
  token = "session-b"
  assert.equal(cachedChatHistory("chat-a"), undefined)
  await loadChatHistory("chat-a")
  browser.dispatchEvent(new Event("storage"))
  assert.equal(cachedChatHistory("chat-a"), undefined)
  await loadChatHistory("chat-a")
  browser.dispatchEvent(new Event(SESSION_CHANGED_EVENT))
  assert.equal(cachedChatHistory("chat-a"), undefined)
  token = null
  prefetchChatHistory("chat-a")
  assert.equal(fetch.calls.length, 3)
})

test("invalidation cannot let an older request replace fresh history", async () => {
  let finish!: (response: Response) => void
  const fetch = stubFetch(
    () =>
      new Promise<Response>((resolve) => {
        finish = resolve
      })
  )
  const old = loadChatHistory("chat-a")
  invalidateChatHistory("chat-a")
  const updated = [{ ...messages[0], content: "Updated question" }]
  fetch.respond = async () => Response.json({ messages: updated })
  await loadChatHistory("chat-a")
  finish(Response.json({ messages }))
  await old
  assert.deepEqual(cachedChatHistory("chat-a"), updated)
})
