import { test } from "node:test"
import assert from "node:assert/strict"
import { ChatSession, applyChatMessagesChange } from "./chat-session"
import type { ChatMessage } from "./chat-messages"
import type { streamChat, ChatStreamHandlers, Question } from "./chat-socket"

/** Exposes a promise's completion for deterministic history/sidebar race tests. */
function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((done, fail) => {
    resolve = done
    reject = fail
  })
  return { promise, resolve, reject }
}

/** Creates an isolated session whose socket controls and rendered messages can be observed without React. */
function harness(
  loadMessages: () => Promise<ChatMessage[]> = async () => [],
  refreshChats: () => Promise<void> = async () => {}
) {
  let messages: ChatMessage[] = []
  let system = "Initial prompt"
  const sockets: {
    handlers: ChatStreamHandlers
    question: Question
    model: string
    budget: string
    system: string | null
    floor: string | null
    cancelled: number
    interjections: string[]
  }[] = []
  /** Opens an observable fake socket, retaining callbacks so tests can deliver late frames after cancellation. */
  const openStream: typeof streamChat = (
    _chat,
    question,
    model,
    prompt,
    budget,
    _sampling,
    floor,
    handlers
  ) => {
    const socket = {
      handlers,
      question,
      model,
      system: prompt,
      budget,
      floor,
      cancelled: 0,
      interjections: [] as string[],
    }
    sockets.push(socket)
    return {
      cancel: () => {
        socket.cancelled++
      },
      interject: (text) => {
        socket.interjections.push(text)
      },
    }
  }
  const session = new ChatSession(
    "chat",
    {
      changeMessages: (change) => {
        messages = applyChatMessagesChange(messages, change)
      },
      loadMessages,
      refreshChats,
    },
    openStream
  )
  session.configure((modelId) => ({
    system,
    sampling: { temperature: 1, topK: 50, topP: 1, contextMultiplier: 1 },
    historyFloor: modelId === "model-a" ? "floor-a" : null,
  }))
  return {
    session,
    sockets,
    messages: () => messages,
    setSystem: (value: string) => {
      system = value
    },
  }
}

/** Allows final-round history and sidebar promise continuations to settle. */
async function settle(): Promise<void> {
  await new Promise<void>((resolve) => setImmediate(resolve))
}

test("queued duplicate steering confirms only one entry and drains FIFO with its own choices/current settings", async () => {
  const view = harness()
  const first = view.session.send("Start", "model-a", "low", [])
  await view.session.send("Same", "model-b", "high", [])
  await view.session.send("Same", "model-a", "med", [])
  await view.session.send("Last", "model-b", "auto", [])
  view.sockets[0].handlers.onInterjected("Same")
  assert.deepEqual(
    view.session.getSnapshot().queued.map((item) => item.budget),
    ["med", "auto"]
  )
  assert.deepEqual(view.sockets[0].interjections, ["Same", "Same", "Last"])
  view.setSystem("Changed before drain")
  view.sockets[0].handlers.onError("Offline")
  await first
  assert.equal(view.sockets[1].question.text, "Same")
  assert.equal(view.sockets[1].model, "model-a")
  assert.equal(view.sockets[1].budget, "med")
  assert.equal(view.sockets[1].system, "Changed before drain")
  assert.equal(view.sockets[1].floor, "floor-a")
  view.session.cancel()
  assert.equal(view.sockets[1].cancelled, 1)
  assert.equal(view.sockets[2].question.text, "Last")
  assert.equal(view.sockets[2].floor, null)
  view.sockets[0].handlers.onToken("stale")
  assert.equal(view.messages().at(-1)?.content, "")
  view.session.dispose()
})

test("final reconciliation waits for sidebar totals and retains authoritative content plus live tallies", async () => {
  const sidebar = deferred<void>()
  const view = harness(
    async () => [
      {
        role: "assistant",
        content: "Stored intact",
        id: "stored",
        modelName: "Model",
      },
    ],
    () => sidebar.promise
  )
  const sent = view.session.send("Ask", "model-a", "auto", [])
  view.sockets[0].handlers.onToken("delta")
  view.sockets[0].handlers.onDone({
    messageId: "round",
    final: false,
    elapsedSecs: 2,
  })
  await settle()
  assert.equal(view.messages().at(-1)?.content, "delta")
  view.sockets[0].handlers.onDone({
    messageId: "stored",
    final: true,
    elapsedSecs: 3,
  })
  await settle()
  assert.equal(view.session.getSnapshot().isSending, true)
  assert.deepEqual(view.messages()[0], {
    role: "assistant",
    content: "Stored intact",
    id: "stored",
    modelName: "Model",
    tokenCount: 1,
    thinkingTokenCount: 0,
    elapsedSecs: 5,
  })
  sidebar.resolve()
  await sent
  assert.equal(view.session.getSnapshot().isSending, false)
  view.session.dispose()
})

test("cancelling during a delayed final history read cannot overwrite or settle the next turn", async () => {
  const history = deferred<ChatMessage[]>()
  const view = harness(() => history.promise)
  const first = view.session.send("First", "model-a", "auto", [])
  await view.session.send("Next", "model-b", "high", [])
  view.sockets[0].handlers.onDone({
    messageId: "old",
    final: true,
    elapsedSecs: 1,
  })
  view.session.cancel()
  await first
  view.sockets[1].handlers.onToken("Current")
  history.resolve([
    { role: "assistant", content: "Obsolete committed history" },
  ])
  await settle()
  assert.equal(view.messages().at(-1)?.content, "Current")
  assert.equal(view.session.getSnapshot().isSending, true)
  assert.equal(view.sockets.length, 2)
  view.session.dispose()
})

test("disposing closes the socket, settles the send and ignores every later callback without draining queued text", async () => {
  const history = deferred<ChatMessage[]>()
  const view = harness(() => history.promise)
  const sent = view.session.send("First", "model-a", "auto", [])
  await view.session.send("Queued", "model-b", "low", [])
  view.sockets[0].handlers.onDone({
    messageId: "old",
    final: true,
    elapsedSecs: 1,
  })
  view.session.dispose()
  await sent
  const before = view.messages()
  view.sockets[0].handlers.onToken("stale")
  view.sockets[0].handlers.onToolCall({ tool: "bash", argument: "pwd" })
  view.sockets[0].handlers.onInterjected("Queued")
  view.sockets[0].handlers.onError("stale")
  history.resolve([{ role: "assistant", content: "stale history" }])
  await settle()
  assert.equal(view.messages(), before)
  assert.equal(view.sockets.length, 1)
  assert.equal(view.sockets[0].cancelled, 1)
})

test("a failed final history read retains deltas and still drains unsent steering", async () => {
  const view = harness(async () => {
    throw new Error("History offline")
  })
  const sent = view.session.send("First", "model-a", "auto", [])
  view.sockets[0].handlers.onToken("Kept approximation")
  await view.session.send("Next", "model-b", "high", [])
  view.sockets[0].handlers.onDone({
    messageId: "stored",
    final: true,
    elapsedSecs: 1,
  })
  await sent
  assert.equal(view.messages()[1].content, "Kept approximation")
  assert.equal(view.sockets[1].question.text, "Next")
  view.session.dispose()
})

test("effect reattachment starts idle and stale callbacks cannot target its new turn", async () => {
  const view = harness()
  const old = view.session.send("Old", "model-a", "auto", [])
  await view.session.send("Abandoned", "model-a", "low", [])
  view.session.dispose()
  await old
  view.session.activate()
  assert.equal(view.session.getSnapshot().isSending, false)
  assert.deepEqual(view.session.getSnapshot().queued, [])
  const current = view.session.send("Current", "model-b", "high", [])
  view.sockets[0].handlers.onError("Stale error")
  assert.equal(view.session.getSnapshot().isSending, true)
  assert.equal(view.session.getSnapshot().error, null)
  view.session.cancel()
  await current
  assert.equal(view.sockets.length, 2)
})

test("a failed sidebar refresh still settles the completed turn", async () => {
  const view = harness(
    async () => [],
    async () => {
      throw new Error("Sidebar offline")
    }
  )
  const sent = view.session.send("First", "model-a", "auto", [])
  view.sockets[0].handlers.onDone({
    messageId: "stored",
    final: true,
    elapsedSecs: 1,
  })
  await sent
  assert.equal(view.session.getSnapshot().isSending, false)
  view.session.dispose()
})
