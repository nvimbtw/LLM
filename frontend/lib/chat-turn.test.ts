import { test } from "node:test"
import assert from "node:assert/strict"
import { groupTurns } from "./chat-messages"
import { createChatTurn, reduceChatTurn } from "./chat-turn"
import type { MessageWire } from "./chat-history"

/** Builds a persisted message fixture with the metadata expected from the history route. */
function row(
  role: MessageWire["role"],
  content: string,
  message_id: string
): MessageWire {
  return {
    role,
    content,
    message_id,
    model_name: role === "assistant" ? "Model" : null,
    feedback: null,
    attachments: [],
  }
}

test("decoded deltas concatenate verbatim and token reduction is replay-safe", () => {
  const initial = createChatTurn()
  const pieces = [
    "<|thinking|>",
    "I",
    "'",
    "ve",
    "<|/thinking|>",
    " b",
    "ust",
    "y",
    " boots",
  ]
  const turn = pieces.reduce(
    (state, value) => reduceChatTurn(state, { type: "token", value }),
    initial
  )
  assert.equal(turn.message.content, pieces.join(""))
  assert.equal(turn.totalTokens, pieces.length)
  assert.equal(turn.thinkingTokens, 5)
  assert.equal(initial.totalTokens, 0)
  assert.equal(initial.message.content, "")
  assert.deepEqual(
    reduceChatTurn(initial, { type: "token", value: " x" }),
    reduceChatTurn(initial, { type: "token", value: " x" })
  )
})

test("multi-round live working, answers and interjection positions agree with committed history", () => {
  const call = '<|tool-call|>{"tool":"bash","argument":"pwd"}<|/tool-call|>'
  const result = {
    tool: "bash",
    argument: "pwd",
    body: "<|code|>/tmp<|/code|>",
  }
  let turn = createChatTurn()
  for (const content of [
    `<|thinking|>First<|/thinking|>${call}`,
    `<|thinking|>Again<|/thinking|>${call}`,
  ]) {
    turn = reduceChatTurn(turn, { type: "token", value: content })
    turn = reduceChatTurn(turn, { type: "tool-call" })
    turn = reduceChatTurn(turn, { type: "tool-result", result })
    if (!turn.message.interjections)
      turn = reduceChatTurn(turn, { type: "interjected", text: "Be brief" })
  }
  turn = reduceChatTurn(turn, { type: "token", value: "<|par|>Done<|/par|>" })
  const history = groupTurns([
    row("user", "Question", "q"),
    row("assistant", `<|thinking|>First<|/thinking|>${call}`, "a1"),
    row("tool", call + result.body, "t1"),
    row("user", "Be brief", "i"),
    row("assistant", `<|thinking|>Again<|/thinking|>${call}`, "a2"),
    row("tool", call + result.body, "t2"),
    row("assistant", "<|par|>Done<|/par|>", "a3"),
  ]).at(-1)
  assert.equal(turn.message.working, history?.working)
  assert.equal(turn.message.content, history?.content)
  assert.deepEqual(turn.message.toolAnswers, history?.toolAnswers)
  assert.deepEqual(turn.message.interjections, history?.interjections)
  assert.equal(history?.id, "a3")
})
