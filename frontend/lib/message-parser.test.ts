import assert from "node:assert/strict"
import { test } from "node:test"

import {
  hasVisibleContent,
  isInThinkingBlock,
  parseMessage,
  textContent,
  type MElement,
} from "./message-parser"

/**
 * Parses without help from the streaming checkpoint, by first feeding the
 * module a string the target cannot extend.
 */
function fresh(input: string): MElement {
  parseMessage("\u0000uncached")
  return parseMessage(input)
}

const REPLY = [
  "<|title|>A short answer<|/title|>",
  "<|par|>Prose with <|bold|>weight<|/bold|> and <|code|>$not math$<|/code|>.</|par|>",
  "<|thinking|>hmm<|/thinking|>",
  "<|par|>Inline $x^2$ and a link <|link|>https://example.com?a=1<|/link|>.</|par|>",
  "<|list|><|item|>one<|/item|><|item|>two<|/item|><|/list|>",
  "<|codeblock|>let price = $5;\n</|codeblock|>",
  "<|divider|>",
  "<|quote|>quoted<|/quote|>",
  "<|par|>Display $$\\frac{1}{2}$$ follows.</|par|>",
].join("")

test("resuming from a streamed prefix matches a parse from scratch", () => {
  for (let end = 1; end <= REPLY.length; end += 3) {
    const prefix = REPLY.slice(0, end)
    assert.equal(
      JSON.stringify(parseMessage(prefix)),
      JSON.stringify(fresh(prefix)),
      `prefix of ${end} characters`
    )
  }
})

test("a token that completes no tag keeps every finished node's identity", () => {
  const opening = "<|par|>First block<|/par|><|par|>Second"
  const first = parseMessage(opening)
  const second = parseMessage(`${opening} block continues`)

  assert.equal(second.children.length, 2)
  assert.equal(second.children[0], first.children[0])
  assert.notEqual(second.children[1], first.children[1])
})

test("a partially streamed tag is text until it closes", () => {
  assert.equal(textContent(parseMessage("<|pa")), "<|pa")

  const completed = parseMessage("<|par|>hi<|/par|>")
  assert.equal(completed.children[0].type, "paragraph")
  assert.equal(textContent(completed), "hi")
})

test("maths is lifted once its run is final and never inside code", () => {
  const tree = parseMessage("<|par|>a $x$ b<|/par|>")
  const paragraph = tree.children[0] as MElement
  assert.deepEqual(
    paragraph.children.map((child) => child.type),
    ["text", "math inline", "text"]
  )

  const code = parseMessage("<|codeblock|>cost $5 and $6<|/codeblock|>")
  assert.deepEqual(
    code.children.map((child) => child.type),
    ["code block"]
  )
  assert.equal(textContent(code), "cost $5 and $6")
})

test("a reply that is replaced rather than extended is parsed from scratch", () => {
  parseMessage("<|par|>one<|/par|>")
  const replaced = parseMessage("<|par|>completely different<|/par|>")
  assert.equal(textContent(replaced), "completely different")
  assert.equal(
    JSON.stringify(replaced),
    JSON.stringify(fresh("<|par|>completely different<|/par|>"))
  )
})

test("the streaming predicates still agree with the parsed tree", () => {
  assert.equal(isInThinkingBlock("<|thinking|>still going"), true)
  assert.equal(isInThinkingBlock("<|thinking|>done<|/thinking|>"), false)
  assert.equal(hasVisibleContent("<|par|>hi<|/par|>"), true)
  assert.equal(hasVisibleContent("<|thinking|>thinking<|/thinking|>"), false)
})
