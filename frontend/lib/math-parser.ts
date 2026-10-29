// Lifts TeX out of the text nodes of an already-parsed tree, so both parsers
// get math for free and the renderer keeps its one job.
//
// Math is *not* a special token. The model writes it the way markdown does —
// `$$…$$` for a displayed equation, `$…$` inline — inside whatever block it was
// already writing, so it arrives here as ordinary text in a `<|par|>` or a list
// item. This pass runs last, over the finished tree, rather than inside either
// parser: `parseMessage` (model markup) and `parseUserMarkdown` (what a person
// types) build the same node vocabulary from different syntax, and the dollar
// rules are identical on both sides.
//
// Produces `math block` and `math inline` nodes whose single text child is the
// TeX source, unrendered — see components/chat/math.tsx for the KaTeX pass.

import { isText, type MElement, type MNode } from "./message-parser"

// Contents that are already literal: a dollar sign in a shell snippet or in a
// tool call's JSON is a dollar sign, not the start of an equation. Exported
// because the streaming parser decides this per text run rather than once per
// finished tree.
export const MATH_LITERAL = new Set(["code block", "code inline", "tool-call"])

/**
 * Display `$$…$$` first, so a displayed equation is never read as two inline
 * ones with an empty body.
 *
 * The inline rule is the one that has to earn its keep, because prose is full
 * of prices. An opener may not be preceded by a word character (so `it$x$` is
 * not math) nor followed by whitespace, and a closer may not be preceded by
 * whitespace nor followed by one — which is what stops "$5 and $10" from
 * closing across the "and": the body cannot contain a bare `$`, so the only
 * candidate closer is followed by a digit and the match is abandoned. A body
 * may not span a blank line either; an unclosed `$` then stays text rather than
 * swallowing the rest of the reply while it streams.
 */
const MATH_RE =
  /\$\$([^]+?)\$\$|(?<![\w$])\$(?![\s$])((?:\\.|[^\\$\n])+?)(?<![\s\\])\$(?![\w$])|(?<!\\)\\\[([^]+?)\\\]|(?<!\\)\\\(([^]+?)\\\)|(?<!\\)\\boxed\s*\{/g

/** Wraps delimiter-free TeX in the node vocabulary consumed by the math renderer. */
function mathNode(tex: string, display: boolean): MElement {
  return {
    type: display ? "math block" : "math inline",
    children: [{ type: "text", value: tex.trim() }],
  }
}

/** Finds the end of a boxed argument, counting nested braces and skipping TeX escapes. */
function boxedEnd(value: string, start: number): number | null {
  let depth = 1
  for (let index = start; index < value.length; index++) {
    const character = value[index]
    if (character === "\\") {
      index++
    } else if (character === "{") {
      depth++
    } else if (character === "}" && --depth === 0) {
      return index + 1
    }
  }
  return null
}

/**
 * Splits complete delimited math and balanced bare boxed answers; unfinished streams stay text.
 * Display math consumes adjacent layout newlines because its renderer supplies that spacing.
 *
 * Exported so the streaming parser can lift math out of one text node as soon
 * as that node stops growing, instead of re-walking the whole message; returns
 * null when the value holds no complete equation.
 */
export function splitText(value: string): MNode[] | null {
  const out: MNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  MATH_RE.lastIndex = 0

  while ((m = MATH_RE.exec(value))) {
    const delimited = m[1] ?? m[2] ?? m[3] ?? m[4]
    const end =
      delimited === undefined
        ? boxedEnd(value, MATH_RE.lastIndex)
        : MATH_RE.lastIndex
    if (end === null) continue
    const display = m[1] !== undefined || m[3] !== undefined
    const tex = delimited ?? value.slice(m.index, end)
    let before = value.slice(last, m.index)
    if (display) before = before.replace(/[ \t]*\n\s*$/, "")
    if (before) out.push({ type: "text", value: before })
    out.push(mathNode(tex, display))
    last = end
    if (display) {
      const skip = /^\s*\n[ \t]*/.exec(value.slice(last))
      if (skip) last += skip[0].length
    }
    MATH_RE.lastIndex = last
  }

  if (!out.length) return null
  if (last < value.length) out.push({ type: "text", value: value.slice(last) })
  return out
}

/** Rewrites the tree in place, returning it for chaining onto a parse. */
export function splitMath(root: MElement): MElement {
  if (MATH_LITERAL.has(root.type)) return root

  const children: MNode[] = []
  for (const child of root.children) {
    if (isText(child)) {
      const parts = splitText(child.value)
      if (parts) children.push(...parts)
      else children.push(child)
    } else {
      children.push(splitMath(child))
    }
  }
  root.children = children
  return root
}
