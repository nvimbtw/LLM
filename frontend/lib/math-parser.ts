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
// tool call's JSON is a dollar sign, not the start of an equation.
const LITERAL = new Set(["code block", "code inline", "tool-call"])

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
  /\$\$([^]+?)\$\$|(?<![\w$])\$(?![\s$])((?:\\.|[^\\$\n])+?)(?<![\s\\])\$(?![\w$])/g

function mathNode(tex: string, display: boolean): MElement {
  return {
    type: display ? "math block" : "math inline",
    children: [{ type: "text", value: tex.trim() }],
  }
}

/** The node split into text and math, or null when it holds no math at all. */
function splitText(value: string): MNode[] | null {
  const out: MNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  MATH_RE.lastIndex = 0

  while ((m = MATH_RE.exec(value))) {
    const display = m[1] !== undefined
    let before = value.slice(last, m.index)
    // A displayed equation is a block of its own, so the newline that put it on
    // its own line is layout the renderer supplies — keeping it would open the
    // paragraph with a stray blank line.
    if (display) before = before.replace(/[ \t]*\n\s*$/, "")
    if (before) out.push({ type: "text", value: before })
    out.push(mathNode(display ? m[1] : m[2], display))
    last = MATH_RE.lastIndex
    if (display) {
      const skip = /^\s*\n[ \t]*/.exec(value.slice(last))
      if (skip) last += skip[0].length
    }
  }

  if (!out.length) return null
  if (last < value.length) out.push({ type: "text", value: value.slice(last) })
  return out
}

/** Rewrites the tree in place, returning it for chaining onto a parse. */
export function splitMath(root: MElement): MElement {
  if (LITERAL.has(root.type)) return root

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
