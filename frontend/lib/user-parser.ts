// Parses what a user naturally types (a small markdown subset) into the same
// node tree the model renderer uses, so user content gets identical styling.
//
// Supported:
//   ```lang            fenced code block (lang optional)
//   code
//   ```
//   `inline code`
//   - item / * item     unordered list
//   1. item             ordered list
//   **bold**  *italic*
//   blank-line-separated paragraphs

import { splitMath } from "./math-parser"
import type { MElement, MNode } from "./message-parser"

const FENCE_OPEN = /^```(\w*)\s*$/
const FENCE_CLOSE = /^```\s*$/
// Captures leading indent, the marker, and the item text.
const LIST_ITEM = /^([ \t]*)([-*]|\d+\.)\s+(.*)$/

interface RawItem {
  indent: number
  ordered: boolean
  text: string
}

// Builds (possibly nested) list elements from raw items, using indentation.
function listsFromItems(items: RawItem[]): MNode[] {
  const roots: MNode[] = []
  const stack: { indent: number; list: MElement; item: MElement }[] = []

  for (const it of items) {
    const itemNode: MElement = { type: "item", children: parseInline(it.text) }

    while (stack.length && it.indent < stack[stack.length - 1].indent) {
      stack.pop()
    }

    const top = stack[stack.length - 1]
    if (!top || it.indent > top.indent) {
      const list: MElement = {
        type: it.ordered ? "ordered list" : "list",
        children: [itemNode],
      }
      if (top) top.item.children.push(list)
      else roots.push(list)
      stack.push({ indent: it.indent, list, item: itemNode })
    } else {
      top.list.children.push(itemNode)
      top.item = itemNode
    }
  }

  return roots
}

// `code` | **bold** | *italic*  (code first so its contents are left alone)
const INLINE_RE = /`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*/g

function parseInline(text: string): MNode[] {
  const nodes: MNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  INLINE_RE.lastIndex = 0

  while ((m = INLINE_RE.exec(text))) {
    if (m.index > last) {
      nodes.push({ type: "text", value: text.slice(last, m.index) })
    }
    last = INLINE_RE.lastIndex

    if (m[1] !== undefined) {
      nodes.push({
        type: "code inline",
        children: [{ type: "text", value: m[1] }],
      })
    } else if (m[2] !== undefined) {
      nodes.push({ type: "bold", children: [{ type: "text", value: m[2] }] })
    } else if (m[3] !== undefined) {
      nodes.push({ type: "italic", children: [{ type: "text", value: m[3] }] })
    }
  }

  if (last < text.length) {
    nodes.push({ type: "text", value: text.slice(last) })
  }

  return nodes
}

export function parseUserMarkdown(input: string): MElement {
  const lines = input.replace(/\r\n/g, "\n").split("\n")
  const root: MElement = { type: "root", children: [] }

  let para: string[] = []
  const flushPara = () => {
    const text = para.join("\n").trim()
    para = []
    if (text) {
      root.children.push({ type: "paragraph", children: parseInline(text) })
    }
  }

  let i = 0
  while (i < lines.length) {
    const line = lines[i]

    // Fenced code block.
    const fence = line.match(FENCE_OPEN)
    if (fence) {
      flushPara()
      const lang = fence[1] || undefined
      i++
      const code: string[] = []
      while (i < lines.length && !FENCE_CLOSE.test(lines[i])) {
        code.push(lines[i])
        i++
      }
      i++ // skip closing fence
      root.children.push({
        type: "code block",
        attrs: lang ? { lang } : undefined,
        children: [{ type: "text", value: code.join("\n") }],
      })
      continue
    }

    // Lists — gather consecutive item lines (any indent/kind), then nest them.
    if (LIST_ITEM.test(line)) {
      flushPara()
      const items: RawItem[] = []
      while (i < lines.length) {
        const m = lines[i].match(LIST_ITEM)
        if (!m) break
        items.push({
          indent: m[1].replace(/\t/g, "  ").length,
          ordered: /\d/.test(m[2]),
          text: m[3],
        })
        i++
      }
      for (const node of listsFromItems(items)) root.children.push(node)
      continue
    }

    // Blank line ends a paragraph.
    if (line.trim() === "") {
      flushPara()
      i++
      continue
    }

    para.push(line)
    i++
  }

  flushPara()
  return splitMath(root)
}
