// Parses the model's custom markup into a small tree the UI can render.
//
// Markup looks like:
//   <|heading 1|>Title<|/heading 1|>
//   <|paragraph|>Text with <|bold|>bold<|/bold|> and
//     <|code inline | lang='rs'|>String<|/code inline|>.<|/paragraph|>
//   <|code block | lang='rs'|>...code...<|/code block|>
//   <|divider|>
//
// Tags may carry attributes after a `|` separator (e.g. `lang='rs'`,
// `href='...'`). `<|divider|>` is void (no closing tag).

export interface MText {
  type: "text"
  value: string
}

export interface MElement {
  type: string
  attrs?: Record<string, string>
  children: MNode[]
}

export type MNode = MText | MElement

export function isText(node: MNode): node is MText {
  return node.type === "text"
}

const VOID_TAGS = new Set(["divider"])

const TAG_RE = /<\|(.*?)\|>/g
const ATTR_RE = /(\w+)\s*=\s*'([^']*)'/g

function parseOpenTag(inner: string): { name: string; attrs?: Record<string, string> } {
  const bar = inner.indexOf("|")
  const name = (bar === -1 ? inner : inner.slice(0, bar)).trim()
  const attrStr = bar === -1 ? "" : inner.slice(bar + 1)

  let attrs: Record<string, string> | undefined
  let m: RegExpExecArray | null
  ATTR_RE.lastIndex = 0
  while ((m = ATTR_RE.exec(attrStr))) {
    attrs = attrs ?? {}
    attrs[m[1]] = m[2]
  }

  return { name, attrs }
}

export function parseMessage(input: string): MElement {
  const root: MElement = { type: "root", children: [] }
  const stack: MElement[] = [root]

  const pushText = (text: string) => {
    if (!text) return
    stack[stack.length - 1].children.push({ type: "text", value: text })
  }

  let last = 0
  let m: RegExpExecArray | null
  TAG_RE.lastIndex = 0
  while ((m = TAG_RE.exec(input))) {
    if (m.index > last) pushText(input.slice(last, m.index))
    last = TAG_RE.lastIndex

    const inner = m[1]
    if (inner.startsWith("/")) {
      // Closing tag — unwind to the nearest matching open element.
      const name = inner.slice(1).trim()
      for (let i = stack.length - 1; i > 0; i--) {
        if (stack[i].type === name) {
          stack.length = i
          break
        }
      }
      continue
    }

    const { name, attrs } = parseOpenTag(inner)
    const node: MElement = { type: name, attrs, children: [] }
    stack[stack.length - 1].children.push(node)
    if (!VOID_TAGS.has(name)) stack.push(node)
  }

  if (last < input.length) pushText(input.slice(last))

  return root
}

/** Flattens an element's descendant text — used for code, which is plain text. */
export function textContent(node: MNode): string {
  if (isText(node)) return node.value
  return node.children.map(textContent).join("")
}
