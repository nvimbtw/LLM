// Parses the model's custom markup into a small tree the UI can render.
//
// The model emits the backend tokenizer's protected vocabulary (SPECIAL_TOKENS
// in backend/src/tokenize/src/pretokenize.rs), which is terse and space-free:
//   <|title|>Title<|/title|>
//   <|par|>Text with <|bold|>bold<|/bold|> and <|code|>String<|/code|>.<|/par|>
//   <|codeblock|>...code...<|/codeblock|>
//   <|divider|>
//
// Two of them are rendered here but not yet emitted by any model — the parser
// and the renderer come first so there is something to generate SFT *against*:
//   <|quote|>Quoted text<|/quote|>
//   <|warn|>Something the reader should not miss<|/warn|>
//
// `<|reply|>` is still parsed, because stored messages may contain one, but the
// renderer drops it: the reply-reference feature is gone. The marker keeps its
// reserved id on the engine side — the ids are positional — and the grammar in
// llm/chat/markup.py already forbids the model from emitting it.
//
// Those token names are mapped here onto the render vocabulary shared with the
// user-input parser (lib/user-parser.ts) and the renderer (rich-message.tsx) —
// e.g. `par` -> `paragraph`, `title` -> `heading 1`. See TOKEN_NAME_MAP. This
// keeps the renderer as the single place that knows how each node type looks,
// while the backend stays the single source of truth for token names.
//
// Tags may carry attributes after a `|` separator (e.g. `href='...'`).
// `<|divider|>` is void (no closing tag).

// Backend token name -> render node type. Names not listed pass through
// unchanged (quote, divider, italic, bold, list, item, link, …).
const TOKEN_NAME_MAP: Record<string, string> = {
  par: "paragraph",
  title: "heading 1",
  heading: "heading 2",
  subheading: "heading 3",
  code: "code inline",
  codeblock: "code block",
}

function mapTokenName(name: string): string {
  return TOKEN_NAME_MAP[name] ?? name
}

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

// Thinking blocks: `<|thinking|>...reasoning...<|/thinking|>`. The content is
// never shown — the UI displays a themed status label while the model is in a
// thinking block — so we only need to strip these out of the rendered message.
const THINKING_RE = /<\|thinking\|>[^]*?<\|\/thinking\|>/g // a complete block
const THINKING_TAIL_RE = /<\|thinking\|>[^]*$/ // an unclosed, still-streaming one

/** Removes thinking blocks (complete and partially-streamed) from the text. */
export function stripThinking(content: string): string {
  return content.replace(THINKING_RE, "").replace(THINKING_TAIL_RE, "")
}

/**
 * True when the content currently ends inside an open `<|thinking|>` block —
 * i.e. the most recent `<|thinking|>` has no `<|/thinking|>` after it. Unlike
 * hasVisibleContent (which latches once the answer starts), this tracks the
 * live thinking state, so a model that reasons mid-reply is detected too.
 */
export function isInThinkingBlock(content: string): boolean {
  const open = content.lastIndexOf("<|thinking|>")
  if (open === -1) return false
  // `<|/thinking|>` contains a slash, so it never matches the opener search.
  return content.lastIndexOf("<|/thinking|>") < open
}

const TAG_RE = /<\|(.*?)\|>/g
const ATTR_RE = /(\w+)\s*=\s*'([^']*)'/g

function parseOpenTag(inner: string): {
  name: string
  attrs?: Record<string, string>
} {
  const bar = inner.indexOf("|")
  const name = mapTokenName((bar === -1 ? inner : inner.slice(0, bar)).trim())
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
      // Closing tag — unwind to the nearest matching open element. Map the
      // name so `<|/par|>` matches the `paragraph` node opened by `<|par|>`.
      const name = mapTokenName(inner.slice(1).trim())
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

/**
 * True once the message has visible answer text — i.e. something beyond the
 * (hidden) <|thinking|> block and control tokens. Used to switch from the
 * "Thinking…" indicator to the streamed answer.
 */
export function hasVisibleContent(content: string): boolean {
  return (
    stripThinking(content)
      .replace(/<\|.*?\|>/g, "") // complete control tokens
      .replace(/<\|[^]*$/, "") // an unclosed trailing token
      .trim().length > 0
  )
}
