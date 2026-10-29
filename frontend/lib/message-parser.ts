// Parses the model's custom markup into a small tree the UI can render.
//
// The model emits the backend tokenizer's protected vocabulary (SPECIAL_TOKENS
// in llm-engine/llm/corpus/special.py), which is terse and space-free:
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
// reserved id on the LLM engine side — the ids are positional — and the grammar in
// llm/chat/markup.py already forbids the model from emitting it.
//
// Those token names are mapped here onto the render vocabulary shared with the
// user-input parser (lib/user-parser.ts) and the renderer (rich-message.tsx) —
// e.g. `par` -> `paragraph`, `title` -> `heading 1`. See TOKEN_NAME_MAP. This
// keeps the renderer as the single place that knows how each node type looks,
// while the backend stays the single source of truth for token names.
//
// Maths is the exception to all of this: it has no token of its own and
// arrives as ordinary markdown (`$$…$$`, `$…$`) inside whatever block the model
// was already writing, so it is lifted out of the finished tree by
// `splitMath` (lib/math-parser.ts) rather than by the tag scanner below.
//
// Tags may carry attributes after a `|` separator (e.g. `href='...'`).
// `<|divider|>` is void (no closing tag).

import { MATH_LITERAL, splitText } from "./math-parser"

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

/**
 * Every tool is called through one marker, `<|tool-call|>`, whose body is a
 * JSON object naming the tool. Deliberately *not* in TOKEN_NAME_MAP: it is not
 * a rendered block, it is a call the trace reads.
 */
export const TOOL_CALL_TYPE = "tool-call"
export const TOOL_CALL_CLOSE = `<|/${TOOL_CALL_TYPE}|>`

export interface ToolCall {
  tool: string
  argument: string
}

/**
 * Keys an argument may arrive under. `llm-engine/llm/chat/tools.py::parse` is the
 * source of truth and is more forgiving than this — it is what decides whether
 * a call *runs*. This only decides how one is captioned, so it reads the same
 * keys and shrugs at anything stranger.
 */
const ARGUMENT_KEYS = ["argument", "query", "command", "input", "text"]

/** The call this node makes, or null if it is not one. */
export function callOf(node: MNode): ToolCall | null {
  if (isText(node) || node.type !== TOOL_CALL_TYPE) return null

  const body = textContent(node).trim()
  let payload: unknown
  try {
    payload = JSON.parse(body)
  } catch {
    // A call the model mangled. The LLM engine did not run it either, and showing
    // the broken text is more honest than hiding that the model tried.
    return { tool: "", argument: body }
  }
  if (typeof payload !== "object" || payload === null) {
    return { tool: "", argument: body }
  }

  const fields = payload as Record<string, unknown>
  const nested = fields.args ?? fields.arguments ?? fields.parameters
  const source =
    typeof nested === "object" && nested !== null
      ? (nested as Record<string, unknown>)
      : fields
  const key = ARGUMENT_KEYS.find((k) => typeof source[k] === "string")

  return {
    tool:
      typeof fields.tool === "string" ? fields.tool : String(fields.name ?? ""),
    argument: key ? String(source[key]) : "",
  }
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

// Thinking blocks: `<|thinking|>...reasoning...<|/thinking|>`. The content *is*
// shown, in the trace beside the answer (components/chat/trace.tsx); these
// helpers are for the callers that need the answer without it — the status row
// deciding whether the model is mid-thought, and the thread's live counts.
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

/** One open element and where it sits in its parent, enough to copy the path back to the root. */
interface Frame {
  node: MElement
  index: number
}

/**
 * Scanner state left behind by the previous parse of a prefix of this message.
 *
 * `root` deliberately omits the trailing text node: everything after `offset`
 * is still-arriving prose, and dropping it here is what makes the next parse
 * resume at the frontier instead of re-reading the message. The cost of the
 * omission is one shallow copy of the open path when the tree is handed out.
 */
interface ParseCheckpoint {
  source: string
  output: MElement
  root: MElement
  stack: Frame[]
  offset: number
}

/**
 * One message's checkpoint. Only one reply streams at a time, so one entry is
 * enough; a message that is not a prefix of it (a second chat, a user bubble
 * parsed in between) simply takes the full-parse path.
 */
let checkpoint: ParseCheckpoint | null = null

/**
 * Copies the path from `deepest` to the root, replacing the deepest open
 * element's children with `children`.
 *
 * Unchanged siblings keep their identity, so a memoised renderer can skip every
 * block a new token did not touch.
 */
function copyPath(
  stack: Frame[],
  children: MNode[]
): { root: MElement; stack: Frame[] } {
  const nodes = stack.map((frame) => frame.node)
  const deepest = nodes.length - 1
  nodes[deepest] = { ...nodes[deepest], children }
  for (let i = deepest; i > 0; i--) {
    const parent = nodes[i - 1]
    const replaced = parent.children.slice()
    replaced[stack[i].index] = nodes[i]
    nodes[i - 1] = { ...parent, children: replaced }
  }
  return {
    root: nodes[0],
    stack: nodes.map((node, index) => ({ node, index: stack[index].index })),
  }
}

/** Appends finished nodes to the deepest open element and returns the new tree and path. */
function append(
  stack: Frame[],
  nodes: MNode[]
): { root: MElement; stack: Frame[] } {
  const current = stack[stack.length - 1]
  return copyPath(stack, [...current.node.children, ...nodes])
}

/** True when text here is literal — code or a tool call — and must not be read for maths. */
function insideLiteral(stack: Frame[]): boolean {
  return stack.some((frame) => MATH_LITERAL.has(frame.node.type))
}

/** Appends a run of prose, lifting out any complete maths once the run has stopped growing. */
function appendProse(
  stack: Frame[],
  text: string
): { root: MElement; stack: Frame[] } {
  if (!text) return { root: stack[0].node, stack }
  const parts = insideLiteral(stack) ? null : splitText(text)
  return append(stack, parts ?? [{ type: "text", value: text }])
}

/**
 * Parses model markup into the render tree, reusing the previous result while
 * the same message keeps growing.
 *
 * The parser is called once per token on the accumulated reply, so a full
 * re-parse each time is quadratic in the reply's length. Resuming from the last
 * complete tag makes a token cost its own length, and identity-preserving path
 * copies make the unchanged part of the message free to re-render. A caller
 * that passes a string which is not an extension of the previous one gets an
 * ordinary full parse.
 */
export function parseMessage(input: string): MElement {
  if (checkpoint?.source === input) return checkpoint.output

  const resumed =
    checkpoint !== null && input.startsWith(checkpoint.source)
      ? checkpoint
      : null
  const stack: Frame[] = resumed
    ? resumed.stack
    : [{ node: { type: "root", children: [] }, index: 0 }]
  let tree = resumed ? resumed.root : stack[0].node
  let last = resumed?.offset ?? 0

  let m: RegExpExecArray | null
  TAG_RE.lastIndex = last
  while ((m = TAG_RE.exec(input))) {
    if (m.index > last) {
      const next = appendProse(stack, input.slice(last, m.index))
      tree = next.root
      stack.splice(0, stack.length, ...next.stack)
    }
    last = TAG_RE.lastIndex

    const inner = m[1]
    if (inner.startsWith("/")) {
      // Closing tag — unwind to the nearest matching open element. Map the
      // name so `<|/par|>` matches the `paragraph` node opened by `<|par|>`.
      const name = mapTokenName(inner.slice(1).trim())
      for (let i = stack.length - 1; i > 0; i--) {
        if (stack[i].node.type === name) {
          stack.length = i
          break
        }
      }
      continue
    }

    const { name, attrs } = parseOpenTag(inner)
    const node: MElement = { type: name, attrs, children: [] }
    const index = stack[stack.length - 1].node.children.length
    const next = append(stack, [node])
    tree = next.root
    stack.splice(0, stack.length, ...next.stack)
    if (!VOID_TAGS.has(name)) stack.push({ node, index })
  }

  // The prose after the last tag is the live edge: it is not stored in the
  // checkpoint, it is attached to the copy that is handed out.
  const tail = input.slice(last)
  const output = tail ? appendProse(stack, tail).root : tree

  checkpoint = { source: input, output, root: tree, stack, offset: last }
  return output
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
