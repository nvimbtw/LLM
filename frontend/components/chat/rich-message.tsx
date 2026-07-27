"use client"

import * as React from "react"
import { Check, Copy } from "lucide-react"

import { cn } from "@/lib/utils"
import {
  isInThinkingBlock,
  isText,
  parseMessage,
  textContent,
  type MElement,
  type MNode,
} from "@/lib/message-parser"
import { useSettings } from "@/components/settings/settings-provider"
import { MessageAside } from "./message-aside"
import { MessageReplyRef } from "./message-reply-ref"
import { ReasoningPanel } from "./reasoning-panel"

export function RichMessage({
  content,
  streaming = false,
  className,
}: {
  content: string
  streaming?: boolean
  className?: string
}) {
  const tree = React.useMemo(() => parseMessage(content), [content])
  // Whether the *reasoning* is what is currently arriving, as opposed to the
  // answer. A message can hold several thinking blocks and only the open one is
  // live, so this is combined with "is the last node" below rather than applied
  // to every block in the message.
  const liveThinking = streaming && isInThinkingBlock(content)
  return (
    <RichContent
      root={tree}
      streaming={streaming}
      liveThinking={liveThinking}
      className={className}
    />
  )
}

/**
 * Renders an already-parsed node tree (model markup or user markdown).
 *
 * The type size is *not* fixed here. This is the app's one long-form reading
 * surface and it wants a larger measure than a table row does — but the same
 * renderer also has to sit inside a preview card, so the size comes from the
 * caller and everything below sizes in `em` off it.
 */
export function RichContent({
  root,
  streaming = false,
  liveThinking = false,
  className,
}: {
  root: MElement
  streaming?: boolean
  liveThinking?: boolean
  className?: string
}) {
  const { settings } = useSettings()
  const ctx: Ctx = {
    streaming,
    liveThinking,
    showReasoning: settings.showReasoning,
  }
  return (
    <div className={cn("space-y-3.5 break-words", className)}>
      {renderChildren(root, ctx)}
    </div>
  )
}

/**
 * What every node needs to know about the render as a whole. An object rather
 * than positional arguments: this is threaded through every branch of the
 * renderer, and each new flag was another parameter to add in nine places.
 */
interface Ctx {
  streaming: boolean
  liveThinking: boolean
  showReasoning: boolean
  /** True when this node is the last child of the message — see `liveThinking`. */
  isLast?: boolean
}

// Block containers where whitespace between children is just formatting noise.
const BLOCK_CONTAINERS = new Set(["root", "ordered list", "list"])

// While streaming, wrap each visible word in a one-shot fade so newly-arrived
// words ease in. Already-rendered words stay mounted, so they don't re-animate.
function fadeWords(value: string): React.ReactNode {
  return value.split(/(\s+)/).map((part, i) =>
    part.trim() === "" ? (
      part
    ) : (
      <span key={i} className="word-fade">
        {part}
      </span>
    )
  )
}

function renderChildren(node: MElement, ctx: Ctx): React.ReactNode {
  const children = BLOCK_CONTAINERS.has(node.type)
    ? node.children.filter((c) => !(isText(c) && c.value.trim() === ""))
    : node.children

  return children.map((child, i) => (
    <React.Fragment key={i}>
      {renderNode(child, {
        ...ctx,
        // Only meaningful at the top level: a thinking block nested inside a
        // paragraph is not the message's trailing node whatever its index says.
        isLast: node.type === "root" && i === children.length - 1,
      })}
    </React.Fragment>
  ))
}

function renderNode(node: MNode, ctx: Ctx): React.ReactNode {
  if (isText(node)) return ctx.streaming ? fadeWords(node.value) : node.value

  const el = node
  const kids = renderChildren(el, ctx)

  if (el.type.startsWith("heading ")) {
    const level = el.type.slice("heading ".length).trim()
    const Tag = level === "1" ? "h2" : level === "2" ? "h3" : "h4"
    // Sized in `em` so a heading stays proportional to whatever the surface set
    // its body at, and set in the display face — a reply with a title in it is
    // the one place in the app that reads like a page rather than a UI.
    return (
      <Tag
        className={cn(
          "font-display font-semibold tracking-tight",
          // Extra space above, none below: a heading belongs to what follows it.
          "mt-5 -mb-1 first:mt-0",
          level === "1" && "text-[1.5em]",
          level === "2" && "text-[1.25em]",
          level !== "1" && level !== "2" && "text-[1.08em]"
        )}
      >
        {kids}
      </Tag>
    )
  }

  switch (el.type) {
    case "thinking": {
      // Rendered in place, at the point in the reply where the model actually
      // reasoned — not hoisted above the whole message — so text/reasoning/text
      // sequences stay in their original order.
      if (!ctx.showReasoning) return null
      const text = textContent(el).trim()
      return text ? (
        <ReasoningPanel
          text={text}
          streaming={ctx.liveThinking && ctx.isLast === true}
        />
      ) : null
    }
    case "paragraph":
      return <p>{kids}</p>
    case "bold":
      return <strong className="font-semibold">{kids}</strong>
    case "italic":
      return <em className="italic">{kids}</em>
    case "inline":
      return <span className="font-medium text-foreground">{kids}</span>
    case "code inline":
      return (
        <code className="rounded bg-chart-4/10 px-1.5 py-0.5 font-mono text-[0.85em] break-all text-chart-4">
          {textContent(el)}
        </code>
      )
    case "code block":
      return (
        <CodeBlock
          code={stripEdgeNewlines(textContent(el))}
          lang={el.attrs?.lang}
        />
      )
    case "ordered list":
      return (
        <ol className="list-decimal space-y-1 pl-5 marker:text-muted-foreground">
          {kids}
        </ol>
      )
    case "list":
      return (
        <ul className="list-disc space-y-1 pl-5 marker:text-muted-foreground">
          {kids}
        </ul>
      )
    case "item":
      return <li className="pl-1">{kids}</li>
    case "divider":
      return <hr className="border-border" />
    case "link":
      // Bare <|link|>dest<|/link|> carries no href attribute, so the visible
      // text is the destination. Legacy data with an explicit href still wins.
      return (
        <a
          href={el.attrs?.href ?? textContent(el)}
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium break-all text-primary underline underline-offset-2 hover:text-primary/80"
        >
          {kids}
        </a>
      )
    case "quote":
      return (
        <MessageAside tone="quote">
          <blockquote>{kids}</blockquote>
        </MessageAside>
      )
    case "warn":
      return <MessageAside tone="warn">{kids}</MessageAside>
    case "reply": {
      // The id is the element's only content, so it is read as text rather than
      // rendered — there is nothing here to mark up.
      const id = textContent(el).trim()
      return id ? <MessageReplyRef id={id} /> : null
    }
    default:
      // Unknown tag — render its contents so nothing is lost.
      return <>{kids}</>
  }
}

function stripEdgeNewlines(s: string) {
  return s.replace(/^\n/, "").replace(/\n$/, "")
}

function CodeBlock({ code, lang }: { code: string; lang?: string }) {
  const [copied, setCopied] = React.useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard unavailable — ignore.
    }
  }

  return (
    <div className="overflow-hidden rounded-lg border bg-muted/40">
      <div className="flex items-center justify-between border-b bg-muted/60 px-3 py-1.5">
        <span className="font-mono text-xs text-muted-foreground">
          {lang ?? "code"}
        </span>
        <button
          type="button"
          onClick={copy}
          className="flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-3">
        <code className="font-mono text-sm text-foreground">{code}</code>
      </pre>
    </div>
  )
}
