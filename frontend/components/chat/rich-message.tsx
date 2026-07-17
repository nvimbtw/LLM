"use client"

import * as React from "react"
import { Check, Copy } from "lucide-react"

import { cn } from "@/lib/utils"
import {
  isText,
  parseMessage,
  stripThinking,
  textContent,
  type MElement,
  type MNode,
} from "@/lib/message-parser"

export function RichMessage({
  content,
  streaming = false,
}: {
  content: string
  streaming?: boolean
}) {
  const tree = React.useMemo(() => parseMessage(stripThinking(content)), [content])
  return <RichContent root={tree} streaming={streaming} />
}

/** Renders an already-parsed node tree (model markup or user markdown). */
export function RichContent({
  root,
  streaming = false,
}: {
  root: MElement
  streaming?: boolean
}) {
  return (
    <div className="space-y-3 text-sm leading-relaxed break-words">
      {renderChildren(root, streaming)}
    </div>
  )
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

function renderChildren(node: MElement, streaming: boolean): React.ReactNode {
  const children = BLOCK_CONTAINERS.has(node.type)
    ? node.children.filter((c) => !(isText(c) && c.value.trim() === ""))
    : node.children

  return children.map((child, i) => (
    <React.Fragment key={i}>{renderNode(child, streaming)}</React.Fragment>
  ))
}

function renderNode(node: MNode, streaming: boolean): React.ReactNode {
  if (isText(node)) return streaming ? fadeWords(node.value) : node.value

  const el = node
  const kids = renderChildren(el, streaming)

  if (el.type.startsWith("heading ")) {
    const level = el.type.slice("heading ".length).trim()
    const Tag = level === "1" ? "h2" : level === "2" ? "h3" : "h4"
    return (
      <Tag
        className={cn(
          "font-heading font-semibold",
          level === "1" && "text-xl",
          level === "2" && "text-lg",
          level !== "1" && level !== "2" && "text-base"
        )}
      >
        {kids}
      </Tag>
    )
  }

  switch (el.type) {
    case "thinking":
      // The model's reasoning is surfaced as a "Thinking…" indicator, not text.
      return null
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
        <CodeBlock code={stripEdgeNewlines(textContent(el))} lang={el.attrs?.lang} />
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
      return (
        <a
          href={el.attrs?.href}
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium break-all text-primary underline underline-offset-2 hover:text-primary/80"
        >
          {kids}
        </a>
      )
    case "quote":
      return (
        <blockquote className="rounded-r-md border-l-2 border-chart-2 bg-chart-2/10 px-3 py-2 text-muted-foreground italic">
          {kids}
        </blockquote>
      )
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
