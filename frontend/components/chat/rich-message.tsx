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
    <div className={cn("break-words", className)}>
      {renderChildren(root, ctx)}
    </div>
  )
}

/**
 * Vertical rhythm, carried by the blocks themselves.
 *
 * This used to be one `space-y-3.5` on the container, which set a heading, a
 * paragraph and a list all the same distance apart — so nothing in a reply
 * belonged to anything else, and the heading had to fight the uniform stack with
 * a negative margin to say that it did. Each block now owns its own leading,
 * measured in `em` so the rhythm scales with whatever size the caller set, and
 * dropped to zero when the block opens the message.
 */
const BLOCK = "mt-[0.9em] first:mt-0"

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

function fadeWords(value: string): React.ReactNode {
  return value.split(/(\s+)/).map((part, i) =>
    part.trim() === "" ? part : (
      <span key={i} className="word-glow">
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
    //
    // The steps are wider than they were (1.5/1.25/1.08). Per the type scale in
    // globals.css, a column whose largest text is half again its smallest reads
    // as a form whatever face it is set in; the jump has to be visible for the
    // heading to do any work. Tracking tightens as the size grows, for the same
    // reason it does there — it is a function of size, not a house style.
    return (
      <Tag
        className={cn(
          "font-display font-semibold",
          // Generous space above, and pulled *into* what follows: a heading
          // belongs to the block under it, and the pair should read as one
          // object. The bottom margin is negative because adjacent vertical
          // margins collapse to the larger of the two — a small positive value
          // here would simply lose to the following block's 0.9em and the
          // heading would float equidistant between its neighbours, which is
          // the exact failure this is meant to fix. -0.55em against that 0.9em
          // leaves a real gap of about a third of a line.
          "mt-[1.9em] mb-[-0.55em] first:mt-0",
          level === "1" && "text-[1.75em] tracking-[-0.021em]",
          level === "2" && "text-[1.35em] tracking-[-0.015em]",
          level !== "1" && level !== "2" && "text-[1.12em] tracking-[-0.008em]"
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
      return <p className={BLOCK}>{kids}</p>
    case "bold":
      return <strong className="font-semibold">{kids}</strong>
    case "italic":
      return <em className="italic">{kids}</em>
    case "inline":
      return <span className="font-medium text-foreground">{kids}</span>
    case "code inline":
      // Set in the mono face against a barely-there wash, not in a chart accent.
      // Two words of code used to be the loudest thing in a reply — a hue that
      // means "series 4" in a chart meant "read me first" in a sentence. The face
      // change is what marks it as code; the wash only has to hold it together.
      // Tight horizontal padding with a hair of negative margin: at 0.35em the
      // tint reads as a space before a following comma or period, which in a
      // sentence full of identifiers is most of them.
      return (
        <code className="mx-[-0.04em] rounded-[3px] bg-foreground/[0.055] px-[0.26em] py-[0.12em] font-mono text-[0.875em] break-words">
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
        <ol
          className={cn(
            BLOCK,
            "list-decimal space-y-[0.35em] pl-[1.4em] marker:text-muted-foreground/70 marker:tabular-nums"
          )}
        >
          {kids}
        </ol>
      )
    case "list":
      return (
        <ul
          className={cn(
            BLOCK,
            "list-disc space-y-[0.35em] pl-[1.4em] marker:text-muted-foreground/60"
          )}
        >
          {kids}
        </ul>
      )
    case "item":
      // A sub-list is part of the item that introduces it, so it overrides the
      // block leading it would otherwise inherit and hugs the line above.
      return (
        <li className="pl-[0.2em] [&>ol]:mt-[0.35em] [&>ul]:mt-[0.35em]">
          {kids}
        </li>
      )
    case "divider":
      // A rule is a paragraph break the reader can see, so it wants more air than
      // any other block — it is separating two things, not introducing one.
      return <hr className="my-[2.2em] border-border" />
    case "link":
      // Bare <|link|>dest<|/link|> carries no href attribute, so the visible
      // text is the destination. Legacy data with an explicit href still wins.
      //
      // No weight change: a bolder run mid-sentence is a second signal saying
      // what the underline already says, and it breaks the colour of the line.
      return (
        <a
          href={el.attrs?.href ?? textContent(el)}
          target="_blank"
          rel="noopener noreferrer"
          className="break-words text-primary underline decoration-primary/30 underline-offset-[3px] transition-colors hover:decoration-primary"
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
    case "reply":
      // Dropped, not rendered. The reply-reference feature is gone; the marker
      // keeps its reserved id — removing it from SPECIAL_TOKENS would shift every
      // id after it — and `llm.chat.markup` already forbids the model from
      // emitting one, so this only ever fires on a stored message from before.
      // Its content is a message id, which is meaningless to a reader, so
      // falling through to `default` would print an opaque string mid-reply.
      return null
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

  // One surface and a hairline, where there used to be four devices for one
  // affordance: a border, a tinted header bar, a language label that said "code"
  // when it knew nothing, and a permanently-lit copy button. The label now
  // appears only when a language is actually known, and both it and the button
  // sit quietly over the code rather than in a strip above it.
  //
  // The button is dimmed rather than hidden. Hover is the obvious way to reveal
  // it and the wrong one: on a touch screen there is no hover, and the copy
  // affordance would simply not exist.
  return (
    <div
      className={cn(
        BLOCK,
        "group relative overflow-hidden rounded-(--radius-control-sm) border border-border/70 bg-foreground/[0.035]"
      )}
    >
      <pre className="overflow-x-auto py-3 pr-16 pl-3.5">
        <code className="font-mono text-[0.85em] leading-relaxed text-foreground">
          {code}
        </code>
      </pre>
      {/* Opacity, not mount/unmount, so revealing the button never shifts the
          label beside it. */}
      <div className="pointer-events-none absolute top-2 right-2.5 flex items-center gap-2.5">
        <button
          type="button"
          onClick={copy}
          aria-label={copied ? "Copied" : "Copy code"}
          className="pointer-events-auto text-muted-foreground opacity-45 transition-opacity duration-150 group-hover:opacity-100 focus-visible:opacity-100"
        >
          {copied ? (
            <Check className="size-3.5" />
          ) : (
            <Copy className="size-3.5" />
          )}
        </button>
        {lang && (
          <span className="tag text-muted-foreground/70">
            {lang}
          </span>
        )}
      </div>
    </div>
  )
}
