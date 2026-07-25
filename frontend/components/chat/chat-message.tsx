"use client"

import * as React from "react"

import type { ChatMessage } from "@/hooks/use-chat"
import { parseUserMarkdown } from "@/lib/user-parser"
import { RichContent, RichMessage } from "./rich-message"

export function ChatMessageBubble({
  message,
  streaming = false,
}: {
  message: ChatMessage
  streaming?: boolean
}) {
  // User messages use a markdown subset; assistant messages use the model's markup.
  if (message.role === "user") {
    const tree = parseUserMarkdown(message.content)
    return (
      <div className="flex w-full justify-end">
        <div className="min-w-0 max-w-[85%] rounded-2xl bg-card px-4 py-3 text-card-foreground ring-1 ring-border/50">
          <RichContent root={tree} />
        </div>
      </div>
    )
  }

  return (
    <div className="w-full min-w-0">
      <RichMessage content={message.content} streaming={streaming} />
      <Byline message={message} streaming={streaming} />
    </div>
  )
}

// Attribution under a reply: which model wrote it, then what it cost.
//
// Either half can be missing, which is why this isn't one string. A reply loaded from
// history has a model name but no token counts (the counts are tallied client-side as
// tokens arrive and aren't persisted), while a reply that just streamed has counts and
// only gains its name when `onDone` refetches it. Rendering the two independently means
// each shows whatever is actually known.
function Byline({ message, streaming }: { message: ChatMessage; streaming: boolean }) {
  // Nothing is attributed mid-stream: the name isn't known yet and a count that ticks
  // upward every frame reads as noise under a reply that's still being written.
  if (streaming) return null

  const figures: string[] = []
  if (message.tokenCount !== undefined) {
    figures.push(`${message.tokenCount.toLocaleString("en-US")} tokens`)
  }
  if (message.thinkingTokenCount) {
    figures.push(`${message.thinkingTokenCount.toLocaleString("en-US")} thinking`)
  }
  if (!message.modelName && figures.length === 0) return null

  return (
    <p className="mt-2 flex flex-wrap items-baseline gap-x-2 px-1 text-xs text-muted-foreground/70">
      {message.modelName && (
        // The one part that's a name rather than a number, so it carries the weight.
        <span className="font-medium text-muted-foreground">{message.modelName}</span>
      )}
      {figures.map((figure, i) => (
        <React.Fragment key={figure}>
          {(i > 0 || message.modelName) && (
            <span aria-hidden className="text-muted-foreground/35">
              ·
            </span>
          )}
          <span className="tabular-nums">{figure}</span>
        </React.Fragment>
      ))}
    </p>
  )
}
